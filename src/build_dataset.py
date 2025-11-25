"""
build_dataset.py

Dataset builders for the BTC predictive model.

Two main entrypoints:

1) build_btc_dataset_from_csv(...)  -> for offline / CLI use (reads local CSV, can save parquet)
2) build_btc_dataset_live(...)      -> for Streamlit app (fetches everything from APIs in real-time)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from trend_regime import add_trend_regime_block
from data_sources import (
    fetch_activity_index,
    fetch_fear_greed,
    fetch_etf_flows,
    fetch_equity_prices,
)


# ---------- Defaults ---------- #

DEFAULT_HALVING_DATES = [
    "2012-11-28",
    "2016-07-09",
    "2020-05-11",
    "2024-04-20",
]

DEFAULT_RETURN_HORIZONS = [1, 7, 30, 90]
DEFAULT_TREND_HORIZONS = [7, 30, 90]


# ---------- Common pieces ---------- #

def load_price_data(path: str) -> pd.DataFrame:
    """
    Load daily BTC price data from CSV.

    Expected columns:
        - date
        - close
    """
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Input file must contain a 'date' column.")
    if "close" not in df.columns:
        raise ValueError("Input file must contain a 'close' column.")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_return_targets(
    df: pd.DataFrame,
    price_col: str = "close",
    horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Add future return targets and up/down labels for multiple horizons.

    For each horizon h:
        y_ret_{h}d = log(P_{t+h} / P_t)
        up_{h}d    = 1 if y_ret_{h}d > 0 else 0
    """
    if horizons is None:
        horizons = DEFAULT_RETURN_HORIZONS

    df = df.copy()
    price = df[price_col]

    for h in horizons:
        future_price = price.shift(-h)
        col_ret = f"y_ret_{h}d"
        col_up = f"up_{h}d"

        df[col_ret] = np.log(future_price / price)
        df[col_up] = np.where(df[col_ret] > 0, 1, 0)

    return df


# ---------- 1) CSV-based builder (offline / CLI) ---------- #

def build_btc_dataset_from_csv(
    price_csv_path: str,
    output_path: str,
    halving_dates: Optional[List[str]] = None,
    regime_threshold: float = 0.03,
    regime_k: int = 3,
    return_horizons: Optional[List[int]] = None,
    trend_horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Build BTC dataset using a local CSV for prices (no live APIs).

    This is useful for offline experiments or batch builds.

    Steps:
        1) Load price CSV (date, close)
        2) Add trend & regime features
        3) Add future return targets
        4) Save to disk (CSV or Parquet)
    """
    if halving_dates is None:
        halving_dates = DEFAULT_HALVING_DATES
    if return_horizons is None:
        return_horizons = DEFAULT_RETURN_HORIZONS
    if trend_horizons is None:
        trend_horizons = DEFAULT_TREND_HORIZONS

    df = load_price_data(price_csv_path)

    # Add trend & regime
    df = add_trend_regime_block(
        df,
        price_col="close",
        halving_dates=halving_dates,
        threshold=regime_threshold,
        k=regime_k,
        horizons=trend_horizons,
    )

    # Add return targets
    df = add_return_targets(df, price_col="close", horizons=return_horizons)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ext = Path(output_path).suffix.lower()
    if ext in [".parquet", ".pq"]:
        df.to_parquet(output_path, index=False)
    else:
        if ext == "":
            output_path = output_path + ".csv"
        df.to_csv(output_path, index=False)

    return df


# ---------- Helper: BTC price from yfinance (LIVE) ---------- #

def _fetch_btc_price_yf(days: int = 365) -> pd.DataFrame:
    """
    Fetch daily BTC price history using yfinance (BTC-USD).

    Returns DataFrame with:
        - date (datetime, normalized to day)
        - close (float)
    """
    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 7)  # small safety margin

    try:
        df_yf = yf.download(
            "BTC-USD",
            start=start,
            end=end + timedelta(days=1),
            interval="1d",
            progress=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to fetch BTC price from yfinance: {e}")

    if df_yf is None or df_yf.empty:
        raise RuntimeError("yfinance returned no data for BTC-USD.")

    # Ensure index is datetime and move it to a column
    df_yf = df_yf.sort_index()
    df_yf.index = pd.to_datetime(df_yf.index)
    df_yf.index.name = "date"

    df_reset = df_yf.reset_index()

    # First column is the former index (date)
    date_col = df_reset.columns[0]
    df_reset["date"] = pd.to_datetime(df_reset[date_col]).dt.normalize()

    # Find the close column robustly
    close_col = None
    for c in df_reset.columns:
        if str(c).lower() == "close":
            close_col = c
            break

    if close_col is None:
        raise RuntimeError(
            f"BTC-USD data has no 'Close' column. Columns: {list(df_reset.columns)}"
        )

    df_reset["close"] = pd.to_numeric(df_reset[close_col], errors="coerce")

    df = (
        df_reset[["date", "close"]]
        .dropna(subset=["date", "close"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Keep only last `days` rows
    if len(df) > days:
        df = df.tail(days).reset_index(drop=True)

    print(
        f"[yfinance BTC] fetched {len(df)} daily rows from "
        f"{df['date'].min().date()} to {df['date'].max().date()}"
    )

    return df


# ---------- 2) LIVE builder (for Streamlit app) ---------- #

def build_btc_dataset_live(
    price_days: int = 365 * 5,
    halving_dates: Optional[List[str]] = None,
    regime_threshold: float = 0.03,
    regime_k: int = 3,
    return_horizons: Optional[List[int]] = None,
    trend_horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Build BTC dataset using live APIs.

    Currently:
      - BTC price from yfinance (BTC-USD)
      - Fear & Greed index (via fetch_fear_greed), if available
      - Other fetch_* functions can be implemented progressively
        (on-chain activity, ETF flows, equities).

    Returns:
        DataFrame ready for modeling & visualization (no file writing).
    """
    if halving_dates is None:
        halving_dates = DEFAULT_HALVING_DATES
    if return_horizons is None:
        return_horizons = DEFAULT_RETURN_HORIZONS
    if trend_horizons is None:
        trend_horizons = DEFAULT_TREND_HORIZONS

    # 1) BTC price (base, live) - using internal yfinance helper
    try:
        df_price = _fetch_btc_price_yf(days=price_days)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch BTC price in live mode: {e}")

    if df_price is None or df_price.empty:
        raise RuntimeError(
            "BTC price DataFrame is empty in live mode. "
            "Check the yfinance BTC-USD fetch implementation."
        )

    df = df_price.copy()

    # 2) Merge Fear & Greed (if implemented)
    try:
        df_fg = fetch_fear_greed()
        if df_fg is not None and not df_fg.empty:
            df = df.merge(df_fg, on="date", how="left")
            print(f"[build_btc_dataset_live] merged Fear & Greed: {df_fg.shape}")
    except Exception as e:
        print(f"[build_btc_dataset_live] Fear & Greed fetch/merge failed: {e}")

    # 3) Merge on-chain activity (stub / future)
    try:
        df_act = fetch_activity_index(days=min(price_days, 365 * 3))
        if df_act is not None and not df_act.empty:
            df = df.merge(df_act, on="date", how="left")
            print(f"[build_btc_dataset_live] merged activity index: {df_act.shape}")
    except Exception as e:
        print(f"[build_btc_dataset_live] Activity index fetch/merge failed: {e}")

    # 4) Merge ETF flows (stub / future)
    try:
        df_etf = fetch_etf_flows()
        if df_etf is not None and not df_etf.empty:
            df = df.merge(df_etf, on="date", how="left")
            print(f"[build_btc_dataset_live] merged ETF flows: {df_etf.shape}")
    except Exception as e:
        print(f"[build_btc_dataset_live] ETF flows fetch/merge failed: {e}")

    # 5) Merge MSTR & COIN daily closes (stub / future)
    try:
        df_eq = fetch_equity_prices(["MSTR", "COIN"], period="5y")
        if df_eq is not None and not df_eq.empty:
            cols_to_keep = [c for c in df_eq.columns if c in {"date", "MSTR", "COIN"}]
            df = df.merge(df_eq[cols_to_keep], on="date", how="left")
            print(f"[build_btc_dataset_live] merged equities: {df_eq.shape}")
    except Exception as e:
        print(f"[build_btc_dataset_live] Equity prices fetch/merge failed: {e}")

    # Ensure sorted & clean
    df = df.sort_values("date").reset_index(drop=True)

    # 6) Trend & regime features
    df = add_trend_regime_block(
        df,
        price_col="close",
        halving_dates=halving_dates,
        threshold=regime_threshold,
        k=regime_k,
        horizons=trend_horizons,
    )

    # 7) Return targets
    df = add_return_targets(df, price_col="close", horizons=return_horizons)

    return df


# ---------- CLI entrypoint (still CSV-based) ---------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BTC dataset from CSV (offline mode).",
    )
    parser.add_argument(
        "--price-csv",
        type=str,
        required=True,
        help="Path to input CSV with BTC daily prices (must contain 'date' and 'close').",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the final dataset (CSV or Parquet).",
    )
    parser.add_argument(
        "--regime-threshold",
        type=float,
        default=0.03,
        help="Threshold for price_over_ma90 when defining raw bull/bear regimes (default: 0.03).",
    )
    parser.add_argument(
        "--regime-k",
        type=int,
        default=3,
        help="Number of consecutive days required to confirm a new trend regime (default: 3).",
    )
    parser.add_argument(
        "--return-horizons",
        type=int,
        nargs="+",
        default=DEFAULT_RETURN_HORIZONS,
        help="List of horizons (in days) for future return targets (default: 1 7 30 90).",
    )
    parser.add_argument(
        "--trend-horizons",
        type=int,
        nargs="+",
        default=DEFAULT_TREND_HORIZONS,
        help="List of horizons (in days) for future trend-change targets (default: 7 30 90).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_btc_dataset_from_csv(
        price_csv_path=args.price_csv,
        output_path=args.output,
        halving_dates=DEFAULT_HALVING_DATES,
        regime_threshold=args.regime_threshold,
        regime_k=args.regime_k,
        return_horizons=args.return_horizons,
        trend_horizons=args.trend_horizons,
    )


if __name__ == "__main__":
    main()
