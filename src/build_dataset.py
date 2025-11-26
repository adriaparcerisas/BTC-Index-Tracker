"""
build_dataset.py

Dataset builders for the BTC predictive model.

Two main entrypoints:

1) build_btc_dataset_from_csv(...)  -> for offline / CLI use (reads local CSV, can save parquet)
2) build_btc_dataset_live(...)      -> for Streamlit app (CoinDesk price + live APIs
                                       for sentiment / other factors)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from trend_regime import add_trend_regime_block
from data_sources import (
    fetch_btc_price_coindesk,
    fetch_activity_index,
    fetch_fear_greed,
    fetch_etf_flows,
    fetch_equity_prices,
)


# ---------- Defaults & paths ---------- #

DEFAULT_HALVING_DATES = [
    "2012-11-28",
    "2016-07-09",
    "2020-05-11",
    "2024-04-20",
]

DEFAULT_RETURN_HORIZONS = [1, 7, 30, 90]
DEFAULT_TREND_HORIZONS = [7, 30, 90]

# Project root is one level above src/
ROOT = Path(__file__).resolve().parents[1]
RAW_PRICE_PATH = ROOT / "data" / "raw" / "btc_price_daily.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "btc_dataset.parquet"


# ---------- Common pieces ---------- #

def load_price_data(path: str) -> pd.DataFrame:
    """
    Load daily BTC price data from CSV.

    Expected columns:
        - date
        - close

    Extra columns like open/high/low/volume are kept as features.
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

    Steps:
        1) Load price CSV (date, close, possibly open/high/low/volume)
        2) Merge Fear & Greed (if available)
        3) Add trend & regime features
        4) Add future return targets
        5) Save to disk (CSV or Parquet)
    """
    if halving_dates is None:
        halving_dates = DEFAULT_HALVING_DATES
    if return_horizons is None:
        return_horizons = DEFAULT_RETURN_HORIZONS
    if trend_horizons is None:
        trend_horizons = DEFAULT_TREND_HORIZONS

    df = load_price_data(price_csv_path)

    # Merge Fear & Greed in offline mode as well
    try:
        df_fg = fetch_fear_greed()
        if df_fg is not None and not df_fg.empty:
            df = df.merge(df_fg, on="date", how="left")
            print(f"[build_btc_dataset_from_csv] merged Fear & Greed: {df_fg.shape}")
    except Exception as e:
        print(f"[build_btc_dataset_from_csv] Fear & Greed fetch/merge failed: {e}")

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


# ---------- 2) LIVE builder (for Streamlit app) ---------- #

def build_btc_dataset_live(
    price_days: Optional[int] = None,
    halving_dates: Optional[List[str]] = None,
    regime_threshold: float = 0.03,
    regime_k: int = 3,
    return_horizons: Optional[List[int]] = None,
    trend_horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Build BTC dataset using CoinDesk Data API for price
    + live APIs (Fear & Greed, on-chain activity, ETFs, equities).

    Parameters
    ----------
    price_days : int or None
        Number of days of history to request from CoinDesk.
        If None, defaults to 3650 (~10 years).
        This is mapped to the `limit` parameter in the API.
    """
    if halving_dates is None:
        halving_dates = DEFAULT_HALVING_DATES
    if return_horizons is None:
        return_horizons = DEFAULT_RETURN_HORIZONS
    if trend_horizons is None:
        trend_horizons = DEFAULT_TREND_HORIZONS

    days = price_days or 3650

    # 1) BTC daily price from CoinDesk Data API (futures)
    try:
        df_price = fetch_btc_price_coindesk(days=days)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch BTC price from CoinDesk Data API: {e}")

    if df_price is None or df_price.empty:
        raise RuntimeError("BTC price DataFrame is empty (CoinDesk Data API).")

    if "close" not in df_price.columns:
        raise RuntimeError(
            f"'close' column not found in BTC price DataFrame. "
            f"Columns: {list(df_price.columns)}"
        )

    df = df_price.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # 2) Merge Fear & Greed (live)
    try:
        df_fg = fetch_fear_greed()
        if df_fg is not None and not df_fg.empty:
            df = df.merge(df_fg, on="date", how="left")
            print(f"[build_btc_dataset_live] merged Fear & Greed: {df_fg.shape}")
    except Exception as e:
        print(f"[build_btc_dataset_live] Fear & Greed fetch/merge failed: {e}")

    # 3) Merge on-chain activity (stub / future)
    try:
        df_act = fetch_activity_index(days=days)
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
        df_eq = fetch_equity_prices(["MSTR", "COIN"], period="max")
        if df_eq is not None and not df_eq.empty:
            cols_to_keep = [c for c in df_eq.columns if c in {"date", "MSTR", "COIN"}]
            df = df.merge(df_eq[cols_to_keep], on="date", how="left")
            print(f"[build_btc_dataset_live] merged equities: {df_eq.shape}")
    except Exception as e:
        print(f"[build_btc_dataset_live] Equity prices fetch/merge failed: {e}")

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
