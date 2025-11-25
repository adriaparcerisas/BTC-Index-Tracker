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

import numpy as np
import pandas as pd

from trend_regime import add_trend_regime_block
from data_sources import (
    fetch_btc_price,
    fetch_activity_index,
    fetch_fear_greed,
    fetch_etf_flows,
    fetch_equity_prices,
)


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
    Build BTC dataset using live APIs (CoinGecko, Blockchain.com, Alternative.me, Farside, yfinance).

    Returns:
        DataFrame ready for modeling & visualization (no file writing).
    """
    if halving_dates is None:
        halving_dates = DEFAULT_HALVING_DATES
    if return_horizons is None:
        return_horizons = DEFAULT_RETURN_HORIZONS
    if trend_horizons is None:
        trend_horizons = DEFAULT_TREND_HORIZONS

    # 1) BTC price (base)
    df_price = fetch_btc_price(days=price_days)
    if df_price.empty:
        raise RuntimeError("Failed to fetch BTC price from CoinGecko.")

    df = df_price.copy()

    # 2) Merge Fear & Greed
    df_fg = fetch_fear_greed()
    if not df_fg.empty:
        df = df.merge(df_fg, on="date", how="left")

    # 3) Merge on-chain activity
    df_act = fetch_activity_index(days=min(price_days, 365 * 3))
    if not df_act.empty:
        df = df.merge(df_act, on="date", how="left")

    # 4) Merge ETF flows
    df_etf = fetch_etf_flows()
    if not df_etf.empty:
        df = df.merge(df_etf, on="date", how="left")

    # 5) Merge MSTR & COIN daily closes
    df_eq = fetch_equity_prices(["MSTR", "COIN"], period="5y")
    if not df_eq.empty:
        # yfinance returns columns exactly as tickers
        df = df.merge(df_eq[["date", "MSTR", "COIN"]], on="date", how="left")

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
