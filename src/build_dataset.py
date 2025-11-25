"""
build_dataset.py

Script to build the core BTC dataset for modeling:
- Loads daily BTC price data.
- Adds price & trend features and trend regimes (bull/bear/neutral).
- Adds trend-change targets (bull_turn_H / bear_turn_H).
- Adds multi-horizon return targets and up/down labels.
- Optionally merges external factor CSVs (Fear & Greed, ETF flows, MSTR, COIN).
- Saves the final dataset to disk.

This is v1: only trend-based + whatever external factors you provide via CSV.
Later we will extend it with on-chain, ETF APIs, social APIs, etc.
"""

from __future__ import annotations
import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from pathlib import Path

from trend_regime import add_trend_regime_block

from data_sources import (
    fetch_btc_price,
    fetch_activity_index,
    fetch_fear_greed,
    fetch_etf_flows,
    fetch_equity_prices,
)


# ------------ CONFIG ------------ #

DEFAULT_HALVING_DATES = [
    "2012-11-28",
    "2016-07-09",
    "2020-05-11",
    "2024-04-20",
]

DEFAULT_RETURN_HORIZONS = [1, 7, 30, 90]
DEFAULT_TREND_HORIZONS = [7, 30, 90]


# ------------ UTILS ------------ #

def load_price_data(path: str) -> pd.DataFrame:
    """
    Load daily BTC price data.

    Expected columns:
        - date (ISO format or any parseable date)
        - close (closing price in USD)

    You can extend this to also include open/high/low/volume if needed.
    """
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Input file must contain a 'date' column.")
    if "close" not in df.columns:
        raise ValueError("Input file must contain a 'close' column.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_return_targets(
    df: pd.DataFrame,
    price_col: str = "close",
    horizons: List[int] = None,
) -> pd.DataFrame:
    """
    Add future return targets and up/down labels for multiple horizons.

    For each horizon h:
        y_ret_{h}d = log(P_{t+h} / P_t)
        up_{h}d    = 1 if y_ret_{h}d > 0 else 0

    The last h rows will be NaN for the y_ret_* and up_* targets,
    because we don't have future data there.
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
        # last h entries remain NaN because future_price is NaN

    return df


def merge_optional_csv(
    df: pd.DataFrame,
    path: str,
    date_col: str = "date",
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Merge an optional factor CSV into the main dataset on 'date'.

    - If file does not exist, simply return df unchanged.
    - If exists, must contain a date_col column (default 'date').
    - All other columns can optionally be prefixed (e.g. fg_, etf_, mstr_, coin_).

    Example expected formats:

    data/raw/btc_fear_greed.csv
        date,fear_greed
        2023-01-01,35
        ...

    data/raw/btc_etf_flows.csv
        date,net_flow_usd
        2024-01-11,250000000
        ...

    data/raw/mstr_daily.csv
        date,close
        ...
    """
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"[build_dataset] Skipping optional factor, file not found: {path}")
        return df

    df_factor = pd.read_csv(path_obj)
    if date_col not in df_factor.columns:
        raise ValueError(f"Optional factor file {path} must contain '{date_col}' column.")

    df_factor[date_col] = pd.to_datetime(df_factor[date_col])
    df_factor = df_factor.sort_values(date_col)

    # Apply prefix (if provided) to all non-date columns
    cols = [c for c in df_factor.columns if c != date_col]
    if prefix is not None:
        rename_map = {c: f"{prefix}{c}" for c in cols}
        df_factor = df_factor.rename(columns=rename_map)

    # Left-join on date
    out = df.merge(df_factor, on=date_col, how="left")
    return out


# ------------ MAIN PIPELINE ------------ #

def build_btc_dataset(
    price_csv_path: str,
    output_path: str,
    halving_dates: List[str] = None,
    regime_threshold: float = 0.03,
    regime_k: int = 3,
    return_horizons: List[int] = None,
    trend_horizons: List[int] = None,
    live: bool = False,
) -> pd.DataFrame:
    ...
    # 1) Load prices
    if live:
        df = fetch_btc_price(days=365*5)  # last 5 years, for example
    else:
        df = load_price_data(price_csv_path)
    ...
    # 4) Optional external factors
    if live:
        df_fg = fetch_fear_greed()
        df = df.merge(df_fg, on="date", how="left")

        df_act = fetch_activity_index(days=365*2)
        df = df.merge(df_act, on="date", how="left")

        df_etf = fetch_etf_flows()
        df = df.merge(df_etf, on="date", how="left")

        df_eq = fetch_equity_prices(["MSTR", "COIN"], period="2y")
        # process eq DataFrame to merge ticker closes etc.
        df = df.merge(df_eq[['date','MSTR','COIN']], on="date", how="left")
    #else:
        # existing CSV-merge logic (merge_optional_csv)
    #...
    # (Later we can add more: on-chain CSVs, Google Trends, etc.)

    # 5) Save dataset
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    ext = os.path.splitext(output_path)[1].lower()
    if ext in [".parquet", ".pq"]:
        df.to_parquet(output_path, index=False)
    elif ext in [".csv", ""]:
        # default to CSV if no extension
        if ext == "":
            output_path = output_path + ".csv"
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(
            f"Unsupported output extension '{ext}'. Use .csv or .parquet."
        )

    return df


# ------------ CLI ENTRYPOINT ------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BTC dataset (price, trend, regimes, targets, optional factors)."
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

    build_btc_dataset(
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
