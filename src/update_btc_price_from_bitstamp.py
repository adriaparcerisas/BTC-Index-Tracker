"""
update_btc_price_from_bitstamp.py

Utility script to build a daily BTC-USD OHLC dataset from the public
Bitstamp minute data repo:

    https://github.com/ff137/bitstamp-btcusd-minute-data

It:
  - Loads the historical 1-min data (2012-2025) from GitHub
  - Loads the latest daily updates file
  - Concatenates both
  - Resamples to 1D OHLCV
  - Saves data/raw/btc_price_daily.csv

You can run it locally with:

    python src/update_btc_price_from_bitstamp.py

Then the Streamlit app can use data/raw/btc_price_daily.csv as usual.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# --- URLs to the public GitHub repo (Bitstamp BTC/USD 1-min data) --- #

HIST_URL = (
    "https://raw.githubusercontent.com/"
    "ff137/bitstamp-btcusd-minute-data/main/"
    "data/historical/btcusd_bitstamp_1min_2012-2025.csv.gz"
)

UPDATES_URL = (
    "https://raw.githubusercontent.com/"
    "ff137/bitstamp-btcusd-minute-data/main/"
    "data/updates/btcusd_bitstamp_1min_latest.csv"
)

OUTPUT_PATH = Path("data/raw/btc_price_daily.csv")


def load_minute_data() -> pd.DataFrame:
    """
    Load Bitstamp BTC/USD 1-minute OHLCV data from GitHub
    (historical bulk + latest daily updates) and concatenate.

    Expected columns (per repo README):
        timestamp (seconds since epoch)
        open, high, low, close, volume
    """
    print("[bitstamp] Loading historical bulk 1-min data...")
    df_hist = pd.read_csv(HIST_URL, compression="gzip")

    print("[bitstamp] Loading latest 1-min updates...")
    df_updates = pd.read_csv(UPDATES_URL)

    # Concatenate and drop any potential duplicates
    df = pd.concat([df_hist, df_updates], ignore_index=True)

    expected_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Bitstamp data is missing expected columns: {missing}. "
            f"Got columns: {df.columns.tolist()}"
        )

    print(f"[bitstamp] Loaded {len(df):,} minute rows.")
    return df


def to_daily_ohlcv(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 1-min OHLCV data to daily OHLCV.

    Aggregation:
        - open  : first open of the day
        - high  : max high of the day
        - low   : min low of the day
        - close : last close of the day
        - volume: sum of volume of the day
    """
    # Convert timestamp (seconds) to datetime (UTC), then drop timezone
    df = df_minute.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp")

    # 1D resample
    daily = df.resample("1D").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # Drop days with no data (NaN close)
    daily = daily.dropna(subset=["close"])

    # Reset index and build a naive 'date' column
    daily = daily.reset_index()
    daily["date"] = daily["timestamp"].dt.tz_localize(None).dt.normalize()

    # Keep only useful columns
    daily = daily[["date", "open", "high", "low", "close", "volume"]]

    daily = daily.sort_values("date").reset_index(drop=True)

    print(
        f"[bitstamp] Built daily OHLCV with {len(daily):,} rows "
        f"from {daily['date'].min().date()} "
        f"to {daily['date'].max().date()}"
    )

    return daily


def main():
    df_min = load_minute_data()
    df_daily = to_daily_ohlcv(df_min)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_daily.to_csv(OUTPUT_PATH, index=False)

    print(f"[bitstamp] Saved daily BTC price to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
