"""
update_btc_price_from_bitstamp.py

Utility functions to build a daily BTC-USD OHLC dataset from the public
Bitstamp minute data repo:

    https://github.com/ff137/bitstamp-btcusd-minute-data

Main entrypoints:

- update_btc_daily_from_bitstamp(output_path) -> build & save daily CSV
- ensure_btc_price_daily(output_path, max_age_hours) -> only update if stale
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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


def load_minute_data_from_bitstamp() -> pd.DataFrame:
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


def minute_to_daily_ohlcv(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 1-min OHLCV data to daily OHLCV.

    Aggregation:
        - open  : first open of the day
        - high  : max high of the day
        - low   : min low of the day
        - close : last close of the day
        - volume: sum of volume of the day
    """
    df = df_minute.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp")

    daily = df.resample("1D").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # Drop days with no data
    daily = daily.dropna(subset=["close"])

    daily = daily.reset_index()
    daily["date"] = daily["timestamp"].dt.tz_localize(None).dt.normalize()

    daily = daily[["date", "open", "high", "low", "close", "volume"]]
    daily = daily.sort_values("date").reset_index(drop=True)

    print(
        f"[bitstamp] Built daily OHLCV with {len(daily):,} rows "
        f"from {daily['date'].min().date()} "
        f"to {daily['date'].max().date()}"
    )

    return daily


def update_btc_daily_from_bitstamp(output_path: Path) -> pd.DataFrame:
    """
    Download Bitstamp minute data, build daily OHLCV and save to output_path.

    Returns the daily DataFrame.
    """
    df_min = load_minute_data_from_bitstamp()
    df_daily = minute_to_daily_ohlcv(df_min)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_daily.to_csv(output_path, index=False)

    print(f"[bitstamp] Saved daily BTC price to: {output_path}")
    return df_daily


def ensure_btc_price_daily(
    output_path: Path,
    max_age_hours: int = 24,
) -> None:
    """
    Ensure that `output_path` (daily BTC CSV) exists and is not older
    than `max_age_hours`. If missing or too old, rebuild from Bitstamp.

    Does NOT raise if the update fails; it just prints a message so that
    the app can keep working with the old file if present.
    """
    now = datetime.now(timezone.utc)

    if not output_path.exists():
        print("[bitstamp] Daily BTC CSV not found, rebuilding from Bitstamp...")
        try:
            update_btc_daily_from_bitstamp(output_path)
        except Exception as e:
            print(f"[bitstamp] Failed to build daily BTC price: {e}")
        return

    mtime = datetime.fromtimestamp(output_path.stat().st_mtime, tz=timezone.utc)
    age_hours = (now - mtime).total_seconds() / 3600.0

    if age_hours > max_age_hours:
        print(
            f"[bitstamp] Daily BTC CSV is {age_hours:.1f} hours old "
            f"(> {max_age_hours}), updating..."
        )
        try:
            update_btc_daily_from_bitstamp(output_path)
        except Exception as e:
            print(f"[bitstamp] Failed to update daily BTC price: {e}")
    else:
        print(
            f"[bitstamp] Daily BTC CSV is fresh enough "
            f"({age_hours:.1f} hours old), no update needed."
        )


if __name__ == "__main__":
    # Manual run (optional)
    OUTPUT = Path("data/raw/btc_price_daily.csv")
    ensure_btc_price_daily(OUTPUT, max_age_hours=0)
