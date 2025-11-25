"""
data_sources.py

Live data fetchers for the BTC predictive model.
Currently implemented:

- fetch_btc_price: daily BTC-USD from yfinance
- fetch_fear_greed: Crypto Fear & Greed index from alternative.me

Other fetch_* functions are stubs (return empty DataFrames) so that
build_btc_dataset_live can call them without breaking.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import pandas as pd
import requests
import yfinance as yf


# ---------------------------------------------------------------------------
# BTC price from yfinance (BTC-USD, daily)
# ---------------------------------------------------------------------------

def fetch_btc_price(days: int = 365) -> pd.DataFrame:
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

    # Ensure index is datetime and move it to a column called 'date'
    df_yf = df_yf.sort_index()
    df_yf.index = pd.to_datetime(df_yf.index)
    df_yf.index.name = "date"

    df = df_yf.reset_index()  # now we have a 'date' column

    # Use Close / close robustly
    close_col = None
    if "Close" in df.columns:
        close_col = "Close"
    elif "close" in df.columns:
        close_col = "close"
    else:
        raise RuntimeError(
            f"yfinance BTC-USD data has no 'Close' or 'close' column. "
            f"Columns: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["close"] = pd.to_numeric(df[close_col], errors="coerce")

    df = (
        df[["date", "close"]]
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


# ---------------------------------------------------------------------------
# Crypto Fear & Greed index
# ---------------------------------------------------------------------------

def fetch_fear_greed() -> pd.DataFrame:
    """
    Fetch the Crypto Fear & Greed Index from Alternative.me.

    Returns DataFrame with columns:
        - date (datetime, daily)
        - fear_greed (numeric 0-100)
    """
    url = "https://api.alternative.me/fng/?limit=0&format=json"

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"[data_sources] Failed to fetch Fear & Greed: {e}")
        return pd.DataFrame(columns=["date", "fear_greed"])

    js = resp.json()
    data = js.get("data", [])
    if not data:
        print("[data_sources] Fear & Greed: empty 'data' list.")
        return pd.DataFrame(columns=["date", "fear_greed"])

    df = pd.DataFrame(data)

    # timestamp is seconds since epoch (string/int)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
    df["fear_greed"] = pd.to_numeric(df["value"], errors="coerce")

    df = df[["date", "fear_greed"]].dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(
        f"[FearGreed] fetched {len(df)} rows from "
        f"{df['date'].min().date()} to {df['date'].max().date()}"
    )

    return df


# ---------------------------------------------------------------------------
# Stubs for other factors (return empty DataFrames)
# ---------------------------------------------------------------------------

def fetch_activity_index(days: int = 365) -> pd.DataFrame:
    """
    Placeholder for on-chain activity index.

    Currently returns an empty DataFrame so it doesn't affect the build.
    """
    return pd.DataFrame(columns=["date"])


def fetch_etf_flows() -> pd.DataFrame:
    """
    Placeholder for ETF net flows.

    Currently returns an empty DataFrame so it doesn't affect the build.
    """
    return pd.DataFrame(columns=["date"])


def fetch_equity_prices(tickers: List[str], period: str = "5y") -> pd.DataFrame:
    """
    Placeholder for equity prices (e.g., MSTR, COIN).

    Currently returns an empty DataFrame so it doesn't affect the build.
    """
    return pd.DataFrame(columns=["date"])


if __name__ == "__main__":
    # Quick local test (if you run this file directly)
    df_test = fetch_btc_price(days=30)
    print(df_test.tail())
    df_fg = fetch_fear_greed()
    print(df_fg.tail())
