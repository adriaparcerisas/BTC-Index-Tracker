"""
data_sources.py

Live data fetchers for the BTC predictive model.
Currently implemented:

- fetch_btc_price_coindesk: daily BTC-USD from CoinDesk BPI (historical close)
- fetch_fear_greed: Crypto Fear & Greed index from alternative.me

Other fetch_* functions are stubs (return empty DataFrames) so that
build_btc_dataset_live can call them without breaking.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# BTC price from CoinDesk BPI (historical daily close)
# ---------------------------------------------------------------------------

COINDESK_HIST_URL = "https://api.coindesk.com/v1/bpi/historical/close.json"


def fetch_btc_price_coindesk(
    days: Optional[int] = None,
    currency: str = "USD",
) -> pd.DataFrame:
    """
    Fetch daily BTC close prices from the free CoinDesk BPI historical API.

    Parameters
    ----------
    days : int or None
        If provided, returns only the last `days` days up to today.
        If None, returns the full available history from 2013-09-01.
    currency : str
        Fiat currency code, e.g. 'USD'.

    Returns
    -------
    DataFrame with columns ['date', 'close'].
    """
    today = date.today()

    if days is None:
        # Earliest reasonable start date for CoinDesk BPI
        start = date(2013, 9, 1)
    else:
        # Last `days` days, inclusive of today
        start = today - timedelta(days=days - 1)

    params = {
        "currency": currency,
        "start": start.strftime("%Y-%m-%d"),
        "end": today.strftime("%Y-%m-%d"),
    }

    try:
        resp = requests.get(COINDESK_HIST_URL, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch BTC price from CoinDesk BPI: {e}")

    data = resp.json()
    # Expected structure: { "bpi": { "YYYY-MM-DD": price_float, ... }, ... }
    bpi = data.get("bpi")
    if not isinstance(bpi, dict) or not bpi:
        raise RuntimeError(f"Unexpected response from CoinDesk BPI: {data}")

    # Convert dict of {date_str: price} to DataFrame
    items = sorted(bpi.items())  # list of (date_str, price)
    df = pd.DataFrame(items, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = (
        df.dropna(subset=["date", "close"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"[CoinDesk BPI] fetched {len(df)} daily rows from "
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
    df_test = fetch_btc_price_coindesk(days=30)
    print(df_test.tail())
    df_fg = fetch_fear_greed()
    print(df_fg.tail())
