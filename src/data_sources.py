"""
data_sources.py

Live data fetchers for the BTC predictive model.
Currently implemented:

- fetch_btc_price_coindesk: daily BTC price from CoinDesk Data API (futures)
- fetch_fear_greed: Crypto Fear & Greed index from alternative.me

Other fetch_* functions are stubs (return empty DataFrames) so that
build_btc_dataset_live can call them without breaking.

NOTE:
    You must set an environment variable COINDESK_API_KEY
    with your CoinDesk Data API key, or pass api_key explicitly.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import requests
from pathlib import Path


COINDESK_BASE_URL = "https://data-api.coindesk.com"
COINDESK_API_KEY_ENV = "COINDESK_API_KEY"

TWELVEDATA_BASE_URL = "https://api.twelvedata.com/time_series"
TWELVEDATA_ENV_VAR = "TWELVEDATA_API_KEY"


def _get_twelvedata_api_key(explicit_key: str | None = None) -> str:
    """
    Resolve Twelve Data API key from argument or environment.
    """
    key = explicit_key or os.getenv(TWELVEDATA_ENV_VAR)
    if not key:
        raise RuntimeError(
            "Twelve Data API key not found. "
            f"Set the env var {TWELVEDATA_ENV_VAR} or pass it explicitly."
        )
    return key


# ---------------------------------------------------------------------------
# Internal helper for API key
# ---------------------------------------------------------------------------

def _get_coindesk_api_key(explicit_key: Optional[str] = None) -> str:
    """
    Resolve CoinDesk Data API key from argument or environment.

    - Prefer explicit_key if provided
    - Otherwise use env var COINDESK_API_KEY
    """
    key = explicit_key or os.getenv(COINDESK_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            "CoinDesk Data API key not found. Please set the environment "
            f"variable {COINDESK_API_KEY_ENV} or pass api_key explicitly."
        )
    return key


# ---------------------------------------------------------------------------
# BTC price from CoinDesk Data API (Futures – daily OHLCV)
# ---------------------------------------------------------------------------

def fetch_btc_price_coindesk(
    days: int = 3650,
    market: str = "cadli",
    instrument: str = "BTC-USD",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch daily BTC price using CoinDesk Data API (futures_v1_historical_days).

    We use a futures instrument (e.g. BTC-USDT perpetual on Binance) as a
    highly liquid proxy for BTC/USD price.

    Parameters
    ----------
    days : int
        Number of daily candles to fetch (via `limit` parameter).
        For example, 3650 ≈ 10 years.
    market : str
        CoinDesk Data market, e.g. "binance".
    instrument : str
        Futures instrument identifier, e.g. "BTC-USDT-VANILLA-PERPETUAL".
    api_key : str, optional
        CoinDesk Data API key. If None, we read it from COINDESK_API_KEY env var.

    Returns
    -------
    DataFrame with at least:
        - date   (datetime64[ns], normalized to day)
        - close  (float)

    And, when available:
        - open, high, low, volume, quote_volume
    """
    key = _get_coindesk_api_key(api_key)

    # Ensure positive limit (safety)
    limit = max(1, int(days))
    url = f"{COINDESK_BASE_URL}/index/cc/v1/historical/days"
    params = {
        "market": market,
        "instrument": instrument,
        "limit": limit,
        "aggregate": 1,
        "fill": "true",
        "apply_mapping": "true",
        # Ask explicitly for OHLCV groups if needed by your plan
        "groups": "OHLC,VOLUME",
        "api_key": key,
    }
    #url = f"{COINDESK_BASE_URL}/futures/v1/historical/days"
    #params = {
    #    "market": market,
    #    "instrument": instrument,
    #    "limit": limit,
    #    "aggregate": 1,
    #    "fill": "true",
    #    "apply_mapping": "true",
    #    # Ask explicitly for OHLCV groups if needed by your plan
    #    "groups": "OHLC,VOLUME",
    #    "api_key": key,
    #}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch BTC price from CoinDesk Data API: {e}")

    js = resp.json()

    # CoinDesk Data APIs generally return {"Data": [...], "Err": {...}}
    rows = js.get("Data") or js.get("data")
    if not rows:
        raise RuntimeError(f"Unexpected response from CoinDesk Data API: {js}")

    df = pd.DataFrame(rows)

    # --- Parse date ---
    if "TIMESTAMP" in df.columns:
        # TIMESTAMP is usually in seconds since epoch
        df["date"] = pd.to_datetime(df["TIMESTAMP"], unit="s").dt.normalize()
    elif "DATE_TIME_ISO_8601" in df.columns:
        df["date"] = pd.to_datetime(df["DATE_TIME_ISO_8601"]).dt.normalize()
    else:
        raise RuntimeError(
            "Could not find a usable timestamp column in CoinDesk response. "
            f"Columns: {list(df.columns)}"
        )

    # --- Parse price columns ---
    # Close
    if "CLOSE" in df.columns:
        close_col = "CLOSE"
    elif "close" in df.columns:
        close_col = "close"
    elif "VALUE" in df.columns:
        close_col = "VALUE"
    else:
        raise RuntimeError(
            "Could not find CLOSE/close/VALUE column in CoinDesk response. "
            f"Columns: {list(df.columns)}"
        )

    df["close"] = pd.to_numeric(df[close_col], errors="coerce")

    # Optional OHLCV fields
    if "OPEN" in df.columns:
        df["open"] = pd.to_numeric(df["OPEN"], errors="coerce")
    if "HIGH" in df.columns:
        df["high"] = pd.to_numeric(df["HIGH"], errors="coerce")
    if "LOW" in df.columns:
        df["low"] = pd.to_numeric(df["LOW"], errors="coerce")
    if "VOLUME" in df.columns:
        df["volume"] = pd.to_numeric(df["VOLUME"], errors="coerce")
    if "QUOTE_VOLUME" in df.columns:
        df["quote_volume"] = pd.to_numeric(df["QUOTE_VOLUME"], errors="coerce")

    # Basic cleaning
    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    # Select columns in a nice order
    cols = ["date", "close"]
    for c in ["open", "high", "low", "volume", "quote_volume"]:
        if c in df.columns:
            cols.append(c)

    df = df[cols]

    first_date = df["date"].min()
    last_date = df["date"].max()
    print(
        f"[CoinDesk futures] fetched {len(df)} daily rows from "
        f"{first_date.date()} to {last_date.date()}"
    )

    return df


# ---------------------------------------------------------------------------
# Crypto Fear & Greed index (Alternative.me)
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

    # Make sure timestamp is numeric and valid
    if "timestamp" not in df.columns or "value" not in df.columns:
        print(
            "[data_sources] Fear & Greed: missing 'timestamp' or 'value' "
            f"columns. Columns: {list(df.columns)}"
        )
        return pd.DataFrame(columns=["date", "fear_greed"])

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # timestamp is seconds since epoch
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
    Fetch a simple on-chain activity proxy for Bitcoin using Blockchain.com:

        - daily number of transactions ("n-transactions" chart).

    Returns DataFrame with:
        - date
        - btc_tx_count
    """
    url = "https://api.blockchain.info/charts/n-transactions"
    params = {
        "timespan": "all",      # 👈 tota la història disponible
        "format": "json",
        "cors": "true",
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"[data_sources] Failed to fetch activity index: {e}")
        return pd.DataFrame(columns=["date", "btc_tx_count"])

    js = resp.json()
    values = js.get("values", [])
    if not values:
        print("[data_sources] Activity index: empty 'values' list.")
        return pd.DataFrame(columns=["date", "btc_tx_count"])

    df = pd.DataFrame(values)

    # Esperem 'x' (timestamp) i 'y' (value)
    if not {"x", "y"}.issubset(df.columns):
        print(
            "[data_sources] Unexpected activity index schema. "
            f"Columns: {list(df.columns)}"
        )
        return pd.DataFrame(columns=["date", "btc_tx_count"])

    # x: unix seconds
    df["date"] = pd.to_datetime(df["x"], unit="s").dt.normalize()
    df["btc_tx_count"] = pd.to_numeric(df["y"], errors="coerce")

    df = (
        df[["date", "btc_tx_count"]]
        .dropna(subset=["date"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"[ActivityIndex] fetched {len(df)} rows from "
        f"{df['date'].min().date()} to {df['date'].max().date()}"
    )

    return df



def fetch_etf_flows() -> pd.DataFrame:
    """
    Load pre-downloaded Bitcoin spot ETF net flows from a local CSV.

    Expected CSV path (relative to project root):
        data/raw/btc_etf_flows.csv

    Expected columns:
        - Date  (e.g. '11 Jan 2024')
        - Total (daily total net flow, in US$m)

    Returns DataFrame with:
        - date         (datetime.date, normalized)
        - etf_flow_usd (float, net flow in USD)
    """
    # Project root = parent of src/
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "data" / "raw" / "btc_etf_flows.csv"

    if not csv_path.exists():
        print(f"[ETFFlows-CSV] File not found: {csv_path}")
        return pd.DataFrame(columns=["date", "etf_flow_usd"])

    try:
        df_raw = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ETFFlows-CSV] Failed to read CSV: {e}")
        return pd.DataFrame(columns=["date", "etf_flow_usd"])

    # Check columns
    if "Date" not in df_raw.columns or "Total" not in df_raw.columns:
        print(
            "[ETFFlows-CSV] CSV must contain 'Date' and 'Total' columns. "
            f"Columns found: {list(df_raw.columns)}"
        )
        return pd.DataFrame(columns=["date", "etf_flow_usd"])

    df = df_raw.copy()

    # Parse dates (e.g. '11 Jan 2024')
    df["date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True).dt.normalize()

    # Clean numeric "Total" (in US$m)
    df["Total_clean"] = (
        df["Total"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("−", "-", regex=False)  # weird minus sign variants
        .str.replace("–", "-", regex=False)
        .str.replace("(", "-", regex=False)  # (36.9) -> -36.9
        .str.replace(")", "", regex=False)
    )
    df["Total_clean"] = pd.to_numeric(df["Total_clean"], errors="coerce")

    df = df.dropna(subset=["date", "Total_clean"])

    # Convert from millions of USD to USD
    df["etf_flow_usd"] = df["Total_clean"] * 1e6

    df = (
        df[["date", "etf_flow_usd"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    if df.empty:
        print("[ETFFlows-CSV] Resulting ETF flows dataframe is empty after cleaning.")
        return df

    print(
        f"[ETFFlows-CSV] loaded {len(df)} rows from "
        f"{df['date'].min().date()} to {df['date'].max().date()}"
    )

    return df




def fetch_equity_prices(
    tickers: List[str],
    period: str = "5y",
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch daily equity prices (e.g. MSTR, COIN) from Twelve Data.

    Returns a DataFrame with:
        - date
        - one column per ticker (daily close price)
    """
    if not tickers:
        return pd.DataFrame(columns=["date"])

    key = _get_twelvedata_api_key(api_key)

    # Decide date range based on period
    end_dt = datetime.utcnow().date()
    if period == "5y":
        start_dt = end_dt - timedelta(days=365 * 5)
    elif period == "10y":
        start_dt = end_dt - timedelta(days=365 * 10)
    else:  # "max" or anything else
        start_dt = datetime(2010, 1, 1).date()

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    def _fetch_one_ticker(ticker: str) -> pd.DataFrame:
        params = {
            "symbol": ticker,
            "interval": "1day",
            "apikey": key,
            "start_date": start_str,
            "end_date": end_str,
            "order": "ASC",
            "outputsize": 5000,  # generous limit
        }

        try:
            resp = requests.get(TWELVEDATA_BASE_URL, params=params, timeout=20)
            resp.raise_for_status()
        except Exception as e:
            print(f"[Equities-TD] Failed request for {ticker}: {e}")
            return pd.DataFrame(columns=["date", ticker])

        js = resp.json()

        # Check status
        if isinstance(js, dict) and js.get("status") != "ok":
            print(f"[Equities-TD] Non-ok status for {ticker}: {js}")
            return pd.DataFrame(columns=["date", ticker])

        values = js.get("values", [])
        if not values:
            print(f"[Equities-TD] Empty 'values' for {ticker}")
            return pd.DataFrame(columns=["date", ticker])

        df_raw = pd.DataFrame(values)

        # Must have datetime + close
        if "datetime" not in df_raw.columns or "close" not in df_raw.columns:
            print(
                f"[Equities-TD] Unexpected schema for {ticker}. "
                f"Columns: {list(df_raw.columns)}"
            )
            return pd.DataFrame(columns=["date", ticker])

        df = df_raw[["datetime", "close"]].copy()
        df.columns = ["date", ticker]

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df[ticker] = pd.to_numeric(df[ticker], errors="coerce")

        df = (
            df.dropna(subset=["date", ticker])
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        print(
            f"[Equities-TD] {ticker}: {len(df)} rows "
            f"{df['date'].min().date()} → {df['date'].max().date()}"
        )

        return df

    merged: pd.DataFrame | None = None
    for t in tickers:
        df_t = _fetch_one_ticker(t)
        if df_t.empty:
            continue
        if merged is None:
            merged = df_t
        else:
            merged = merged.merge(df_t, on="date", how="outer")

    if merged is None or merged.empty:
        print(f"[Equities-TD] No equity data fetched for tickers {tickers}")
        return pd.DataFrame(columns=["date"])

    merged = merged.sort_values("date").reset_index(drop=True)

    print(
        f"[Equities-TD] Final merged equities df: {len(merged)} rows, "
        f"columns: {list(merged.columns)}"
    )

    return merged



if __name__ == "__main__":
    # Quick local test (if you run this file directly)
    df_test = fetch_btc_price_coindesk(days=30)
    print(df_test.tail())
    df_fg = fetch_fear_greed()
    print(df_fg.tail())
