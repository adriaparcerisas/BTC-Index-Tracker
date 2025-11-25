"""
data_sources.py

Live data fetchers for the BTC predictive model:
- BTC price (CoinGecko)
- On-chain activity (Blockchain.com charts)
- Fear & Greed Index (Alternative.me)
- BTC spot ETF net flows (Farside Investors)
- MSTR & COIN daily prices (yfinance)
"""

from __future__ import annotations

import time
from typing import List, Optional

import pandas as pd
import requests


# ---------- Helper ---------- #

def _safe_get(url: str, params: Optional[dict] = None, timeout: int = 15):
    """Simple wrapper around requests.get with basic error handling."""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"[data_sources] Error fetching {url}: {e}")
        return None


# ---------- BTC PRICE (COINGECKO) ---------- #

def fetch_btc_price(days: int = 365) -> pd.DataFrame:
    """
    Fetch daily BTC price history from DIA (free endpoint).

    Uses:
      GET /v1/assetChartPoints/MAIR120/Bitcoin/0x000...000
      with scale=1d and a [starttime, endtime] window.

    Returns a DataFrame with columns:
        - date (datetime, normalized to day)
        - close (float, DIA 'value' column)
    """
    # Define time window (last `days` days)
    end_ts = int(time.time())
    start_ts = end_ts - days * 24 * 60 * 60

    base_url = (
        "https://api.diadata.org/v1/assetChartPoints/"
        "MAIR120/Bitcoin/0x0000000000000000000000000000000000000000"
    )
    params = {
        "starttime": start_ts,
        "endtime": end_ts,
        "scale": "1d",
    }

    try:
        resp = requests.get(base_url, params=params, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch BTC price from DIA: {e}")

    js = resp.json()
    datapoints = js.get("DataPoints", [])
    if not datapoints:
        raise RuntimeError("DIA response has no 'DataPoints'.")

    series_list = datapoints[0].get("Series", [])
    if not series_list:
        raise RuntimeError("DIA response has no 'Series'.")

    series = series_list[0]
    cols = series.get("columns", [])
    values = series.get("values", [])

    if not values:
        raise RuntimeError("DIA response 'values' list is empty.")

    # Build DataFrame from [columns, values]
    df = pd.DataFrame(values, columns=cols)

    # Expect 'time' and 'value' columns
    if "time" not in df.columns or "value" not in df.columns:
        raise RuntimeError(
            f"Unexpected DIA columns: {df.columns.tolist()}"
        )

    df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    df = df[["date", "value"]].rename(columns={"value": "close"})
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ---------- ON-CHAIN ACTIVITY (BLOCKCHAIN.COM) ---------- #

def fetch_activity_index(days: int = 365 * 3) -> pd.DataFrame:
    """
    Fetch a couple of on-chain activity metrics from Blockchain.com Charts API.

    We use:
        - 'n-transactions' (Confirmed Transactions Per Day)
        - 'n-unique-addresses' (Unique Addresses Used)

    Returns:
        DataFrame with columns: ['date', 'n_transactions', 'n_unique_addresses']
    """
    chart_names = {
        "n-transactions": "n_transactions",
        "n-unique-addresses": "n_unique_addresses",
    }

    frames = []

    for chart_slug, col_name in chart_names.items():
        url = (
            f"https://api.blockchain.info/charts/{chart_slug}"
            f"?timespan={days}days&format=json"
        )
        resp = _safe_get(url)
        if resp is None:
            continue

        js = resp.json()
        series = js.get("values", [])
        if not series:
            continue

        df_metric = pd.DataFrame(series)
        df_metric["date"] = pd.to_datetime(df_metric["x"], unit="s").dt.normalize()
        df_metric = df_metric[["date", "y"]].rename(columns={"y": col_name})
        frames.append(df_metric)

        # small pause to be polite
        time.sleep(1)

    if not frames:
        return pd.DataFrame(columns=["date"])

    df = frames[0]
    for other in frames[1:]:
        df = df.merge(other, on="date", how="outer")

    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------- FEAR & GREED (ALTERNATIVE.ME) ---------- #

def fetch_fear_greed() -> pd.DataFrame:
    """
    Fetch the Crypto Fear & Greed Index from Alternative.me.

    Returns:
        DataFrame with columns: ['date', 'fear_greed']
    """
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    resp = _safe_get(url)
    if resp is None:
        return pd.DataFrame(columns=["date", "fear_greed"])

    js = resp.json()
    data = js.get("data", [])
    if not data:
        return pd.DataFrame(columns=["date", "fear_greed"])

    df = pd.DataFrame(data)
    # timestamp is a string of seconds since epoch
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
    df = df[["date", "value"]].rename(columns={"value": "fear_greed"})
    df["fear_greed"] = pd.to_numeric(df["fear_greed"], errors="coerce")
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


# ---------- ETF FLOWS (FARSIDE) ---------- #

def fetch_etf_flows() -> pd.DataFrame:
    """
    Fetch US spot Bitcoin ETF flows from Farside Investors (all-data table).

    Returns:
        DataFrame with columns: ['date', 'etf_total_flow_usd', ... per-ETF columns]
    """
    url = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
    resp = _safe_get(url)
    if resp is None:
        return pd.DataFrame(columns=["date", "etf_total_flow_usd"])

    try:
        tables = pd.read_html(resp.text)
    except Exception as e:
        print(f"[data_sources] Error parsing ETF flows HTML: {e}")
        return pd.DataFrame(columns=["date", "etf_total_flow_usd"])

    if not tables:
        return pd.DataFrame(columns=["date", "etf_total_flow_usd"])

    df = tables[0].copy()
    # First column is Date
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    # Parse dates (they look like '01 Apr 2024')
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # Clean numeric columns
    for c in df.columns:
        if c == "date":
            continue
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("(", "-", regex=False)
            .str.replace(")", "", regex=False)
            .str.strip()
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Rename Total column if it exists
    if "Total" in df.columns:
        df = df.rename(columns={"Total": "etf_total_flow_usd"})

    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------- EQUITY PRICES (MSTR, COIN) ---------- #

def fetch_equity_prices(
    tickers: List[str],
    period: str = "5y",
) -> pd.DataFrame:
    """
    Fetch daily close prices for a list of equity tickers using yfinance.

    Returns:
        DataFrame with columns: ['date', <ticker1>, <ticker2>, ...]
    """
    import yfinance as yf

    if not tickers:
        return pd.DataFrame(columns=["date"])

    df = yf.download(tickers, period=period, interval="1d", progress=False)["Close"]

    # If single ticker, yfinance returns a Series; convert to DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])

    df = df.reset_index().rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Quick manual tests (you can run locally)
    print(fetch_btc_price(30).tail())
    print(fetch_fear_greed().tail())
    print(fetch_activity_index(30).tail())
    print(fetch_etf_flows().tail())
