"""
data_sources.py

Live data fetchers for BTC predictive model.

Currently implemented:
- BTC price history from DIA assetChartPoints (daily).
"""

from __future__ import annotations

import time
from typing import List

import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta


def fetch_btc_price(days: int = 365) -> pd.DataFrame:
    """
    Fetch daily BTC price history using yfinance (BTC-USD).

    Returns DataFrame with:
        - date (datetime, normalized to day)
        - close (float)
    """
    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 7)  # small margin

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

    # Ensure the index is datetime and use it as 'date'
    df_yf = df_yf.sort_index()
    df_yf.index = pd.to_datetime(df_yf.index)

    if "Close" not in df_yf.columns:
        raise RuntimeError(
            f"yfinance BTC-USD data has no 'Close' column. Columns: {df_yf.columns}"
        )

    df = pd.DataFrame({
        "date": df_yf.index.normalize(),
        "close": df_yf["Close"].astype(float),
    })

    # Drop duplicate dates just in case and keep only last `days`
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    if len(df) > days:
        df = df.tail(days).reset_index(drop=True)

    print(
        f"[yfinance BTC] fetched {len(df)} daily rows from "
        f"{df['date'].min().date()} to {df['date'].max().date()}"
    )

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

    # timestamp en segons des de l'epoch (string)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
    df["fear_greed"] = pd.to_numeric(df["value"], errors="coerce")

    df = df[["date", "fear_greed"]].dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(
        f"[FearGreed] fetched {len(df)} rows from "
        f"{df['date'].min().date()} to {df['date'].max().date()}"
    )

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
