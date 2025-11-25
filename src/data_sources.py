"""
data_sources.py

Live data fetchers for BTC predictive model:
- BTC price (CoinGecko)
- On-chain activity (Blockchain.info charts)
- Fear & Greed Index (Alternative.me)
- BTC spot ETF net flows (Farside Investors)
"""

import requests
import pandas as pd
import time
from typing import List


def fetch_btc_price(days: int = 365) -> pd.DataFrame:
    """
    Fetch daily closing price for BTC from CoinGecko for last `days` days.
    Returns DataFrame with columns: ['date', 'close']
    """
    url = (
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        f"?vs_currency=usd&days={days}&interval=daily"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    prices = data["prices"]  # list of [timestamp_ms, price]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
    df = df[["date", "price"]].rename(columns={"price": "close"})
    return df


def fetch_activity_index(days: int = 365) -> pd.DataFrame:
    """
    Fetch a couple of on‐chain activity metrics from Blockchain.info charts.
    Returns DataFrame with date + two metrics.
    Example metrics: transactions per second, unique addresses used.
    """
    chart_names = ["n-transactions", "unique-addresses-used"]  # example
    frames = []

    for chart in chart_names:
        url = (
            f"https://api.blockchain.info/charts/{chart}"
            f"?timespan={days}days&format=json&samples={days}"
        )
        resp = requests.get(url)
        resp.raise_for_status()
        js = resp.json()
        series = js["values"]
        df_metric = pd.DataFrame(series)
        df_metric["date"] = pd.to_datetime(df_metric["x"], unit="s").dt.normalize()
        df_metric = df_metric[["date", "y"]].rename(columns={"y": chart})
        frames.append(df_metric)
        time.sleep(1)  # avoid hammering API

    # Merge them
    df = frames[0]
    for other in frames[1:]:
        df = df.merge(other, on="date", how="outer")

    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_fear_greed() -> pd.DataFrame:
    """
    Fetch the Crypto Fear & Greed Index from Alternative.me
    Returns DataFrame with columns: date (datetime), fear_greed (int)
    """
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    resp = requests.get(url)
    resp.raise_for_status()
    js = resp.json()
    data = js["data"]

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.normalize()
    df = df[["date", "value"]].rename(columns={"value": "fear_greed"})
    df["fear_greed"] = pd.to_numeric(df["fear_greed"], errors="coerce")
    return df


def fetch_etf_flows() -> pd.DataFrame:
    """
    Scrape or fetch spot BTC ETF net flows from Farside Investors.
    Returns DataFrame: ['date', 'total_flow_usd', ... other ETF columns if you wish]
    """
    # Example URL (you might need to inspect actual table structure)
    url = "https://farside.co.uk/btc/?format=html"  # update if crossing domains
    resp = requests.get(url)
    resp.raise_for_status()
    df_list = pd.read_html(resp.text)
    # Assume the first table is the flow table
    df = df_list[0]
    # Clean the table accordingly:
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    # Cast all other columns to numeric
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c].str.replace(",","").str.replace("$",""), errors="coerce")
    return df


def fetch_equity_prices(tickers: List[str], period: str = "5y") -> pd.DataFrame:
    """
    Fetch daily close prices for equities using yfinance.
    Returns DataFrame with ['date', ticker1, ticker2, ...]
    """
    import yfinance as yf

    df = yf.download(tickers, period=period, interval="1d", progress=False)["Close"]
    df = df.reset_index().rename(columns={"Date": "date"})
    return df


if __name__ == "__main__":
    # Quick test
    print(fetch_btc_price(30).tail())
    print(fetch_fear_greed().tail())
    # print(fetch_etf_flows().head())
