"""
trend_regime.py

Trend features + regime (bull/bear) and trend-change labels
for the BTC predictive model.

Main entrypoint:

    add_trend_regime_block(
        df,
        price_col="close",
        halving_dates=[...],
        threshold=0.03,
        k=3,
        horizons=[7, 30, 90],
    )

It will add:
- Technical features (returns, moving averages, vol, drawdown, cycle position)
- Regime labels:
    * regime_raw          : based on price_over_ma90 vs threshold
    * regime_smooth       : smoothed with k-day confirmation
    * bull_turn           : day where regime_smooth flips -1 → +1
    * bear_turn           : day where regime_smooth flips +1 → -1
- Future trend-change labels:
    * bull_turn_7d,  bull_turn_30d,  bull_turn_90d
    * bear_turn_7d,  bear_turn_30d,  bear_turn_90d

Where, for example, bull_turn_30d = 1 if there is at least
one bull_turn in the next 30 days (t+1 ... t+30).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1) Base price / technical features
# ---------------------------------------------------------------------


def add_price_trend_features(
    df: pd.DataFrame,
    price_col: str = "close",
    halving_dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Add price-based technical features + cycle position (halvings).

    Requires:
        - df["date"]
        - df[price_col]
    """
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column.")
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain price column '{price_col}'.")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)

    price = df[price_col].astype(float)

    # --- Returns & momentum ---
    df["ret_1d"] = price.pct_change(1)
    df["ret_7d"] = price.pct_change(7)
    df["ret_30d"] = price.pct_change(30)

    # Simple 30d momentum (price / price_30d_ago - 1)
    df["mom_30d"] = price / price.shift(30) - 1

    # --- Moving averages ---
    df["ma_7"] = price.rolling(7, min_periods=1).mean()
    df["ma_30"] = price.rolling(30, min_periods=1).mean()
    df["ma_90"] = price.rolling(90, min_periods=1).mean()

    # --- Relative to MAs ---
    df["price_over_ma30"] = (price - df["ma_30"]) / df["ma_30"]
    df["price_over_ma90"] = (price - df["ma_90"]) / df["ma_90"]
    df["ma_ratio_30_90"] = df["ma_30"] / df["ma_90"] - 1

    # --- Realized volatility (annualized std of daily log-returns) ---
    log_ret = np.log(price / price.shift(1))
    ann = np.sqrt(365)
    df["rv_7d"] = log_ret.rolling(7).std() * ann
    df["rv_30d"] = log_ret.rolling(30).std() * ann

    # --- Drawdown vs last 90d high ---
    roll_max_90 = price.rolling(90, min_periods=1).max()
    df["drawdown_90d"] = price / roll_max_90 - 1

    # --- Halving-based cycle position ---
    if halving_dates:
        halving_dt = pd.to_datetime(halving_dates).sort_values()
        halving_df = pd.DataFrame({"halving_date": halving_dt})

        # Align dtypes for merge_asof
        dates = df[["date"]].copy().sort_values("date")
        last = pd.merge_asof(
            dates,
            halving_df,
            left_on="date",
            right_on="halving_date",
            direction="backward",
        )
        df = df.merge(last, on="date", how="left")

        df["days_since_halving"] = (df["date"] - df["halving_date"]).dt.days
        df["days_since_halving"] = df["days_since_halving"].fillna(-1)

        cycle_len = 4 * 365  # approx. 4-year cycle
        df["cycle_position"] = df["days_since_halving"].clip(lower=0) / cycle_len

        df.drop(columns=["halving_date"], inplace=True, errors="ignore")
    else:
        df["days_since_halving"] = np.nan
        df["cycle_position"] = np.nan

    return df


# ---------------------------------------------------------------------
# 2) Regimes
# ---------------------------------------------------------------------


def compute_regime_raw(df: pd.DataFrame, threshold: float = 0.03) -> pd.Series:
    """
    Raw regime based on price_over_ma90:

        price_over_ma90 >=  threshold → +1 (bull)
        price_over_ma90 <= -threshold → -1 (bear)
        otherwise                      →  0 (neutral)
    """
    if "price_over_ma90" not in df.columns:
        raise ValueError("DataFrame must contain 'price_over_ma90'.")

    po90 = df["price_over_ma90"]
    cond_bull = po90 >= threshold
    cond_bear = po90 <= -threshold

    regime_raw = np.where(cond_bull, 1, np.where(cond_bear, -1, 0))
    return pd.Series(regime_raw, index=df.index, name="regime_raw")


def smooth_regime(regime_raw: pd.Series, k: int = 3) -> pd.Series:
    """
    Smooth the raw regime by requiring k consecutive days before
    confirming a new regime.

    States:
        +1  bull
        -1  bear
         0  no confirmed regime yet (initial phase)
    """
    arr = regime_raw.to_numpy()
    n = len(arr)

    smooth = np.zeros(n, dtype=int)
    cur = 0
    count_bull = 0
    count_bear = 0

    for i in range(n):
        r = arr[i]
        if r == 1:
            count_bull += 1
            count_bear = 0
            if cur != 1 and count_bull >= k:
                cur = 1
        elif r == -1:
            count_bear += 1
            count_bull = 0
            if cur != -1 and count_bear >= k:
                cur = -1
        else:
            # neutral day resets the counters
            count_bull = 0
            count_bear = 0

        smooth[i] = cur

    return pd.Series(smooth, index=regime_raw.index, name="regime_smooth")


# ---------------------------------------------------------------------
# 3) Main entrypoint: add_trend_regime_block
# ---------------------------------------------------------------------


def add_trend_regime_block(
    df: pd.DataFrame,
    price_col: str = "close",
    halving_dates: Optional[List[str]] = None,
    threshold: float = 0.03,
    k: int = 3,
    horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Enrich df with technical features, regimes and future trend-change labels.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least 'date' and price_col.
    price_col : str
        Name of the price column (usually 'close').
    halving_dates : list[str]
        Dates of halvings (YYYY-MM-DD).
    threshold : float
        Threshold on price_over_ma90 for raw regime.
    k : int
        Consecutive days required to confirm a new regime.
    horizons : list[int]
        Horizons in days to build future trend-change labels:
        bull_turn_{h}d, bear_turn_{h}d.

    Returns
    -------
    df : pd.DataFrame
        Original df plus many extra columns.
    """
    if horizons is None:
        horizons = [7, 30, 90]

    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column.")
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain price column '{price_col}'.")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)

    # 1) Technical features + halving context
    df = add_price_trend_features(df, price_col=price_col, halving_dates=halving_dates)

    # 2) Raw & smooth regimes
    df["regime_raw"] = compute_regime_raw(df, threshold=threshold)
    df["regime_smooth"] = smooth_regime(df["regime_raw"], k=k)

    # 3) Instant regime flips (t day)
    reg = df["regime_smooth"]
    df["bull_turn"] = ((reg == 1) & (reg.shift(1) == -1)).astype(int)
    df["bear_turn"] = ((reg == -1) & (reg.shift(1) == 1)).astype(int)

    # 4) Future trend-change labels: within next h days
    n = len(df)
    bull = df["bull_turn"].astype(bool)
    bear = df["bear_turn"].astype(bool)

    for h in horizons:
        future_bull = np.zeros(n, dtype=bool)
        future_bear = np.zeros(n, dtype=bool)

        # For each offset 1..h, mark if a bull/bear turn occurs
        for i in range(1, h + 1):
            future_bull |= bull.shift(-i).fillna(False).to_numpy()
            future_bear |= bear.shift(-i).fillna(False).to_numpy()

        df[f"bull_turn_{h}d"] = future_bull.astype(int)
        df[f"bear_turn_{h}d"] = future_bear.astype(int)

    return df
