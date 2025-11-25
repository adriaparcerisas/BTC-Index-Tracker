# trend_regime.py

from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
import pandas as pd


# ---------- 1. PRICE & TREND FEATURES ---------- #

def add_price_trend_features(
    df: pd.DataFrame,
    price_col: str = "close",
    halving_dates: Optional[List[pd.Timestamp]] = None,
    trading_days_per_year: int = 365,
) -> pd.DataFrame:
    """
    Add price & trend features to a daily DataFrame with a price column.

    Expects:
        df: DataFrame with at least ['date', price_col], sorted by 'date' ascending.
        price_col: name of the closing price column.
        halving_dates: list of halving dates (for BTC). If provided, will create
                       days_since_halving and cycle_position.
        trading_days_per_year: used to annualize volatility (optional).

    Returns:
        df with new feature columns.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    p = df[price_col]

    # Returns
    df["ret_1d"] = np.log(p / p.shift(1))
    df["ret_7d"] = np.log(p / p.shift(7))
    df["ret_30d"] = np.log(p / p.shift(30))

    # Momentum
    df["mom_30d"] = p / p.shift(30) - 1

    # Moving averages
    df["ma_7"] = p.rolling(window=7, min_periods=7).mean()
    df["ma_30"] = p.rolling(window=30, min_periods=30).mean()
    df["ma_90"] = p.rolling(window=90, min_periods=90).mean()

    # Price vs MAs
    df["price_over_ma30"] = p / df["ma_30"] - 1
    df["price_over_ma90"] = p / df["ma_90"] - 1
    df["ma_ratio_30_90"] = df["ma_30"] / df["ma_90"] - 1

    # Realized volatility (optionally annualized)
    rv_7 = df["ret_1d"].rolling(window=7, min_periods=7).std()
    rv_30 = df["ret_1d"].rolling(window=30, min_periods=30).std()

    df["rv_7d"] = rv_7 * np.sqrt(trading_days_per_year)
    df["rv_30d"] = rv_30 * np.sqrt(trading_days_per_year)

    # 90d drawdown
    rolling_max_90 = p.rolling(window=90, min_periods=90).max()
    df["drawdown_90d"] = p / rolling_max_90 - 1

    # Halving-related features (BTC only)
    if halving_dates is not None and len(halving_dates) > 0:
        # Normalize df["date"] to timezone-naive
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

        # Normalize halving dates to timezone-naive (DatetimeIndex -> tz-naive)
        halving_dt = pd.to_datetime(halving_dates, utc=True).tz_localize(None)
        halving_df = (
            pd.DataFrame({"halving_date": halving_dt})
            .sort_values("halving_date")
            .reset_index(drop=True)
        )

        dates = df[["date"]].copy().sort_values("date")

        # Last halving before or on each date (backward merge)
        last_merge = pd.merge_asof(
            dates,
            halving_df,
            left_on="date",
            right_on="halving_date",
            direction="backward",
        )
        last = last_merge["halving_date"]

        # Next halving after or on each date (forward merge)
        next_merge = pd.merge_asof(
            dates,
            halving_df,
            left_on="date",
            right_on="halving_date",
            direction="forward",
        )
        nxt = next_merge["halving_date"]

        # days since last halving & position in cycle
        df["days_since_halving"] = (df["date"] - last).dt.days
        days_between = (nxt - last).dt.days
        df["cycle_position"] = df["days_since_halving"] / days_between.replace(0, np.nan)

        # Before the first halving: set NaN
        df.loc[last.isna(), ["days_since_halving", "cycle_position"]] = np.nan

    return df


# ---------- 2. RAW TREND REGIME ---------- #

def compute_regime_raw(
    df: pd.DataFrame,
    price_over_ma90_col: str = "price_over_ma90",
    ma_ratio_col: str = "ma_ratio_30_90",
    mom_30d_col: str = "mom_30d",
    threshold: float = 0.03,
) -> pd.Series:
    """
    Compute the raw trend regime (bull = +1, bear = -1, neutral = 0) for each day.

    Rules:
        Bullish if:
            price_over_ma90 > threshold
            ma_ratio_30_90 > 0
            mom_30d > 0

        Bearish if:
            price_over_ma90 < -threshold
            ma_ratio_30_90 < 0
            mom_30d < 0

        Neutral otherwise.
    """
    po90 = df[price_over_ma90_col]
    ma_ratio = df[ma_ratio_col]
    mom = df[mom_30d_col]

    bull = (po90 > threshold) & (ma_ratio > 0) & (mom > 0)
    bear = (po90 < -threshold) & (ma_ratio < 0) & (mom < 0)

    regime_raw = pd.Series(0, index=df.index, dtype=int)
    regime_raw[bull] = 1
    regime_raw[bear] = -1

    # Where inputs are NaN, set neutral
    regime_raw[(po90.isna()) | (ma_ratio.isna()) | (mom.isna())] = 0

    return regime_raw


# ---------- 3. REGIME SMOOTHING ---------- #

def smooth_regime(
    regime_raw: pd.Series,
    k: int = 3,
) -> pd.Series:
    """
    Smooth the raw regime series by requiring a new regime to persist
    for k consecutive days before accepting the change.

    Args:
        regime_raw: Series with values in {-1, 0, +1}.
        k: number of consecutive days required to confirm a new regime.

    Returns:
        regime_smooth: Series with the smoothed regime.
    """
    regime_raw = regime_raw.fillna(0).astype(int)
    values = regime_raw.to_numpy()
    n = len(values)
    out = np.zeros(n, dtype=int)

    if n == 0:
        return regime_raw

    current_regime = values[0]
    pending_regime = current_regime
    pending_count = 0
    out[0] = current_regime

    for i in range(1, n):
        raw = values[i]

        if raw == current_regime:
            # No change
            pending_regime = current_regime
            pending_count = 0
            out[i] = current_regime

        elif raw == 0:
            # Neutral: keep current_regime as smoothed regime to avoid noise
            out[i] = current_regime
            pending_regime = current_regime
            pending_count = 0

        else:
            # raw is +1 or -1 and different from current_regime
            if raw == pending_regime:
                pending_count += 1
            else:
                pending_regime = raw
                pending_count = 1

            if pending_count >= k:
                current_regime = pending_regime
                out[i] = current_regime
                pending_count = 0
            else:
                out[i] = current_regime

    return pd.Series(out, index=regime_raw.index, name="regime_smooth")


# ---------- 4. BULL / BEAR TURN EVENTS ---------- #

def compute_turn_events(regime_smooth: pd.Series) -> Dict[str, pd.Series]:
    """
    From the smoothed regime series, compute:
        bull_turn: 1 when there is a switch into bull regime
        bear_turn: 1 when there is a switch into bear regime
    """
    regime = regime_smooth.fillna(0).astype(int)
    prev = regime.shift(1)

    bull_turn = ((regime == 1) & (prev != 1)).astype(int)
    bear_turn = ((regime == -1) & (prev != -1)).astype(int)

    bull_turn.iloc[0] = 0
    bear_turn.iloc[0] = 0

    bull_turn.name = "bull_turn"
    bear_turn.name = "bear_turn"

    return {"bull_turn": bull_turn, "bear_turn": bear_turn}


# ---------- 5. FUTURE TREND-CHANGE TARGETS ---------- #

def compute_turn_targets(
    df: pd.DataFrame,
    horizons: List[int] = [7, 30, 90],
    regime_col: str = "regime_smooth",
    bull_turn_col: str = "bull_turn",
    bear_turn_col: str = "bear_turn",
) -> pd.DataFrame:
    """
    Create binary targets for predicting whether a trend change will occur
    in the next H days (for each H in horizons).

    For each horizon H:
        bull_turn_H[t] = 1 if there is any bull_turn in (t, t+H] and regime at t is not bull.
        bear_turn_H[t] = 1 if there is any bear_turn in (t, t+H] and regime at t is not bear.
        The last H days are set to NaN (no future data).

    Returns:
        df with new columns bull_turn_{H}d and bear_turn_{H}d.
    """
    df = df.copy()
    regime = df[regime_col].astype(float)
    bull_turn = df[bull_turn_col].fillna(0).astype(int)
    bear_turn = df[bear_turn_col].fillna(0).astype(int)

    n = len(df)

    for H in horizons:
        bull_target = pd.Series(np.nan, index=df.index, name=f"bull_turn_{H}d")
        bear_target = pd.Series(np.nan, index=df.index, name=f"bear_turn_{H}d")

        # We can only look ahead up to n - H - 1
        for i in range(0, n - H):
            # If already in bull, set bull_turn_H = 0
            if regime.iat[i] != 1:
                future_bull = bull_turn.iloc[i + 1 : i + H + 1].any()
                bull_target.iat[i] = 1 if future_bull else 0
            else:
                bull_target.iat[i] = 0

            # Similarly for bear
            if regime.iat[i] != -1:
                future_bear = bear_turn.iloc[i + 1 : i + H + 1].any()
                bear_target.iat[i] = 1 if future_bear else 0
            else:
                bear_target.iat[i] = 0

        df[bull_target.name] = bull_target
        df[bear_target.name] = bear_target

    return df


# ---------- 6. FULL TREND & REGIME PIPELINE ---------- #

def add_trend_regime_block(
    df: pd.DataFrame,
    price_col: str = "close",
    halving_dates: Optional[List[pd.Timestamp]] = None,
    threshold: float = 0.03,
    k: int = 3,
    horizons: List[int] = [7, 30, 90],
) -> pd.DataFrame:
    """
    Full pipeline:
        1) Add price & trend features.
        2) Compute regime_raw and regime_smooth.
        3) Compute bull_turn and bear_turn.
        4) Compute future trend-change targets bull_turn_H and bear_turn_H.

    Returns:
        df enriched with all these columns.
    """
    df = df.copy()
    df = add_price_trend_features(df, price_col=price_col, halving_dates=halving_dates)

    df["regime_raw"] = compute_regime_raw(
        df,
        price_over_ma90_col="price_over_ma90",
        ma_ratio_col="ma_ratio_30_90",
        mom_30d_col="mom_30d",
        threshold=threshold,
    )

    df["regime_smooth"] = smooth_regime(df["regime_raw"], k=k)

    turns = compute_turn_events(df["regime_smooth"])
    df["bull_turn"] = turns["bull_turn"]
    df["bear_turn"] = turns["bear_turn"]

    df = compute_turn_targets(
        df,
        horizons=horizons,
        regime_col="regime_smooth",
        bull_turn_col="bull_turn",
        bear_turn_col="bear_turn",
    )

    return df


# ---------- 7. USAGE EXAMPLE ---------- #

if __name__ == "__main__":
    # Example: df with ['date', 'close'] read from a CSV
    df_price = pd.read_csv("btc_price_daily.csv", parse_dates=["date"])

    # Bitcoin halving dates (example)
    halving_dates = [
        "2012-11-28",
        "2016-07-09",
        "2020-05-11",
        "2024-04-20",
    ]

    df_with_trend = add_trend_regime_block(
        df_price,
        price_col="close",
        halving_dates=halving_dates,
        threshold=0.03,
        k=3,
        horizons=[7, 30, 90],
    )

    df_with_trend.to_csv("btc_with_trend_regime.csv", index=False)
    print(df_with_trend.tail())
