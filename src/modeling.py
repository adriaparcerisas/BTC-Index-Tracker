"""
modeling.py

Feature engineering and baseline models for the BTC predictive project.

Two big blocks:

1) Directional models:
   - Target: up_{h}d  (1 if future log-return over h days is > 0, else 0)
   - Horizons: e.g. 1, 7, 30, 90 days.

2) Trend-change models:
   - Targets:
       bull_turn_{h}d = 1 if a bull regime starts in the next h days.
       bear_turn_{h}d = 1 if a bear regime starts in the next h days.
   - Regime is based on `regime_smooth` (from trend_regime.add_trend_regime_block).

We reuse the same feature set (technical + optional factors) for both blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ModelMeta:
    horizon: int
    feature_cols: List[str]
    target_col: str
    n_samples: int
    n_train: int
    n_test: int


# ---------------------------------------------------------------------------
# Common feature engineering
# ---------------------------------------------------------------------------

def _add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical features derived from the daily close price.

    Creates:
        - log_ret_1d, log_ret_3d, log_ret_7d
        - vol_7d, vol_30d
        - ma_20, ma_50, ma_90
        - price_over_ma20/50/90
        - drawdown_90d
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    price = df["close"]

    # Log returns
    df["log_ret_1d"] = np.log(price / price.shift(1))
    df["log_ret_3d"] = np.log(price / price.shift(3))
    df["log_ret_7d"] = np.log(price / price.shift(7))

    # Realized volatility
    df["vol_7d"] = df["log_ret_1d"].rolling(7).std()
    df["vol_30d"] = df["log_ret_1d"].rolling(30).std()

    # Moving averages
    df["ma_20"] = price.rolling(20).mean()
    df["ma_50"] = price.rolling(50).mean()
    df["ma_90"] = price.rolling(90).mean()

    df["price_over_ma20"] = price / df["ma_20"]
    df["price_over_ma50"] = price / df["ma_50"]
    df["price_over_ma90"] = price / df["ma_90"]

    # Drawdown vs 90d rolling max
    roll_max_90 = price.rolling(90).max()
    df["drawdown_90d"] = price / roll_max_90 - 1.0

    return df


def _collect_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Decide which columns to use as features, given a dataframe where
    _add_technical_features has already been applied.
    """
    feature_cols: List[str] = [
        # Pure price/return-based
        "log_ret_1d",
        "log_ret_3d",
        "log_ret_7d",
        "vol_7d",
        "vol_30d",
        "price_over_ma20",
        "price_over_ma50",
        "price_over_ma90",
        "drawdown_90d",
    ]

    # Include any of these columns if present (from trend_regime or data_sources)
    optional_cols = [
        # From trend_regime block (if present)
        "regime_raw",
        "regime_smooth",
        "days_since_last_halving",
        "days_to_next_halving",
        # From OHLC / volume
        "open",
        "high",
        "low",
        "volume",
        "quote_volume",
        # External factors
        "fear_greed",
    ]

    for c in optional_cols:
        if c in df.columns:
            feature_cols.append(c)

    return feature_cols


# ---------------------------------------------------------------------------
# 1) Directional models (up/down)
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    horizon: int,
    include_factors: bool = True,  # kept for API compatibility, currently unused
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], str]:
    """
    Build supervised learning matrix (X, y) for a given horizon.

    Assumes df already contains:
        - 'date'
        - 'close'
        - y_ret_{h}d, up_{h}d  (from add_return_targets)

    Returns
    -------
    X : DataFrame of features
    y : Series of binary labels (up_{h}d)
    dates : Series of dates aligned with X/y
    feature_cols : list of feature column names
    target_col : name of target column used
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    target_col = f"up_{horizon}d"
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Make sure add_return_targets() was applied with horizon {horizon}."
        )

    # 1) Technical features
    df = _add_technical_features(df)

    # 2) Candidate feature columns
    feature_cols: List[str] = _collect_feature_columns(df)

    # 3) Build X, y, dates & remove rows with NaNs
    X = df[feature_cols]
    y = df[target_col].astype(float)  # will cast to int later
    dates = df["date"]

    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].astype(int).reset_index(drop=True)
    dates = dates[mask].reset_index(drop=True)

    return X, y, dates, feature_cols, target_col


def train_directional_model(
    df: pd.DataFrame,
    horizon: int,
    test_size_days: int = 365,
) -> Tuple[Pipeline, Dict[str, float], ModelMeta]:
    """
    Train a baseline directional model for a given horizon.

    Target:
        up_{h}d  (1 if future log-return > 0 over h days, else 0)

    Model:
        - StandardScaler
        - LogisticRegression (balanced class weights)

    Split:
        - Chronological: last `test_size_days` go to test set.
    """
    X, y, dates, feature_cols, target_col = build_feature_matrix(df, horizon=horizon)

    if len(X) < test_size_days * 2:
        raise ValueError(
            f"Not enough data ({len(X)} samples) for a {test_size_days}-day "
            "test window. Reduce test_size_days or collect more history."
        )

    cutoff_date = dates.max() - pd.Timedelta(days=test_size_days)

    train_mask = dates <= cutoff_date
    test_mask = dates > cutoff_date

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    if X_train.empty or X_test.empty:
        raise ValueError(
            f"Empty train or test split. Check cutoff_date={cutoff_date} "
            f"and the date distribution."
        )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=200,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    prob_train = model.predict_proba(X_train)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]

    y_pred_train = (prob_train >= 0.5).astype(int)
    y_pred_test = (prob_test >= 0.5).astype(int)

    metrics: Dict[str, float] = {}
    metrics["train_accuracy"] = float(accuracy_score(y_train, y_pred_train))
    metrics["test_accuracy"] = float(accuracy_score(y_test, y_pred_test))

    try:
        metrics["train_auc"] = float(roc_auc_score(y_train, prob_train))
        metrics["test_auc"] = float(roc_auc_score(y_test, prob_test))
    except ValueError:
        metrics["train_auc"] = float("nan")
        metrics["test_auc"] = float("nan")

    try:
        metrics["train_brier"] = float(brier_score_loss(y_train, prob_train))
        metrics["test_brier"] = float(brier_score_loss(y_test, prob_test))
    except ValueError:
        metrics["train_brier"] = float("nan")
        metrics["test_brier"] = float("nan")

    meta = ModelMeta(
        horizon=horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        n_samples=len(X),
        n_train=len(X_train),
        n_test=len(X_test),
    )

    return model, metrics, meta


def fit_all_directional_models(
    df: pd.DataFrame,
    horizons: List[int] | None = None,
    test_size_days: int = 365,
) -> Tuple[Dict[int, Pipeline], Dict[int, Dict[str, float]], Dict[int, ModelMeta]]:
    """
    Convenience helper: train a directional model for each horizon in `horizons`.

    Returns
    -------
    models : dict[horizon -> sklearn Pipeline]
    metrics : dict[horizon -> dict[str, float]]
    metas : dict[horizon -> ModelMeta]
    """
    if horizons is None:
        horizons = [1, 7, 30, 90]

    models: Dict[int, Pipeline] = {}
    metrics_all: Dict[int, Dict[str, float]] = {}
    metas: Dict[int, ModelMeta] = {}

    for h in horizons:
        print(f"[modeling] Training directional model for horizon {h}d...")
        model, m, meta = train_directional_model(df, horizon=h, test_size_days=test_size_days)
        models[h] = model
        metrics_all[h] = m
        metas[h] = meta

    return models, metrics_all, metas


# ---------------------------------------------------------------------------
# 2) Trend-change models (bull/bear turns)
# ---------------------------------------------------------------------------

def _compute_trend_change_labels(
    df: pd.DataFrame,
    horizon: int,
    regime_col: str = "regime_smooth",
    bull_threshold: float = 0.5,
    bear_threshold: float = -0.5,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute binary labels for trend changes over a given horizon.

    bull_turn_{h}d = 1 if, starting from t, we are NOT in bull
                     and there exists a day in (t, t+h] with regime >= bull_threshold.

    bear_turn_{h}d = 1 if, starting from t, we are NOT in bear
                     and there exists a day in (t, t+h] with regime <= bear_threshold.

    Regime is taken from `regime_col` (typically `regime_smooth`).
    """
    if regime_col not in df.columns:
        raise ValueError(
            f"Column '{regime_col}' not found in dataframe. "
            "Trend-change labels require a regime column (e.g. 'regime_smooth')."
        )

    reg = df[regime_col].values.astype(float)
    n = len(reg)

    y_bull = np.zeros(n, dtype=int)
    y_bear = np.zeros(n, dtype=int)

    for i in range(n):
        if np.isnan(reg[i]):
            continue

        future_start = i + 1
        future_end = min(i + 1 + horizon, n)
        if future_start >= future_end:
            continue

        future_reg = reg[future_start:future_end]

        # Bull turn: we are not already bull, and future goes into bull
        if reg[i] < bull_threshold and np.any(future_reg >= bull_threshold):
            y_bull[i] = 1

        # Bear turn: we are not already bear, and future goes into bear
        if reg[i] > bear_threshold and np.any(future_reg <= bear_threshold):
            y_bear[i] = 1

    bull_col = pd.Series(y_bull, index=df.index, name=f"bull_turn_{horizon}d")
    bear_col = pd.Series(y_bear, index=df.index, name=f"bear_turn_{horizon}d")
    return bull_col, bear_col


def build_trend_change_feature_matrix(
    df: pd.DataFrame,
    horizon: int,
    direction: str = "bull",  # "bull" or "bear"
    regime_col: str = "regime_smooth",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], str]:
    """
    Build X, y for trend-change prediction for a given horizon and direction.

    direction:
        - "bull": predict bull_turn_{h}d
        - "bear": predict bear_turn_{h}d

    Returns
    -------
    X : features
    y : binary labels
    dates : dates aligned with X/y
    feature_cols : feature column names
    target_col : label column name
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # 1) Technical features
    df = _add_technical_features(df)
    feature_cols: List[str] = _collect_feature_columns(df)

    # 2) Trend-change labels
    bull_col, bear_col = _compute_trend_change_labels(
        df,
        horizon=horizon,
        regime_col=regime_col,
    )
    df[bull_col.name] = bull_col
    df[bear_col.name] = bear_col

    if direction == "bull":
        target_col = bull_col.name
    elif direction == "bear":
        target_col = bear_col.name
    else:
        raise ValueError("direction must be 'bull' or 'bear'.")

    X = df[feature_cols]
    y = df[target_col].astype(float)
    dates = df["date"]

    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].astype(int).reset_index(drop=True)
    dates = dates[mask].reset_index(drop=True)

    return X, y, dates, feature_cols, target_col


def train_trend_change_model(
    df: pd.DataFrame,
    horizon: int,
    direction: str = "bull",
    test_size_days: int = 365,
) -> Tuple[Pipeline, Dict[str, float], ModelMeta]:
    """
    Train a baseline trend-change model.

    direction:
        - "bull": predict start of bull regime within h days
        - "bear": predict start of bear regime within h days
    """
    X, y, dates, feature_cols, target_col = build_trend_change_feature_matrix(
        df,
        horizon=horizon,
        direction=direction,
    )

    if len(X) < test_size_days * 2:
        raise ValueError(
            f"Not enough data ({len(X)} samples) for a {test_size_days}-day "
            "test window. Reduce test_size_days or collect more history."
        )

    cutoff_date = dates.max() - pd.Timedelta(days=test_size_days)

    train_mask = dates <= cutoff_date
    test_mask = dates > cutoff_date

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    if X_train.empty or X_test.empty:
        raise ValueError(
            f"Empty train or test split for direction={direction}, horizon={horizon}d. "
            f"Check cutoff_date={cutoff_date} and the date distribution."
        )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=200,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    prob_train = model.predict_proba(X_train)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]

    y_pred_train = (prob_train >= 0.5).astype(int)
    y_pred_test = (prob_test >= 0.5).astype(int)

    metrics: Dict[str, float] = {}
    metrics["train_accuracy"] = float(accuracy_score(y_train, y_pred_train))
    metrics["test_accuracy"] = float(accuracy_score(y_test, y_pred_test))

    try:
        metrics["train_auc"] = float(roc_auc_score(y_train, prob_train))
        metrics["test_auc"] = float(roc_auc_score(y_test, prob_test))
    except ValueError:
        metrics["train_auc"] = float("nan")
        metrics["test_auc"] = float("nan")

    try:
        metrics["train_brier"] = float(brier_score_loss(y_train, prob_train))
        metrics["test_brier"] = float(brier_score_loss(y_test, prob_test))
    except ValueError:
        metrics["train_brier"] = float("nan")
        metrics["test_brier"] = float("nan")

    meta = ModelMeta(
        horizon=horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        n_samples=len(X),
        n_train=len(X_train),
        n_test=len(X_test),
    )

    return model, metrics, meta


def fit_all_trend_change_models(
    df: pd.DataFrame,
    horizons: List[int] | None = None,
    test_size_days: int = 365,
) -> Tuple[
    Dict[str, Dict[int, Pipeline]],
    Dict[str, Dict[int, Dict[str, float]]],
    Dict[str, Dict[int, ModelMeta]],
]:
    """
    Train trend-change models for multiple horizons and both directions.

    Returns
    -------
    models      : dict[direction -> dict[horizon -> model]]
    metrics_all : dict[direction -> dict[horizon -> metrics]]
    metas       : dict[direction -> dict[horizon -> ModelMeta]]
    """
    if horizons is None:
        horizons = [7, 30, 90]  # 1d is usually too short for regime flips

    directions = ["bull", "bear"]

    models: Dict[str, Dict[int, Pipeline]] = {d: {} for d in directions}
    metrics_all: Dict[str, Dict[int, Dict[str, float]]] = {d: {} for d in directions}
    metas: Dict[str, Dict[int, ModelMeta]] = {d: {} for d in directions}

    for d in directions:
        for h in horizons:
            print(f"[modeling] Training {d}-turn model for horizon {h}d...")
            model, m, meta = train_trend_change_model(
                df,
                horizon=h,
                direction=d,
                test_size_days=test_size_days,
            )
            models[d][h] = model
            metrics_all[d][h] = m
            metas[d][h] = meta

    return models, metrics_all, metas
