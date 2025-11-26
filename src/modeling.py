"""
modeling.py

Feature engineering and baseline models for the BTC predictive project.

Main ideas:
- Use the enriched daily dataset (from build_dataset_live / from_csv).
- Build a feature matrix X and labels y for a given prediction horizon h.
- Train a simple baseline classifier (Logistic Regression) for:
    up_{h}d  (probability that BTC is higher in h days).

Later we can extend this to:
- trend-change labels (bull/bear regime flips)
- more complex models (Random Forest, Gradient Boosting, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Data container for model metadata / diagnostics
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
# Feature engineering
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


def build_feature_matrix(
    df: pd.DataFrame,
    horizon: int,
    include_factors: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], str]:
    """
    Build supervised learning matrix (X, y) for a given horizon.

    Assumes df already contains:
        - 'date'
        - 'close'
        - y_ret_{h}d, up_{h}d  (from add_return_targets)

    Steps:
        1) Add technical features from price.
        2) Optionally include external factors if present (fear_greed, volume, etc.).
        3) Drop rows with NaNs due to rolling windows and targets.

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

    # 3) Build X, y, dates & remove rows with NaNs
    X = df[feature_cols]
    y = df[target_col].astype(float)  # will cast to int later
    dates = df["date"]

    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].astype(int).reset_index(drop=True)
    dates = dates[mask].reset_index(drop=True)

    return X, y, dates, feature_cols, target_col


# ---------------------------------------------------------------------------
# Modeling: directional up/down classifier
# ---------------------------------------------------------------------------

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

    # Safety: if something odd happens
    if X_train.empty or X_test.empty:
        raise ValueError(
            f"Empty train or test split. Check cutoff_date={cutoff_date} "
            f"and the date distribution."
        )

    # Pipeline: standardization + logistic regression
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

    # Predictions & metrics
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
        # If only one class in y_train or y_test, ROC AUC is undefined
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
