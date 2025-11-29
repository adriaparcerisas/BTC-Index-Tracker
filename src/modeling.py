"""
modeling.py

Feature matrix builder + directional models for the BTC predictive app.

Interfaces expected by app.py:

- build_feature_matrix(df, horizon) ->
    X, y, dates, feature_cols, target_col

- fit_all_directional_models(df, horizons, test_size_days) ->
    models, metrics_all, metas

- fit_all_trend_change_models(df, horizons, test_size_days) ->
    (currently a placeholder that returns empty dicts)

- build_trend_change_feature_matrix(df, horizon, kind) ->
    (currently a placeholder that returns empty structures)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------
# 1) Feature matrix builder (directional models)
# -------------------------------------------------------------------


def build_feature_matrix(
    df: pd.DataFrame,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], str]:
    """
    Build the feature matrix X and target y for a given horizon (in days).

    Directional target:
        up_{horizon}d = 1 if y_ret_{horizon}d > 0 else 0

    We:
        - sort by date
        - drop rows where target NaN
        - select numeric feature columns, excluding any future-target columns.
    """
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    ret_col = f"y_ret_{horizon}d"
    up_col = f"up_{horizon}d"

    if ret_col not in df.columns or up_col not in df.columns:
        raise ValueError(
            f"Columns '{ret_col}' and/or '{up_col}' not found in DataFrame. "
            "Did you run add_return_targets() in build_dataset?"
        )

    # Keep only rows where the target is defined
    df = df.dropna(subset=[ret_col, up_col]).reset_index(drop=True)

    # ----- Select feature columns -----
    # Start from numeric columns only
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    feature_cols: List[str] = []
    for c in numeric_cols:
        # Drop explicit target columns for any horizon
        if c.startswith("y_ret_") or c.startswith("up_"):
            continue
        # Drop future trend-change labels if present
        if c.startswith("bull_turn_") or c.startswith("bear_turn_"):
            continue

        # Everything else numeric is allowed as a feature:
        # price (close), technicals, regimes, Fear & Greed, tx_count,
        # etf_flow_usd, MSTR, COIN, etc.
        feature_cols.append(c)

    if not feature_cols:
        raise ValueError("No numeric feature columns found after filtering.")

    # Build X, y, dates
    X = df[feature_cols].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    # Simple missing-value strategy: forward-fill then 0
    X = X.ffill().fillna(0.0)

    y = df[up_col].astype(int)
    dates = df["date"]

    return X, y, dates, feature_cols, up_col


# -------------------------------------------------------------------
# 2) Time-based train/test split
# -------------------------------------------------------------------


def _time_based_train_test_split(
    dates: pd.Series,
    test_size_days: int = 365,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a sequence of dates (aligned with X / y),
    return boolean masks for train and test based on a cutoff:

        test_start_date = max(dates) - test_size_days

    If that yields too few test samples, we fall back to a simple
    last-20%-of-samples as test split.

    Returns:
        train_mask, test_mask
    """
    if len(dates) == 0:
        raise ValueError("Empty dates series in _time_based_train_test_split")

    dates = pd.to_datetime(dates)
    max_date = dates.max()
    split_date = max_date - pd.Timedelta(days=test_size_days)

    test_mask = dates >= split_date
    train_mask = ~test_mask

    # Fallback: if test set too small or empty, use last 20% as test
    if test_mask.sum() < 50 or train_mask.sum() < 100:
        n = len(dates)
        split_idx = int(n * 0.8)
        idx = np.arange(n)
        train_mask = idx < split_idx
        test_mask = idx >= split_idx

    return train_mask, test_mask


# -------------------------------------------------------------------
# 3) Directional models (up/down)
# -------------------------------------------------------------------


def fit_all_directional_models(
    df: pd.DataFrame,
    horizons: List[int],
    test_size_days: int = 365,
) -> Tuple[Dict[int, Pipeline], Dict[int, Dict[str, float]], Dict[int, Dict]]:
    """
    Train a separate directional (up/down) model for each horizon in `horizons`.

    For each horizon h:
        - Build feature matrix with build_feature_matrix(df, horizon=h)
        - Make a time-based train/test split (last N days as test)
        - Fit a LogisticRegression model (with scaling)
        - Compute metrics on train & test

    Returns:
        models:      dict[horizon] -> sklearn Pipeline
        metrics_all: dict[horizon] -> dict of metrics
        metas:       dict[horizon] -> dict with meta-information
    """
    models: Dict[int, Pipeline] = {}
    metrics_all: Dict[int, Dict[str, float]] = {}
    metas: Dict[int, Dict] = {}

    for h in horizons:
        try:
            X, y, dates, feature_cols, target_col = build_feature_matrix(df, horizon=h)
        except Exception as e:
            print(f"[modeling] Skipping horizon {h}d due to error in build_feature_matrix: {e}")
            continue

        n_samples = len(X)
        if n_samples < 200:
            print(f"[modeling] Horizon {h}d: not enough samples ({n_samples}), skipping.")
            continue

        # Time-based split
        train_mask, test_mask = _time_based_train_test_split(dates, test_size_days=test_size_days)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        dates_train, dates_test = dates[train_mask], dates[test_mask]

        if len(X_train) < 100 or len(X_test) < 50:
            print(
                f"[modeling] Horizon {h}d: insufficient train/test after time split "
                f"(train={len(X_train)}, test={len(X_test)}), skipping."
            )
            continue

        # Define model: standard scaler + logistic regression
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )

        clf.fit(X_train, y_train)

        # Predictions & metrics
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        # Probabilities for AUC / Brier / app
        if hasattr(clf, "predict_proba"):
            y_train_proba = clf.predict_proba(X_train)[:, 1]
            y_test_proba = clf.predict_proba(X_test)[:, 1]
        else:
            # Should not happen with LogisticRegression, but just in case:
            y_train_proba = np.full_like(y_train, y_train.mean(), dtype=float)
            y_test_proba = np.full_like(y_test, y_test.mean(), dtype=float)

        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        bal_acc_test = balanced_accuracy_score(y_test, y_test_pred)

        # AUC can fail si només hi ha una classe al test
        try:
            if len(np.unique(y_test)) > 1:
                roc_auc = roc_auc_score(y_test, y_test_proba)
            else:
                roc_auc = np.nan
        except Exception:
            roc_auc = np.nan

        # Brier score (calibració de probabilitats)
        try:
            if len(np.unique(y_test)) > 1:
                brier = brier_score_loss(y_test, y_test_proba)
            else:
                brier = np.nan
        except Exception:
            brier = np.nan

        # Diccionari de mètriques amb DOS jocs de claus:
        # - les "nostres"
        # - les que espera app.py: test_accuracy, test_auc, test_brier
        metrics = {
            # General info
            "n_samples": float(n_samples),
            "n_train": float(len(X_train)),
            "n_test": float(len(X_test)),

            # Our original names
            "acc_train": float(acc_train),
            "acc_test": float(acc_test),
            "bal_acc_test": float(bal_acc_test),
            "roc_auc_test": float(roc_auc),
            "brier_test": float(brier) if not np.isnan(brier) else float("nan"),

            # Names expected in app.py
            "train_accuracy": float(acc_train),
            "test_accuracy": float(acc_test),
            "test_auc": float(roc_auc) if not np.isnan(roc_auc) else float("nan"),
            "test_brier": float(brier) if not np.isnan(brier) else float("nan"),
        }

        meta = {
            "horizon": h,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "train_start": str(dates_train.min().date()),
            "train_end": str(dates_train.max().date()),
            "test_start": str(dates_test.min().date()),
            "test_end": str(dates_test.max().date()),
        }

        models[h] = clf
        metrics_all[h] = metrics
        metas[h] = meta

        roc_str = "nan" if np.isnan(roc_auc) else f"{roc_auc:.3f}"
        print(
            f"[modeling] Horizon {h}d: "
            f"train={len(X_train)}, test={len(X_test)}, "
            f"acc_test={acc_test:.3f}, bal_acc_test={bal_acc_test:.3f}, "
            f"roc_auc={roc_str}, brier={metrics['test_brier']:.3f}"
        )

    return models, metrics_all, metas


# -------------------------------------------------------------------
# 4) Trend-change models (placeholders per compatibilitat amb app.py)
# -------------------------------------------------------------------


def fit_all_trend_change_models(
    df: pd.DataFrame,
    horizons: List[int],
    test_size_days: int = 365,
):
    """
    Placeholder so that app.py can import this without error.

    In the future, we can implement models that predict:
        - bull_turn_{horizon}d
        - bear_turn_{horizon}d

    For now, just return empty dicts.
    """
    print("[modeling] fit_all_trend_change_models is not implemented yet; returning empty dicts.")
    return {}, {}, {}


def build_trend_change_feature_matrix(
    df: pd.DataFrame,
    horizon: int,
    kind: str = "bull",
):
    """
    Placeholder for a future feature-matrix builder for trend-change models.

    Intended target columns (if implemented later) would be something like:
        - bull_turn_{horizon}d
        - bear_turn_{horizon}d

    For now, we just return empty structures to keep app.py happy.
    """
    print(
        "[modeling] build_trend_change_feature_matrix is not implemented yet; "
        "returning empty placeholders."
    )
    X = pd.DataFrame()
    y = pd.Series(dtype=int)
    dates = pd.Series(dtype="datetime64[ns]")
    feature_cols: List[str] = []
    target_col = ""
    return X, y, dates, feature_cols, target_col


if __name__ == "__main__":
    # Small smoke test if run locally
    print("modeling.py is a library; import it from app.py or notebooks.")
