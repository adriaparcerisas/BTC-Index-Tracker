"""
modeling.py

Model training utilities for the BTC predictive app.

We provide:

- build_feature_matrix(...)
- fit_all_directional_models(...)

- build_trend_change_feature_matrix(...)
- fit_all_trend_change_models(...)

Each model stores not only accuracy / AUC / Brier, but also
optimal probability thresholds based on F1-score and balanced accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Dataclass for metadata
# ---------------------------------------------------------------------


@dataclass
class ModelMeta:
    horizon_days: int
    n_train: int
    n_test: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    direction: str = "up"  # "up" for directional, "bull"/"bear" for trend-change


# ---------------------------------------------------------------------
# Helper: feature selection
# ---------------------------------------------------------------------


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Select numeric feature columns that are safe to use for modeling.

    - Excludes:
        * 'date'
        * target columns: y_ret_*, up_*
        * trend-change labels: bull_turn_*, bear_turn_*
    """
    ignore_prefixes = ("y_ret_", "up_", "bull_turn_", "bear_turn_")

    feature_cols: List[str] = []
    for c in df.columns:
        if c == "date":
            continue
        if any(c.startswith(p) for p in ignore_prefixes):
            continue

        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    return feature_cols


def _time_based_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    test_size_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Time-based split: train on early data, test on the most recent period.

    We approximate "test_size_days" in rows. If there isn't enough data,
    we fall back to a 80/20 split with a minimum test size.
    """
    n = len(X)
    if n < 60:
        raise ValueError("Not enough observations to create train/test split (<60 rows).")

    # Target test size in rows
    n_test_target = min(test_size_days, int(n * 0.3))
    n_test = max(40, n_test_target)  # ensure decent sample

    if n_test >= n:
        n_test = max(20, n // 3)

    split_idx = n - n_test

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    dates_train = dates.iloc[:split_idx]
    dates_test = dates.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, dates_train, dates_test


def _find_optimal_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray | List[float] | None = None,
) -> Dict[str, float]:
    """
    Scan a grid of thresholds and compute:

        - best_f1_thr, best_f1
        - best_bal_acc_thr, best_bal_acc
        - base_rate (mean of y_true)
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    else:
        thresholds = np.asarray(thresholds)

    best_f1 = -1.0
    best_f1_thr = 0.5
    best_bal_acc = -1.0
    best_bal_acc_thr = 0.5

    for thr in thresholds:
        y_pred = (proba >= thr).astype(int)

        # F1-score (protect against cases with no positives)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        # Update F1
        if (f1 > best_f1 + 1e-9) or (np.isclose(f1, best_f1) and thr < best_f1_thr):
            best_f1 = f1
            best_f1_thr = float(thr)

        # Update balanced accuracy
        if (bal_acc > best_bal_acc + 1e-9) or (
            np.isclose(bal_acc, best_bal_acc) and thr < best_bal_acc_thr
        ):
            best_bal_acc = bal_acc
            best_bal_acc_thr = float(thr)

    base_rate = float(np.mean(y_true)) if len(y_true) > 0 else np.nan

    return {
        "best_f1_thr": best_f1_thr,
        "best_f1": float(best_f1),
        "best_bal_acc_thr": best_bal_acc_thr,
        "best_bal_acc": float(best_bal_acc),
        "base_rate": base_rate,
    }


# ---------------------------------------------------------------------
# Directional models (up / down)
# ---------------------------------------------------------------------


def build_feature_matrix(
    df: pd.DataFrame,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], str]:
    """
    Build feature matrix for directional models (up / down).

    Target:
        up_{horizon}d  (1 if log-return > 0, else 0)

    Returns:
        X: features
        y: binary target
        dates: timestamp index
        feature_cols: list of feature column names
        target_col: name of target column
    """
    target_col = f"up_{horizon}d"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    data = df.dropna(subset=[target_col]).copy()
    data = data.sort_values("date").reset_index(drop=True)

    dates = pd.to_datetime(data["date"])
    y = data[target_col].astype(int)

    feature_cols = _select_feature_columns(data)
    X = data[feature_cols].copy()
    # Fill missing values with forward fill then zeros
    X = X.ffill().bfill().fillna(0.0)

    return X, y, dates, feature_cols, target_col


def fit_all_directional_models(
    df: pd.DataFrame,
    horizons: List[int],
    test_size_days: int = 365,
):
    """
    Train one up/down classifier per horizon.

    Returns:
        models: dict[horizon -> sklearn Pipeline]
        metrics_all: dict[horizon -> metrics dict]
        metas: dict[horizon -> ModelMeta]
    """
    models: Dict[int, Pipeline] = {}
    metrics_all: Dict[int, Dict[str, float]] = {}
    metas: Dict[int, ModelMeta] = {}

    for h in horizons:
        try:
            X, y, dates, feature_cols, target_col = build_feature_matrix(df, horizon=h)
        except Exception as e:
            print(f"[fit_all_directional_models] Skipping horizon {h}d: {e}")
            continue

        if len(X) < 80:
            print(
                f"[fit_all_directional_models] Not enough rows for horizon {h}d "
                f"({len(X)} rows). Skipping."
            )
            continue

        X_train, X_test, y_train, y_test, dates_train, dates_test = (
            _time_based_train_test_split(X, y, dates, test_size_days=test_size_days)
        )

        # Logistic regression with standardization
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)

        proba_test = model.predict_proba(X_test)[:, 1]
        y_pred_default = (proba_test >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred_default)
        try:
            auc = roc_auc_score(y_test, proba_test)
        except ValueError:
            auc = np.nan
        try:
            brier = brier_score_loss(y_test, proba_test)
        except ValueError:
            brier = np.nan

        thr_info = _find_optimal_thresholds(y_test.values, proba_test)

        metrics = {
            "test_accuracy": float(acc),
            "test_auc": float(auc),
            "test_brier": float(brier),
            **thr_info,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "test_start": dates_test.iloc[0],
            "test_end": dates_test.iloc[-1],
        }

        meta = ModelMeta(
            horizon_days=h,
            n_train=len(y_train),
            n_test=len(y_test),
            test_start=dates_test.iloc[0],
            test_end=dates_test.iloc[-1],
            direction="up",
        )

        models[h] = model
        metrics_all[h] = metrics
        metas[h] = meta

        print(
            f"[fit_all_directional_models] h={h}d | "
            f"acc={acc:.3f}, auc={auc:.3f}, brier={brier:.3f}, "
            f"best_bal_thr={thr_info['best_bal_acc_thr']:.2f}, "
            f"best_bal_acc={thr_info['best_bal_acc']:.3f}"
        )

    return models, metrics_all, metas


# ---------------------------------------------------------------------
# Trend-change models (bull / bear)
# ---------------------------------------------------------------------


def build_trend_change_feature_matrix(
    df: pd.DataFrame,
    horizon: int,
    direction: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], str]:
    """
    Build feature matrix for trend-change models.

    direction:
        - "bull" -> target = bull_turn_{horizon}d
        - "bear" -> target = bear_turn_{horizon}d

    Returns:
        X, y, dates, feature_cols, target_col
    """
    direction = direction.lower()
    if direction not in {"bull", "bear"}:
        raise ValueError("direction must be 'bull' or 'bear'.")

    target_col = f"{direction}_turn_{horizon}d"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    data = df.dropna(subset=[target_col]).copy()
    data = data.sort_values("date").reset_index(drop=True)

    dates = pd.to_datetime(data["date"])
    y = data[target_col].astype(int)

    # We reuse the same feature selection logic
    feature_cols = _select_feature_columns(data)
    X = data[feature_cols].copy()
    X = X.ffill().bfill().fillna(0.0)

    return X, y, dates, feature_cols, target_col


def fit_all_trend_change_models(
    df: pd.DataFrame,
    horizons: List[int],
    test_size_days: int = 365,
):
    """
    Train separate models for bull / bear trend-change labels.

    Returns:
        models_by_dir:  dict[direction]['bull'/'bear'][horizon] -> model
        metrics_by_dir: dict[direction][horizon] -> metrics dict
        metas_by_dir:   dict[direction][horizon] -> ModelMeta
    """
    directions = ["bull", "bear"]

    models_by_dir: Dict[str, Dict[int, Pipeline]] = {d: {} for d in directions}
    metrics_by_dir: Dict[str, Dict[int, Dict[str, float]]] = {d: {} for d in directions}
    metas_by_dir: Dict[str, Dict[int, ModelMeta]] = {d: {} for d in directions}

    for direction in directions:
        for h in horizons:
            try:
                X, y, dates, feature_cols, target_col = build_trend_change_feature_matrix(
                    df, horizon=h, direction=direction
                )
            except Exception as e:
                print(
                    f"[fit_all_trend_change_models] Skipping {direction} {h}d: {e}"
                )
                continue

            # Need enough positive events; otherwise model is meaningless
            n_pos = int(y.sum())
            if n_pos < 10:
                print(
                    f"[fit_all_trend_change_models] Too few positive events for "
                    f"{direction} {h}d (only {n_pos}). Skipping."
                )
                continue

            if len(X) < 80:
                print(
                    f"[fit_all_trend_change_models] Not enough rows for "
                    f"{direction} {h}d ({len(X)} rows). Skipping."
                )
                continue

            X_train, X_test, y_train, y_test, dates_train, dates_test = (
                _time_based_train_test_split(
                    X, y, dates, test_size_days=test_size_days
                )
            )

            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=500,
                            n_jobs=-1,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )

            model.fit(X_train, y_train)

            proba_test = model.predict_proba(X_test)[:, 1]
            y_pred_default = (proba_test >= 0.5).astype(int)

            acc = accuracy_score(y_test, y_pred_default)
            try:
                auc = roc_auc_score(y_test, proba_test)
            except ValueError:
                auc = np.nan

            thr_info = _find_optimal_thresholds(y_test.values, proba_test)

            metrics = {
                "test_accuracy": float(acc),
                "test_auc": float(auc),
                **thr_info,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "test_start": dates_test.iloc[0],
                "test_end": dates_test.iloc[-1],
            }

            meta = ModelMeta(
                horizon_days=h,
                n_train=len(y_train),
                n_test=len(y_test),
                test_start=dates_test.iloc[0],
                test_end=dates_test.iloc[-1],
                direction=direction,
            )

            models_by_dir[direction][h] = model
            metrics_by_dir[direction][h] = metrics
            metas_by_dir[direction][h] = meta

            print(
                f"[fit_all_trend_change_models] {direction} h={h}d | "
                f"acc={acc:.3f}, auc={auc:.3f}, "
                f"best_bal_thr={thr_info['best_bal_acc_thr']:.2f}, "
                f"best_bal_acc={thr_info['best_bal_acc']:.3f}"
            )

    return models_by_dir, metrics_by_dir, metas_by_dir

def compute_directional_backtest(
    df: pd.DataFrame,
    horizon: int,
    model,
    metrics: Dict[str, float],
    threshold_type: str = "bal",
):
    """
    Simple long/flat backtest per al model direccional d'un horitzó concret.

    - Opera només a partir de test_start (out-of-sample)
    - Long si proba_up >= threshold, flat si no
    - Usa y_ret_{h}d com a log-return per trade

    Retorna:
        backtest_df: DataFrame amb sèries d'equity i senyals
        stats: dict amb total_return, buyhold_return, cagr, hit_rate, n_trades, threshold
    """
    target_col = f"up_{horizon}d"
    ret_col = f"y_ret_{horizon}d"

    if target_col not in df.columns or ret_col not in df.columns:
        return None, {}

    # Dataset ordenat i net
    data = df.dropna(subset=[target_col, ret_col]).copy()
    data = data.sort_values("date").reset_index(drop=True)

    dates = pd.to_datetime(data["date"])
    y = data[target_col].astype(int)
    rets = data[ret_col].astype(float)

    feature_cols = _select_feature_columns(data)
    X = data[feature_cols].copy()
    X = X.ffill().bfill().fillna(0.0)

    if len(X) < 40:
        return None, {}

    # Definim període de test a partir de metrics['test_start']
    test_start = metrics.get("test_start", None)
    if test_start is not None:
        test_start = pd.to_datetime(test_start)
        mask = dates >= test_start
    else:
        # fallback: últim 30%
        n = len(X)
        split_idx = int(n * 0.7)
        mask = np.zeros(n, dtype=bool)
        mask[split_idx:] = True

    if mask.sum() < 20:
        # fallback si hi ha massa pocs punts
        n = len(X)
        split_idx = int(n * 0.7)
        mask = np.zeros(n, dtype=bool)
        mask[split_idx:] = True

    X_test = X[mask]
    dates_test = dates[mask]
    rets_test = rets[mask]

    if len(X_test) == 0:
        return None, {}

    proba = model.predict_proba(X_test)[:, 1]

    if threshold_type == "f1":
        thr = metrics.get("best_f1_thr", 0.5)
    else:
        thr = metrics.get("best_bal_acc_thr", 0.5)

    thr = float(thr if thr is not None else 0.5)

    # Estratègia long / flat
    pos = (proba >= thr).astype(float)
    strat_log_ret = pos * rets_test.values
    buyhold_log_ret = rets_test.values

    cum_strat = np.cumsum(strat_log_ret)
    cum_buy = np.cumsum(buyhold_log_ret)

    equity_strat = np.exp(cum_strat)
    equity_buy = np.exp(cum_buy)

    backtest_df = pd.DataFrame(
        {
            "date": dates_test.values,
            "proba_up": proba,
            "position": pos,
            "asset_log_ret": rets_test.values,
            "strategy_log_ret": strat_log_ret,
            "strategy_equity": equity_strat,
            "buyhold_equity": equity_buy,
        }
    )

    total_return_strat = float(equity_strat[-1] - 1.0)
    total_return_buy = float(equity_buy[-1] - 1.0)

    n_periods = len(dates_test)
    total_days = horizon * n_periods
    years = total_days / 365.0 if total_days > 0 else np.nan
    if years > 0:
        cagr = (1.0 + total_return_strat) ** (1.0 / years) - 1.0
    else:
        cagr = np.nan

    mask_traded = pos != 0
    if mask_traded.any():
        hit_rate = float(
            (strat_log_ret[mask_traded] > 0).sum() / mask_traded.sum()
        )
    else:
        hit_rate = np.nan

    stats = {
        "total_return": total_return_strat,
        "buyhold_return": total_return_buy,
        "cagr": float(cagr),
        "hit_rate": hit_rate,
        "n_trades": int(mask_traded.sum()),
        "threshold": thr,
    }

    return backtest_df, stats

