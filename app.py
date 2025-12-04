import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------
# Path & imports
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from build_dataset import (
    build_btc_dataset_live,
    build_btc_dataset_from_csv,
)
from modeling import (
    fit_all_directional_models,
    build_feature_matrix,
    fit_all_trend_change_models,
    build_trend_change_feature_matrix,
    compute_directional_backtest,  # not used directly but kept for completeness
)

PROCESSED_PATH = Path("data/processed/btc_dataset.parquet")
RAW_PATH = Path("data/raw/btc_price_daily.csv")

# ---------------------------------------------------------------------
# Display / quality thresholds (public view)
# ---------------------------------------------------------------------

# Classifier-level thresholds
MIN_TEST_ACC = 0.60
MIN_AUC = 0.60

# Backtest-level thresholds
MIN_HIT_RATE = 0.60
MIN_TRADES = 25  # mínim de trades perquè la hit-rate tingui sentit


def has_decent_clf_metrics(h: int, metrics_all: Dict) -> bool:
    """Check if a directional model for horizon h has decent test metrics."""
    m = metrics_all.get(h)
    if m is None:
        return False
    return (
        m.get("test_accuracy", 0.0) >= MIN_TEST_ACC
        and m.get("test_auc", 0.0) >= MIN_AUC
    )


def is_strong_directional_model(
    h: int, metrics_all: Dict, bt_cache: Dict[int, Dict]
) -> bool:
    """Combine classifier + backtest quality to decide if a model is robust."""
    if not has_decent_clf_metrics(h, metrics_all):
        return False

    bt = bt_cache.get(h)
    if not bt:
        return False

    conds = [
        bt.get("hit_rate", 0.0) >= MIN_HIT_RATE,
        bt.get("n_trades", 0) >= MIN_TRADES,
        bt.get("sharpe", 0.0) > 0.0,
    ]
    return all(conds)


def is_strong_trend_model(direction: str, horizon: int, trend_metrics: Dict) -> bool:
    """Quality filter for bull/bear trend-change models."""
    m_dir = trend_metrics.get(direction, {})
    m = m_dir.get(horizon)
    if m is None:
        return False

    # AUC decent i un F1 mínim (events són rars → marge generós)
    return (
        m.get("test_auc", 0.0) >= 0.60
        and m.get("best_f1", 0.0) >= 0.25
    )


# ---------------------------------------------------------------------
# Regime helpers
# ---------------------------------------------------------------------

def regime_label_from_value(x: float) -> str:
    """Map numeric regime value to label."""
    if x >= 0.5:
        return "Bull"
    elif x <= -0.5:
        return "Bear"
    else:
        return "Sideways"


def compute_regime_quality(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Valida la utilitat dels règims Bull / Bear / Sideways:
    per cada règim mira el retorn futur a `horizon` dies.
    """
    ret_col = f"y_ret_{horizon}d"
    up_col = f"up_{horizon}d"

    if "regime_smooth" not in df.columns or ret_col not in df.columns:
        return pd.DataFrame()

    df_q = df[["date", "regime_smooth", ret_col]].dropna().copy()
    df_q["regime_label"] = df_q["regime_smooth"].apply(regime_label_from_value)

    # Si tenim la columna "up_{h}d", la fem servir; si no, la derivem
    if up_col in df.columns:
        df_q["up_flag"] = df[up_col].reindex(df_q.index)
    else:
        df_q["up_flag"] = (df_q[ret_col] > 0).astype(float)

    grp = df_q.groupby("regime_label")
    stats = grp.agg(
        n_obs=("date", "count"),
        mean_log_ret=(ret_col, "mean"),
        median_log_ret=(ret_col, "median"),
        up_prob=("up_flag", "mean"),
    ).reset_index()

    # Convertim a % per llegibilitat
    stats["up_prob_%"] = stats["up_prob"] * 100.0
    stats["mean_ret_%"] = (np.exp(stats["mean_log_ret"]) - 1.0) * 100.0
    stats["median_ret_%"] = (np.exp(stats["median_log_ret"]) - 1.0) * 100.0

    return stats[["regime_label", "n_obs", "up_prob_%", "mean_ret_%", "median_ret_%"]]


# ---------------------------------------------------------------------
# MCIS-like composite score helpers
# ---------------------------------------------------------------------

def compute_composite_score(
    df: pd.DataFrame,
    target_col: str,
    factor_config: Dict[str, Dict],
    lam: float = 1.0,
    min_weight: float = 0.05,
    max_lag: int = 2,
):
    """
    Versió generalitzada de l'MCIS:

      - df: DataFrame amb 'date', target_col i factors.
      - target_col: columna de retorn/activitat a predir (ex: 'y_ret_30d').
      - factor_config: dict {factor_name: {"sign": +1/-1}}.
      - lam: regularització ridge.
      - min_weight: pes mínim per factor (abans de normalitzar).
      - max_lag: lag màxim a explorar per cada factor.

    Retorna:
      - score_z: sèrie z-score del composite (index = date)
      - weights: pesos finals per factor
      - lags: millor lag triat per factor
    """
    if "date" not in df.columns:
        raise ValueError("df must contain a 'date' column")

    dfc = df.copy()
    dfc["date"] = pd.to_datetime(dfc["date"])
    dfc = dfc.sort_values("date").set_index("date")

    y = pd.to_numeric(dfc[target_col], errors="coerce")

    # Construïm la matriu X amb els factors
    X_raw = pd.DataFrame(
        {
            name: pd.to_numeric(dfc[name], errors="coerce")
            for name in factor_config.keys()
        },
        index=dfc.index,
    )

    # Selecció de lags (max corr signada amb y)
    lags: Dict[str, int] = {}
    for col, cfg in factor_config.items():
        sign = cfg.get("sign", +1)
        best_L, best = 0, -np.inf
        for L in range(0, max_lag + 1):
            series = X_raw[col].shift(L)
            c = pd.concat([y, series], axis=1).corr().iloc[0, 1]
            if pd.notna(c) and abs(c * sign) > best:
                best, best_L = abs(c * sign), L
        lags[col] = best_L

    # Apliquem lags i signes
    X_lag = pd.DataFrame(
        {
            c: X_raw[c].shift(lags[c]) * factor_config[c].get("sign", +1)
            for c in factor_config.keys()
        },
        index=dfc.index,
    )

    # Z-score de cada driver
    Z = (X_lag - X_lag.mean()) / X_lag.std(ddof=0)

    data = pd.concat([Z, y.rename("y")], axis=1).dropna()
    if data.empty:
        raise ValueError("Not enough non-null data to fit composite score.")

    Zfit, yfit = data[list(factor_config.keys())], data["y"]

    # Ridge: w = (Z'Z + lam I)^(-1) Z'y
    k = Zfit.shape[1]
    XtX = Zfit.T @ Zfit
    w_vec = np.linalg.solve(XtX + lam * np.eye(k), Zfit.T @ yfit)
    w = pd.Series(w_vec, index=factor_config.keys()).clip(lower=0)

    # Floor & normalització
    w = (1 - min_weight * k) * (w / w.sum()) + min_weight
    w = w / w.sum()

    score_raw = (Zfit @ w).rename("MCIS_SCORE")

    score_z = (score_raw - score_raw.mean()) / score_raw.std(ddof=0)
    score_z.index.name = "date"

    return score_z, w, lags


def build_mcis_factor_config(df: pd.DataFrame):
    """
    Intenta detectar automàticament factors per al MCIS-like score
    basant-se en els noms de columna.

    Busquem:
      - ETF net flows
      - Rates / rate probabilities
      - Equity risk / equity index
      - Fear & Greed
      - BTC activity (tx count / activity)

    Retorna:
      factor_config: {colname: {"sign": +1/-1}}
      labels:        {colname: "Nice label"}
    """
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    def pick_by_keywords(keyword_sets):
        # keyword_sets = list of lists of keywords; tries each set
        for keywords in keyword_sets:
            for c in cols:
                name = c.lower()
                if all(k in name for k in keywords):
                    return c
        return None

    factor_config: Dict[str, Dict] = {}
    labels: Dict[str, str] = {}

    # 1) ETF net flows
    col_etf = pick_by_keywords(
        [["etf", "flow"], ["etf", "net"], ["etf", "inflow"], ["etf", "usd"]]
    )
    if col_etf:
        factor_config[col_etf] = {"sign": +1}
        labels[col_etf] = "ETF net flows"

    # 2) Rates probability / rate backdrop
    col_rate = pick_by_keywords(
        [["rate", "prob"], ["rates", "prob"], ["ffr", "prob"], ["rate", "cut"]]
    )
    if col_rate:
        factor_config[col_rate] = {"sign": +1}
        labels[col_rate] = "Rates backdrop"

    # 3) Equity risk / equity index
    col_equity = pick_by_keywords(
        [["equity", "risk"], ["equity", "index"], ["sp500"], ["s&p"], ["eq_risk"]]
    )
    if col_equity:
        factor_config[col_equity] = {"sign": +1}
        labels[col_equity] = "Equity risk"

    # 4) Fear & Greed index
    col_fg = pick_by_keywords(
        [["fear", "greed"], ["fear_greed"], ["fng_index"], ["sentiment_index"]]
    )
    if col_fg:
        factor_config[col_fg] = {"sign": +1}
        labels[col_fg] = "Fear & Greed"

    # 5) BTC on-chain activity (tx count / activity)
    col_activity = pick_by_keywords(
        [["tx_count"], ["txs"], ["transactions"], ["activity", "btc"]]
    )
    if col_activity:
        factor_config[col_activity] = {"sign": +1}
        labels[col_activity] = "BTC activity"

    return factor_config, labels


# ---------------------------------------------------------------------
# Dataset loader (live + offline fallback), cached
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def load_dataset(use_live: bool = True):
    """
    Load or build the BTC dataset.

    Returns:
        df: DataFrame
        source: "live", "offline" or "none"
    """
    # 1) LIVE MODE
    if use_live:
        try:
            df = build_btc_dataset_live()
            df["date"] = pd.to_datetime(df["date"])
            return df, "live"
        except Exception as e:
            st.warning(
                "Live mode failed while building the dataset.\n\n"
                f"Error: `{e}`\n\n"
                "Using the offline CSV-based dataset instead."
            )

    # 2) OFFLINE MODE from local CSV
    if RAW_PATH.exists():
        df = build_btc_dataset_from_csv(
            price_csv_path=str(RAW_PATH),
            output_path=str(PROCESSED_PATH),
        )
        df["date"] = pd.to_datetime(df["date"])
        return df, "offline"

    st.error(
        "No BTC daily price CSV found at data/raw/btc_price_daily.csv "
        "and live mode failed. Cannot build dataset."
    )
    return pd.DataFrame(), "none"


# ---------------------------------------------------------------------
# Model training (cached)
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def train_all_models(df: pd.DataFrame):
    """
    Train directional models for 1d / 7d / 30d / 90d horizons.
    """
    return fit_all_directional_models(
        df,
        horizons=[1, 7, 30, 90],
        test_size_days=365,
    )


@st.cache_resource(show_spinner=False)
def train_all_trend_models(df: pd.DataFrame):
    """
    Train bull/bear trend-change models for horizons 7, 30, 90 days.
    """
    return fit_all_trend_change_models(
        df,
        horizons=[7, 30, 90],
        test_size_days=365,
    )


# ---------------------------------------------------------------------
# Long/flat backtest helper
# ---------------------------------------------------------------------

def run_long_flat_backtest(
    df: pd.DataFrame,
    horizon: int,
    model,
    threshold: float = 0.55,
    cost_bps_per_side: float = 0.0,
    test_days: int = 365,
):
    """
    Long/flat backtest utilitzant el model direccional amb passos NO solapats.

    Cada pas representa un bloc de `horizon` dies:
      - Retorn del bloc: y_ret_{h}d = log(P_{t+h} / P_t)
      - Si P(up) >= threshold -> LONG durant aquest bloc; si no -> FLAT (cash)
      - Buy & hold acumula TOTS els blocs igualment (sense filtrar per senyal).

    Aquí els retorns provenen explícitament de la columna `y_ret_{h}d`
    (log-returns), no pas del target de classificació.
    """

    ret_col = f"y_ret_{horizon}d"
    if ret_col not in df.columns:
        return {}

    # 1) Matriu de features i dates associades al model per aquest horitzó
    X_all, _, dates_all, feature_cols, target_col = build_feature_matrix(
        df, horizon=horizon
    )
    if X_all is None or X_all.empty:
        return {}

    # Ens assegurem que les dates estiguin en ordre creixent
    dates_all = pd.to_datetime(dates_all)
    order = np.argsort(dates_all.values)
    dates_all = dates_all.iloc[order].reset_index(drop=True)
    X_all = X_all.iloc[order].reset_index(drop=True)

    # 2) Sèrie de log-returns y_ret_{h}d alineada per data
    df_ret = df.copy()
    df_ret["date"] = pd.to_datetime(df_ret["date"])
    s_ret = df_ret.set_index("date")[ret_col].astype(float)

    # Reindexem perquè cada fila de X_all tingui el seu log-return futur
    logret_all = s_ret.reindex(dates_all)

    # Eliminem qualsevol fila sense retorn definit
    mask = np.isfinite(logret_all.values)
    if mask.sum() < 3 * horizon:
        return {}

    X_all = X_all.loc[mask].reset_index(drop=True)
    dates_all = dates_all.loc[mask].reset_index(drop=True)
    logret_all = logret_all.values[mask]

    # 3) Definim finestra temporal de test (últims `test_days` + marge horizon)
    last_date = dates_all.iloc[-1]
    test_start_date = last_date - pd.Timedelta(days=test_days + horizon)

    idx_start = int(
        np.searchsorted(dates_all.values, np.datetime64(test_start_date))
    )
    if idx_start >= len(dates_all) - horizon:
        return {}

    # 4) Índex NO solapats: i, i+h, i+2h, ...
    indices = list(range(idx_start, len(dates_all) - horizon, horizon))
    if len(indices) < 2:
        return {}

    X_test = X_all.iloc[indices]
    dates_test = dates_all.iloc[indices]
    logret_test = logret_all[indices]

    # 5) Probabilitats i posicions (1 = long, 0 = flat)
    proba_up = model.predict_proba(X_test)[:, 1]
    signal = (proba_up >= threshold).astype(int)

    position = pd.Series(signal, index=dates_test, name="position")
    ret_h = pd.Series(logret_test, index=dates_test, name="log_ret_h")

    # 6) Retorns de l'estratègia (log) bloc a bloc
    strategy_ret = position * ret_h

    # Costos de trading (round-trip 2 * cost per side)
    n_trades = int((position == 1).sum())
    if cost_bps_per_side > 0 and n_trades > 0:
        cost = cost_bps_per_side / 10000.0
        log_cost_roundtrip = np.log(1.0 - 2.0 * cost)
        strategy_ret.loc[position == 1] += log_cost_roundtrip

    # 7) Equity (normalitzada a 1)
    equity_strategy = np.exp(strategy_ret.cumsum())
    equity_buyhold = np.exp(ret_h.cumsum())

    total_ret_strategy = float(equity_strategy.iloc[-1] - 1.0)
    total_ret_buyhold = float(equity_buyhold.iloc[-1] - 1.0)

    # 8) Mètriques anualitzades
    steps_per_year = 365.0 / float(horizon)

    mean_step_ret = float(strategy_ret.mean())
    cagr_strategy = np.exp(mean_step_ret * steps_per_year) - 1.0

    if n_trades > 0:
        hits = int((ret_h[position == 1] > 0).sum())
        hit_rate = hits / n_trades
    else:
        hit_rate = np.nan

    step_vol = float(strategy_ret.std())
    if step_vol > 0:
        ann_vol = step_vol * np.sqrt(steps_per_year)
        sharpe = (mean_step_ret * steps_per_year) / ann_vol
    else:
        ann_vol = np.nan
        sharpe = np.nan

    # 9) Max drawdown helper
    def _max_drawdown(equity: pd.Series) -> float:
        running_max = equity.cummax()
        dd = equity / running_max - 1.0
        return float(dd.min())  # negatiu (ex: -0.35 = -35%)

    max_dd_strategy = _max_drawdown(equity_strategy)
    max_dd_buyhold = _max_drawdown(equity_buyhold)

    return {
        "dates": dates_test,
        "strategy_equity": equity_strategy,
        "buyhold_equity": equity_buyhold,
        "total_return_strategy": total_ret_strategy,
        "total_return_buyhold": total_ret_buyhold,
        "cagr_strategy": cagr_strategy,
        "hit_rate": hit_rate,
        "n_trades": n_trades,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd_strategy": max_dd_strategy,
        "max_dd_buyhold": max_dd_buyhold,
        "threshold": threshold,
        "cost_bps_per_side": cost_bps_per_side,
        "test_days": int(test_days),
    }


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Bitcoin Predictive Model – Data & Trend Explorer",
        layout="wide",
    )

    st.title("Bitcoin Predictive Model – Data & Trend Explorer")

    # ---- Sidebar ----
    st.sidebar.header("Options")

    use_live = st.sidebar.checkbox(
        "Use live data (CoinDesk + Fear & Greed)",
        value=True,
    )

    horizon = st.sidebar.selectbox(
        "Prediction horizon:",
        options=[1, 7, 30, 90],
        index=1,
        format_func=lambda h: f"{h} days",
        key="sidebar_horizon",
    )

    show_raw_regime = st.sidebar.checkbox("Show raw regime (debug)", value=False)

    show_advanced = st.sidebar.checkbox("Show advanced diagnostics", value=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Trading / backtest")

    trade_threshold = st.sidebar.slider(
        "Probability threshold to go LONG",
        min_value=0.50,
        max_value=0.80,
        value=0.55,
        step=0.01,
        help="If model prob. BTC ↑ is above this value, strategy goes long.",
    )

    trade_cost_bps = st.sidebar.slider(
        "Trading cost per side (bps)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
        help="1 bp = 0.01%. Cost per side (entry AND exit). Round-trip ≈ 2 × this value.",
    )

    backtest_window_days = st.sidebar.slider(
        "Backtest window (days)",
        min_value=365,
        max_value=2920,  # ~8 anys
        value=365,
        step=30,
        help="Number of days used for the out-of-sample backtest.",
    )

    # ---- Load dataset ----
    with st.spinner("Building dataset..."):
        df, source = load_dataset(use_live=use_live)

    if df is None or df.empty:
        st.error("Dataset is empty.")
        st.stop()

    # ---- Train models ----
    with st.spinner("Training directional models (1d / 7d / 30d / 90d)..."):
        models, metrics_all, metas = train_all_models(df)

    with st.spinner("Training trend-change models (bull/bear)..."):
        trend_models, trend_metrics, trend_metas = train_all_trend_models(df)

    # ---- Source info (always) ----
    st.caption(
        f"**Source:** "
        f"{'LIVE (CoinDesk Data API + factors)' if source == 'live' else 'LOCAL CSV + factors'}"
        f" · Rows: {len(df):,} · "
        f"Date range: {df['date'].min().date()} → {df['date'].max().date()}"
    )

    # ---- Debug info only if advanced ----
    if show_advanced:
        st.write(
            "DEBUG date range in df:",
            pd.to_datetime(df["date"]).min(),
            "→",
            pd.to_datetime(df["date"]).max(),
        )

        with st.expander("Debug: first rows of dataset"):
            st.dataframe(df.head(10))

        with st.expander("Debug: last rows of dataset"):
            st.dataframe(df.tail(10))

        with st.expander("Debug: columns & dtypes"):
            st.write(df.dtypes)

    # ---- Sidebar footer ----
    st.sidebar.markdown("---")
    st.sidebar.write(f"Rows in dataset: **{len(df):,}**")

    # =================================================================
    # PRICE CHART
    # =================================================================
    st.subheader("BTC price (daily close)")

    df_price = df[["date", "close"]].dropna().copy()

    if df_price.empty:
        st.warning("No price data available to plot.")
    else:
        price_chart = (
            alt.Chart(df_price)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("close:Q", title="BTC Price (close)"),
                tooltip=["date:T", "close:Q"],
            )
            .properties(height=400)
        )

        st.altair_chart(price_chart, use_container_width=True)

    # =================================================================
    # TREND REGIME & TURNING POINTS
    # =================================================================
    if "regime_smooth" in df.columns:
        st.subheader("Trend regime & turning points")

        df_reg = df[["date", "close", "regime_smooth"]].dropna().copy()
        df_reg = df_reg.sort_values("date").reset_index(drop=True)

        # Identify regime changes
        df_reg["regime_change"] = df_reg["regime_smooth"].diff().fillna(0)

        # Bull turns: regime_smooth becomes 1 from <= 0
        bull_turns = df_reg[
            (df_reg["regime_smooth"] == 1) & (df_reg["regime_change"] > 0)
        ]

        # Bear turns: regime_smooth becomes -1 from >= 0
        bear_turns = df_reg[
            (df_reg["regime_smooth"] == -1) & (df_reg["regime_change"] < 0)
        ]

        df_reg["regime_label"] = df_reg["regime_smooth"].apply(regime_label_from_value)

        base_regime_chart = (
            alt.Chart(df_reg)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("close:Q", title="BTC Price (close)"),
                color=alt.Color(
                    "regime_label:N",
                    scale=alt.Scale(
                        domain=["Bear", "Sideways", "Bull"],
                        range=["#d62728", "#7f7f7f", "#2ca02c"],
                    ),
                    legend=alt.Legend(title="Regime"),
                ),
                tooltip=["date:T", "close:Q", "regime_label:N"],
            )
            .properties(height=400)
        )

        bull_points = (
            alt.Chart(bull_turns)
            .mark_point(shape="triangle-up", size=80, filled=True, color="#2ca02c")
            .encode(x="date:T", y="close:Q", tooltip=["date:T", "close:Q"])
        )

        bear_points = (
            alt.Chart(bear_turns)
            .mark_point(shape="triangle-down", size=80, filled=True, color="#d62728")
            .encode(x="date:T", y="close:Q", tooltip=["date:T", "close:Q"])
        )

        st.altair_chart(
            base_regime_chart + bull_points + bear_points,
            use_container_width=True,
        )
        st.caption(
            "Green ▲ = start of **bull** regime · Red ▼ = start of **bear** regime"
        )

        # Optional: raw regime debug plot
        if show_raw_regime and "regime_raw" in df.columns:
            df_reg_raw = df[["date", "regime_raw", "regime_smooth"]].dropna().copy()
            df_reg_raw = df_reg_raw.sort_values("date")
            regime_chart = (
                alt.Chart(df_reg_raw.melt("date", var_name="series", value_name="value"))
                .mark_line()
                .encode(
                    x="date:T",
                    y="value:Q",
                    color="series:N",
                    tooltip=["date:T", "series:N", "value:Q"],
                )
                .properties(height=200)
            )
            st.altair_chart(regime_chart, use_container_width=True)
            st.caption("Raw vs smoothed regime (debug view).")

        # Current regime summary
        latest_row = df_reg.iloc[-1]
        latest_regime = latest_row["regime_smooth"]
        latest_label = regime_label_from_value(latest_regime)
        latest_date = latest_row["date"]

        last_change_idx = (
            df_reg.index[df_reg["regime_change"] != 0].max()
            if (df_reg["regime_change"] != 0).any()
            else None
        )

        if last_change_idx is None:
            start_date = df_reg["date"].min()
        else:
            start_idx = min(last_change_idx + 1, len(df_reg) - 1)
            start_date = df_reg.loc[start_idx, "date"]

        days_in_regime = (latest_date - start_date).days

        color_map = {"Bull": "#2ca02c", "Bear": "#d62728", "Sideways": "#7f7f7f"}
        pill_color = color_map.get(latest_label, "#7f7f7f")

        st.markdown(
            f"""
            <div style="font-size:16px; margin-top:0.5rem;">
              <strong>Current regime:</strong>
              <span style="
                  display:inline-block;
                  padding:2px 8px;
                  border-radius:999px;
                  background-color:{pill_color}20;
                  color:{pill_color};
                  font-weight:600;
              ">
                {latest_label}
              </span>
              <span style="margin-left:4px;">
                (since {start_date.date()}, ~{days_in_regime} days)
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------- Regime quality summary ----------------
        st.markdown("### How good are these regimes?")

        regime_horizon = st.selectbox(
            "Validation horizon for regime quality:",
            [7, 30, 90],
            index=1,
            format_func=lambda h: f"{h} days ahead",
            key="regime_quality_horizon",
        )

        regime_stats = compute_regime_quality(df, horizon=regime_horizon)
        if regime_stats.empty:
            st.info(
                f"Not enough data to validate regimes for a {regime_horizon}-day horizon."
            )
        else:
            st.dataframe(
                regime_stats.style.format(
                    {
                        "up_prob_%": "{:.1f}",
                        "mean_ret_%": "{:.1f}",
                        "median_ret_%": "{:.1f}",
                    }
                )
            )
            st.caption(
                f"When the model labels a day as Bull / Bear / Sideways, this table "
                f"shows how often BTC ended up higher {regime_horizon} days later, "
                f"and the average/median {regime_horizon}-day return from those dates."
            )

    else:
        st.info(
            "Trend regime features (`regime_smooth`) are not available in the dataset. "
            "Make sure `trend_regime.add_trend_regime_block` is applied in build_dataset."
        )

    # =================================================================
    # MCIS-LIKE SCORE & REGIMES
    # =================================================================
    st.subheader("BTC price with MCIS-like regimes")

    df_mcis = None
    factor_config_mcis, mcis_labels = build_mcis_factor_config(df)

    if "y_ret_30d" not in df.columns:
        st.info(
            "Column `y_ret_30d` not found in dataset – cannot compute MCIS-like score "
            "based on 30-day returns."
        )
    elif len(factor_config_mcis) < 2:
        st.info(
            "Not enough macro/activity factors detected to compute an MCIS-like score. "
            "Check that ETF flows, rates, equity risk, sentiment and activity columns "
            "are present and correctly named."
        )
        if show_advanced:
            st.write("Detected factors:", factor_config_mcis)
    else:
        try:
            mcis_z, mcis_weights, mcis_lags = compute_composite_score(
                df=df,
                target_col="y_ret_30d",
                factor_config=factor_config_mcis,
                lam=1.0,
                min_weight=0.05,
                max_lag=2,
            )
            df_mcis = df[["date", "close"]].copy()
            df_mcis = df_mcis.merge(
                mcis_z.rename("MCIS_Z"),
                left_on="date",
                right_index=True,
                how="left",
            )
        except Exception as e:
            df_mcis = None
            st.info("MCIS-like score could not be computed.")
            if show_advanced:
                st.warning(f"MCIS error: {e}")

    if df_mcis is not None and not df_mcis["MCIS_Z"].dropna().empty:
        df_mcis = df_mcis.sort_values("date").reset_index(drop=True)

        def mcis_regime(z):
            if z >= 1.0:
                return "Tailwind"
            elif z <= -1.0:
                return "Headwind"
            else:
                return "Neutral"

        df_mcis["MCIS_regime"] = df_mcis["MCIS_Z"].apply(mcis_regime)

        # Blocs consecutius amb el mateix règim per pintar rectangles
        df_mcis["regime_change"] = (
            df_mcis["MCIS_regime"] != df_mcis["MCIS_regime"].shift()
        ).astype(int)
        df_mcis["block_id"] = df_mcis["regime_change"].cumsum()

        blocks = (
            df_mcis.groupby(["block_id", "MCIS_regime"])
            .agg(start=("date", "min"), end=("date", "max"))
            .reset_index()
        )

        shading = (
            alt.Chart(blocks)
            .mark_rect(opacity=0.15)
            .encode(
                x=alt.X("start:T", title="Date"),
                x2="end:T",
                color=alt.Color(
                    "MCIS_regime:N",
                    scale=alt.Scale(
                        domain=["Headwind", "Neutral", "Tailwind"],
                        range=["#fcaeae", "#f5f5f5", "#c7f5c4"],
                    ),
                    legend=alt.Legend(title="MCIS regime"),
                ),
            )
        )

        price_line_mcis = (
            alt.Chart(df_mcis)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("close:Q", title="BTC price (USD)"),
                tooltip=["date:T", "close:Q", "MCIS_Z:Q", "MCIS_regime:N"],
            )
        )

        st.altair_chart(
            shading + price_line_mcis,
            use_container_width=True,
        )

        st.caption(
            "Shaded regions mark MCIS-like regimes: green (Tailwind, MCIS_Z ≥ 1), "
            "red (Headwind, MCIS_Z ≤ -1). Neutral zone in between."
        )

        # Qualitat de MCIS: com afecta als retorns futurs
        mcis_horizon = st.selectbox(
            "Validation horizon for MCIS regimes:",
            [30, 90],
            index=0,
            format_func=lambda h: f"{h} days ahead",
            key="mcis_quality_horizon",
        )

        ret_col_h = f"y_ret_{mcis_horizon}d"
        if ret_col_h in df.columns:
            df_q = df[["date", ret_col_h]].merge(
                df_mcis[["date", "MCIS_regime"]],
                on="date",
                how="inner",
            ).dropna()

            if not df_q.empty:
                df_q["up_flag"] = (df_q[ret_col_h] > 0).astype(float)
                g = df_q.groupby("MCIS_regime")
                q_stats = g.agg(
                    n_obs=("date", "count"),
                    up_prob=("up_flag", "mean"),
                    mean_log_ret=(ret_col_h, "mean"),
                ).reset_index()
                q_stats["up_prob_%"] = q_stats["up_prob"] * 100.0
                q_stats["mean_ret_%"] = (
                    np.exp(q_stats["mean_log_ret"]) - 1.0
                ) * 100.0

                st.dataframe(
                    q_stats[["MCIS_regime", "n_obs", "up_prob_%", "mean_ret_%"]]
                    .sort_values("MCIS_regime")
                    .style.format(
                        {"up_prob_%": "{:.1f}", "mean_ret_%": "{:.1f}"}
                    )
                )
                st.caption(
                    f"When MCIS regime is Tailwind / Headwind / Neutral, this table "
                    f"shows how often BTC ended up higher {mcis_horizon} days later "
                    f"and the average {mcis_horizon}-day return."
                )

        if show_advanced:
            st.markdown("**Advanced – MCIS factor weights & lags**")
            # pretty labels
            labels_used = []
            weights_list = []
            lags_list = []
            for col in mcis_weights.index:
                labels_used.append(mcis_labels.get(col, col))
                weights_list.append(mcis_weights[col])
                lags_list.append(mcis_lags.get(col, 0))

            df_mcis_diag = pd.DataFrame(
                {
                    "factor": labels_used,
                    "column": list(mcis_weights.index),
                    "weight": weights_list,
                    "lag_days": lags_list,
                }
            )
            st.dataframe(df_mcis_diag.style.format({"weight": "{:.3f}"}))

    # =================================================================
    # TREND-CHANGE SIGNALS (MODELS)
    # =================================================================
    st.subheader("Trend-change signals (bull / bear)")

    trend_h = max(horizon, 7)  # at least 7 days for trend-change models

    if "regime_smooth" not in df.columns:
        st.info(
            "Trend-change models require 'regime_smooth' in the dataset. "
            "Make sure trend_regime.add_trend_regime_block() is applied."
        )
    else:
        try:
            X_bull, y_bull, dates_bull, feat_bull, target_bull = (
                build_trend_change_feature_matrix(
                    df,
                    horizon=trend_h,
                    direction="bull",
                )
            )
            X_bear, y_bear, dates_bear, feat_bear, target_bear = (
                build_trend_change_feature_matrix(
                    df,
                    horizon=trend_h,
                    direction="bear",
                )
            )
        except Exception as e:
            st.warning(f"Could not build trend-change feature matrix: {e}")
            X_bull = X_bear = None

        if (
            X_bull is None
            or X_bull.empty
            or "bull" not in trend_models
            or "bear" not in trend_models
            or trend_h not in trend_models["bull"]
            or trend_h not in trend_models["bear"]
        ):
            st.info(
                "Not enough data to compute trend-change signals "
                f"for horizon {trend_h} days."
            )
        else:
            bull_model = trend_models["bull"][trend_h]
            bear_model = trend_models["bear"][trend_h]

            idx_bull = len(X_bull) - 1
            idx_bear = len(X_bear) - 1

            date_ref = dates_bull.iloc[idx_bull]

            p_bull = bull_model.predict_proba(X_bull.iloc[[idx_bull]])[0, 1]
            p_bear = bear_model.predict_proba(X_bear.iloc[[idx_bear]])[0, 1]

            m_bull = trend_metrics["bull"][trend_h]
            m_bear = trend_metrics["bear"][trend_h]

            thr_bull = m_bull.get("best_bal_acc_thr", 0.5)
            thr_bear = m_bear.get("best_bal_acc_thr", 0.5)

            strong_bull = is_strong_trend_model("bull", trend_h, trend_metrics)
            strong_bear = is_strong_trend_model("bear", trend_h, trend_metrics)

            if not (strong_bull and strong_bear) and not show_advanced:
                st.info(
                    "Trend-change models for this horizon do not reach the "
                    "configured quality thresholds – signals hidden."
                )
            else:
                if not (strong_bull and strong_bear):
                    st.warning(
                        "⚠️ Trend-change models for this horizon are relatively weak – "
                        "use these signals with caution."
                    )

                c1, c2 = st.columns(2)

                c1.metric(
                    label=f"Prob. start of BULL regime in next {trend_h} days",
                    value=f"{p_bull * 100:.1f} %",
                )
                c1.write(
                    f"Test acc (0.5 thr): {m_bull['test_accuracy']:.3f} · "
                    f"AUC: {m_bull['test_auc']:.3f} · "
                    f"Best bal. thr: {thr_bull:.2f} "
                    f"(bal. acc: {m_bull.get('best_bal_acc', float('nan')):.3f})"
                )

                c2.metric(
                    label=f"Prob. start of BEAR regime in next {trend_h} days",
                    value=f"{p_bear * 100:.1f} %",
                )
                c2.write(
                    f"Test acc (0.5 thr): {m_bear['test_accuracy']:.3f} · "
                    f"AUC: {m_bear['test_auc']:.3f} · "
                    f"Best bal. thr: {thr_bear:.2f} "
                    f"(bal. acc: {m_bear.get('best_bal_acc', float('nan')):.3f})"
                )

                st.caption(
                    f"Reference date for probabilities: {date_ref.date()}. "
                    f"Targets are: bull_turn_{trend_h}d / bear_turn_{trend_h}d."
                )

    # =================================================================
    # HORIZON STATS + FORECAST RANGE + MODEL PROBABILITY
    # =================================================================
    st.subheader(f"Targets & forecast for {horizon}-day horizon")

    ret_col = f"y_ret_{horizon}d"
    up_col = f"up_{horizon}d"

    mean_ret = std_ret = up_ratio = None
    price_low = price_high = expected_price = None
    valid = None

    if ret_col in df.columns:
        valid = df.dropna(subset=[ret_col])
        if not valid.empty:
            mean_ret = valid[ret_col].mean()
            std_ret = valid[ret_col].std()
            if up_col in valid.columns:
                up_ratio = valid[up_col].mean()

            last_row = df.dropna(subset=["close"]).iloc[-1]
            last_date = last_row["date"]
            last_price = last_row["close"]

            forecast_end_date = last_date + pd.Timedelta(days=horizon)
            expected_price = last_price * np.exp(mean_ret)

            low_log = mean_ret - 1.96 * std_ret
            high_log = mean_ret + 1.96 * std_ret
            price_low = last_price * np.exp(low_log)
            price_high = last_price * np.exp(high_log)

    if mean_ret is None or valid is None or valid.empty:
        st.warning(f"Column `{ret_col}` not found or not enough data.")
    else:
        cols = st.columns(4)

        cols[0].metric("Mean log-return", f"{mean_ret:.4f}")
        cols[1].metric("Std. of log-return", f"{std_ret:.4f}")
        if up_ratio is not None:
            cols[2].metric(
                "Historical up probability",
                f"{100 * up_ratio:.1f} %",
            )
        if expected_price is not None:
            cols[3].metric(
                f"Expected price in {horizon}d",
                f"${expected_price:,.0f}",
            )

        if price_low is not None and price_high is not None:
            st.markdown(
                f"**Estimated 95% price range in {horizon} days:** "
                f"${price_low:,.0f}  →  ${price_high:,.0f}"
            )

        # ---- Model-based probability (directional classifier) ----
        if horizon in models:
            X_all, y_all, dates_all, feature_cols, target_col = build_feature_matrix(
                df, horizon=horizon
            )
            if not X_all.empty:
                m = metrics_all[horizon]
                strong_clf = has_decent_clf_metrics(horizon, metrics_all)

                if not strong_clf and not show_advanced:
                    st.info(
                        "Directional model for this horizon does not reach the "
                        "configured quality thresholds – signal hidden."
                    )
                else:
                    latest_idx = len(X_all) - 1
                    latest_date = dates_all.iloc[latest_idx]
                    proba_up = models[horizon].predict_proba(
                        X_all.iloc[[latest_idx]]
                    )[0, 1]

                    thr_bal = m.get("best_bal_acc_thr", 0.5)
                    thr_f1 = m.get("best_f1_thr", 0.5)

                    # Qualitative signal using balanced-accuracy threshold
                    if proba_up >= thr_bal:
                        signal = "Bullish (strong)"
                        color = "🟢"
                    elif proba_up <= 1.0 - thr_bal:
                        signal = "Bearish (strong)"
                        color = "🔴"
                    else:
                        signal = "Neutral / low edge"
                        color = "⚪️"

                    st.markdown("### Model-based directional signal")
                    if not strong_clf:
                        st.warning(
                            "⚠️ Classifier metrics for this horizon are below "
                            "the configured thresholds – debug view only."
                        )

                    st.metric(
                        label=f"Prob. BTC higher in {horizon} days "
                        f"(from {latest_date.date()})",
                        value=f"{proba_up * 100:.1f} %",
                        delta=f"{color} {signal}",
                    )

                    st.write(
                        f"**Test acc (0.5 thr):** {m['test_accuracy']:.3f}  |  "
                        f"**Test AUC:** {m['test_auc']:.3f}  |  "
                        f"**Test Brier:** {m['test_brier']:.3f}"
                    )
                    st.write(
                        f"**Optimal threshold (balanced acc):** "
                        f"{thr_bal:.2f} "
                        f"(bal. acc: {m.get('best_bal_acc', float('nan')):.3f})  ·  "
                        f"**Optimal threshold (F1):** {thr_f1:.2f} "
                        f"(F1: {m.get('best_f1', float('nan')):.3f})"
                    )

                    # Edge vs historical probability
                    if up_ratio is not None:
                        edge_pp = (proba_up - up_ratio) * 100.0
                        st.write(
                            f"**Model edge vs historical up probability:** "
                            f"{edge_pp:+.1f} percentage points."
                        )

        st.caption("Distribution of future log-returns")
        hist = (
            alt.Chart(valid)
            .mark_bar()
            .encode(
                x=alt.X(
                    ret_col,
                    bin=alt.Bin(maxbins=50),
                    title=f"log-return over {horizon} days",
                ),
                y=alt.Y("count()", title="Count"),
            )
            .properties(height=300)
        )
        st.altair_chart(hist, use_container_width=True)

    # -----------------------------------------------------------------
    # MULTI-HORIZON SNAPSHOT (SIGNALS)
    # -----------------------------------------------------------------
    st.subheader("Multi-horizon model snapshot")

    snapshot_rows = []
    for h in [1, 7, 30, 90]:
        if h not in models:
            continue

        X_h, y_h, dates_h, feat_h, target_h = build_feature_matrix(df, horizon=h)
        if X_h is None or X_h.empty:
            continue

        last_idx = len(X_h) - 1
        date_ref_h = pd.to_datetime(dates_h.iloc[last_idx])
        proba_up_h = float(models[h].predict_proba(X_h.iloc[[last_idx]])[0, 1])

        m_h = metrics_all[h]
        thr_bal_h = m_h.get("best_bal_acc_thr", 0.5)
        strong_h = has_decent_clf_metrics(h, metrics_all)

        # Etiqueta qualitativa
        if proba_up_h >= thr_bal_h:
            label = "Bullish"
        elif proba_up_h <= 1.0 - thr_bal_h:
            label = "Bearish"
        else:
            label = "Neutral"

        snapshot_rows.append(
            {
                "horizon_days": h,
                "as_of": date_ref_h.date(),
                "prob_up_%": proba_up_h * 100.0,
                "signal": label,
                "thr_bal": thr_bal_h,
                "test_accuracy": m_h.get("test_accuracy", np.nan),
                "test_auc": m_h.get("test_auc", np.nan),
                "strong_model": strong_h,
            }
        )

    if snapshot_rows:
        df_snapshot = pd.DataFrame(snapshot_rows).sort_values("horizon_days")
        df_snapshot["prob_up_%"] = df_snapshot["prob_up_%"].round(1)
        df_snapshot["thr_bal"] = df_snapshot["thr_bal"].round(2)
        df_snapshot["test_accuracy"] = df_snapshot["test_accuracy"].round(3)
        df_snapshot["test_auc"] = df_snapshot["test_auc"].round(3)

        df_public = df_snapshot[df_snapshot["strong_model"]]
        if not df_public.empty:
            st.dataframe(df_public.drop(columns=["strong_model"]))
        else:
            st.info(
                "No horizon currently meets the classifier quality thresholds "
                "(accuracy/AUC) for the multi-horizon snapshot."
            )

        if show_advanced and not df_snapshot.equals(df_public):
            st.markdown("**Advanced – all horizons (including weak models)**")
            st.dataframe(df_snapshot)
    else:
        st.info("Not enough data to compute the multi-horizon snapshot.")

    # -----------------------------------------------------------------
    # BACKTEST – STRATEGY USING MODEL SIGNAL
    # -----------------------------------------------------------------
    st.subheader("Backtest – strategy using model signal (long / flat)")

    bt_cache: Dict[int, Dict] = {}

    # 1) Resum de backtests per horitzó (1d / 7d / 30d / 90d)
    backtest_summary_public = []
    backtest_summary_all = []

    for h in [1, 7, 30, 90]:
        if h not in models:
            continue

        bt_h = run_long_flat_backtest(
            df=df,
            horizon=h,
            model=models[h],
            threshold=trade_threshold,
            cost_bps_per_side=trade_cost_bps,
            test_days=backtest_window_days,
        )
        if not bt_h:
            continue

        bt_cache[h] = bt_h

        row = {
            "horizon_days": h,
            "total_return_strategy_%": bt_h["total_return_strategy"] * 100.0,
            "total_return_buyhold_%": bt_h["total_return_buyhold"] * 100.0,
            "cagr_strategy_%": bt_h["cagr_strategy"] * 100.0,
            "hit_rate_%": (
                bt_h["hit_rate"] * 100.0
                if not np.isnan(bt_h["hit_rate"])
                else np.nan
            ),
            "ann_vol_%": (
                bt_h["ann_vol"] * 100.0
                if not np.isnan(bt_h["ann_vol"])
                else np.nan
            ),
            "sharpe": bt_h["sharpe"],
            "max_dd_strategy_%": bt_h["max_dd_strategy"] * 100.0,
            "n_trades": bt_h["n_trades"],
        }

        backtest_summary_all.append(row)
        if is_strong_directional_model(h, metrics_all, bt_cache):
            backtest_summary_public.append(row)

    if backtest_summary_public:
        df_bt_summary = pd.DataFrame(backtest_summary_public).sort_values(
            "horizon_days"
        )
        st.dataframe(
            df_bt_summary.style.format(
                {
                    "total_return_strategy_%": "{:.1f}",
                    "total_return_buyhold_%": "{:.1f}",
                    "cagr_strategy_%": "{:.1f}",
                    "hit_rate_%": "{:.1f}",
                    "ann_vol_%": "{:.1f}",
                    "max_dd_strategy_%": "{:.1f}",
                }
            )
        )
    else:
        st.info(
            "No horizon currently meets the combined quality thresholds "
            "(model + backtest) for the strategy summary."
        )

    if show_advanced and backtest_summary_all:
        st.markdown("**Advanced – backtest summary for all horizons**")
        df_bt_all = pd.DataFrame(backtest_summary_all).sort_values("horizon_days")
        st.dataframe(
            df_bt_all.style.format(
                {
                    "total_return_strategy_%": "{:.1f}",
                    "total_return_buyhold_%": "{:.1f}",
                    "cagr_strategy_%": "{:.1f}",
                    "hit_rate_%": "{:.1f}",
                    "ann_vol_%": "{:.1f}",
                    "max_dd_strategy_%": "{:.1f}",
                }
            )
        )

    st.markdown("---")

    # 2) Gràfic detallat per a l'horitzó seleccionat a la sidebar
    if horizon not in models:
        st.info("No trained model available for this horizon.")
    else:
        bt = bt_cache.get(horizon)
        if bt is None:
            bt = run_long_flat_backtest(
                df=df,
                horizon=horizon,
                model=models[horizon],
                threshold=trade_threshold,
                cost_bps_per_side=trade_cost_bps,
                test_days=backtest_window_days,
            )
            if bt:
                bt_cache[horizon] = bt

        if not bt:
            st.info("Not enough data to run backtest for this horizon.")
        else:
            strong = is_strong_directional_model(horizon, metrics_all, bt_cache)

            if not strong and not show_advanced:
                st.info(
                    "Backtest metrics for this horizon do not meet the quality "
                    "thresholds (hit rate / Sharpe). "
                    "Select another horizon or enable 'advanced diagnostics' "
                    "to inspect it anyway."
                )
            else:
                if not strong:
                    st.warning(
                        "⚠️ Strategy for this horizon is relatively weak – "
                        "use these backtest results with caution."
                    )

                # KPIs principals
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Total return (strategy)",
                    f"{bt['total_return_strategy'] * 100:,.1f} %",
                )
                c2.metric(
                    "Total return (buy & hold)",
                    f"{bt['total_return_buyhold'] * 100:,.1f} %",
                )
                c3.metric(
                    "CAGR (strategy)",
                    f"{bt['cagr_strategy'] * 100:,.1f} %",
                )
                if not np.isnan(bt["hit_rate"]):
                    c4.metric(
                        "Hit rate (on trades)",
                        f"{bt['hit_rate'] * 100:,.1f} %",
                    )
                else:
                    c4.metric("Hit rate (on trades)", "N/A")

                # Risc
                c5, c6, c7 = st.columns(3)
                if not np.isnan(bt["ann_vol"]):
                    c5.metric(
                        "Ann. volatility (strategy)",
                        f"{bt['ann_vol'] * 100:,.1f} %",
                    )
                else:
                    c5.metric("Ann. volatility (strategy)", "N/A")

                if not np.isnan(bt["sharpe"]):
                    c6.metric("Sharpe (approx)", f"{bt['sharpe']:.2f}")
                else:
                    c6.metric("Sharpe (approx)", "N/A")

                c7.metric(
                    "Max drawdown (strategy)",
                    f"{bt['max_dd_strategy'] * 100:,.1f} %",
                )

                # Equity curves
                equity_df = pd.DataFrame(
                    {
                        "date": bt["dates"],
                        "buyhold_equity": bt["buyhold_equity"].values,
                        "strategy_equity": bt["strategy_equity"].values,
                    }
                )

                equity_long = equity_df.melt(
                    "date", value_name="equity", var_name="series"
                )

                eq_chart = (
                    alt.Chart(equity_long)
                    .mark_line()
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("equity:Q", title="Equity (normalized)"),
                        color=alt.Color(
                            "series:N",
                            title="Series",
                            scale=alt.Scale(
                                domain=["buyhold_equity", "strategy_equity"],
                                range=["#1f77b4", "#aec7e8"],
                            ),
                        ),
                        tooltip=["date:T", "series:N", "equity:Q"],
                    )
                    .properties(height=400)
                )

                st.altair_chart(eq_chart, use_container_width=True)

                st.caption(
                    f"Backtest run on the last ~{bt['test_days']} days "
                    f"with a {horizon}-day horizon. "
                    f"Threshold = {bt['threshold']:.2f}, trading cost per side = "
                    f"{bt['cost_bps_per_side']:.1f} bps."
                )

    # -----------------------------------------------------------------
    # BACKTEST KPI OVERVIEW BY HORIZON
    # -----------------------------------------------------------------
    st.subheader("Backtest KPIs by horizon (using model signal)")

    kpi_rows_public = []
    kpi_rows_all = []

    for h in [7, 30, 90]:
        if h not in models:
            continue

        res = bt_cache.get(h)
        if res is None:
            res = run_long_flat_backtest(
                df=df,
                horizon=h,
                model=models[h],
                threshold=trade_threshold,
                cost_bps_per_side=trade_cost_bps,
                test_days=int(backtest_window_days),
            )
            if res:
                bt_cache[h] = res

        if not res:
            continue

        row = {
            "horizon_days": h,
            "total_return_strategy": res["total_return_strategy"],
            "total_return_buyhold": res["total_return_buyhold"],
            "cagr_strategy": res["cagr_strategy"],
            "hit_rate": res["hit_rate"],
            "ann_vol": res["ann_vol"],
            "sharpe": res["sharpe"],
            "max_dd_strategy": res["max_dd_strategy"],
            "n_trades": res["n_trades"],
        }

        kpi_rows_all.append(row)
        if is_strong_directional_model(h, metrics_all, bt_cache):
            kpi_rows_public.append(row)

    if kpi_rows_public:
        kpi_df = pd.DataFrame(kpi_rows_public).sort_values("horizon_days")
        st.dataframe(kpi_df)
        if backtest_window_days <= 370:
            st.caption(
                "KPIs computed on the last ~365 days (pure out-of-sample for the directional models)."
            )
        else:
            st.caption(
                "KPIs computed on a longer window (~8 years). "
                "These results are partly in-sample because the models were trained on this history."
            )
    else:
        st.info(
            "No horizon currently meets the combined quality thresholds "
            "for the KPI overview."
        )

    if show_advanced and kpi_rows_all:
        st.markdown("**Advanced – KPIs for all horizons**")
        kpi_df_all = pd.DataFrame(kpi_rows_all).sort_values("horizon_days")
        st.dataframe(kpi_df_all)

    # =================================================================
    # FACTOR EXPLORER (ADVANCED)
    # =================================================================
    if show_advanced:
        st.subheader("Factor Explorer")

        ignore_exact = {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "log_ret_1d",
            "log_ret_3d",
            "log_ret_7d",
            "vol_7d",
            "vol_30d",
            "ma_20",
            "ma_50",
            "ma_90",
            "price_over_ma20",
            "price_over_ma50",
            "price_over_ma90",
            "drawdown_90d",
            "regime_raw",
            "regime_smooth",
            "days_since_last_halving",
            "days_to_next_halving",
        }
        ignore_prefixes = ("y_ret_", "up_", "bull_turn_", "bear_turn_")

        numeric_cols = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]

        factor_cols = []
        for c in numeric_cols:
            if c in ignore_exact:
                continue
            if any(c.startswith(pfx) for pfx in ignore_prefixes):
                continue
            factor_cols.append(c)

        with st.expander("Debug: candidate factor columns", expanded=False):
            st.write(factor_cols)

        if not factor_cols:
            st.info(
                "No external factor columns detected yet.\n\n"
                "This is expected if live factor APIs (on-chain activity, ETF flows, "
                "equity prices) are still stubs or failed to fetch."
            )
        else:
            factor_name = st.selectbox(
                "Choose a factor to inspect:",
                sorted(factor_cols),
                key="factor_name_selector",
            )

            factor_horizon = st.selectbox(
                "Prediction horizon (factor explorer):",
                [1, 7, 30, 90],
                format_func=lambda h: f"{h} days",
                key="factor_horizon_selector",
            )
            factor_ret_col = f"y_ret_{factor_horizon}d"

            if factor_ret_col not in df.columns:
                st.warning(f"Return column '{factor_ret_col}' not found in dataset.")
            else:
                df_factor = df[["date", factor_name, factor_ret_col]].dropna().copy()
                if df_factor.empty:
                    st.warning(
                        "No overlapping data for this factor and return horizon."
                    )
                else:
                    factor_chart = (
                        alt.Chart(df_factor)
                        .mark_line()
                        .encode(
                            x=alt.X("date:T", title="Date"),
                            y=alt.Y(f"{factor_name}:Q", title=factor_name),
                            tooltip=["date:T", f"{factor_name}:Q"],
                        )
                        .properties(height=250)
                    )

                    scatter = (
                        alt.Chart(df_factor)
                        .mark_circle(opacity=0.5)
                        .encode(
                            x=alt.X(f"{factor_name}:Q", title=factor_name),
                            y=alt.Y(
                                f"{factor_ret_col}:Q",
                                title=f"log-return {factor_horizon}d ahead",
                            ),
                            tooltip=[
                                "date:T",
                                f"{factor_name}:Q",
                                f"{factor_ret_col}:Q",
                            ],
                        )
                        .properties(height=250)
                    )

                    corr = df_factor[factor_name].corr(df_factor[factor_ret_col])

                    st.write(
                        f"**Factor:** `{factor_name}` | "
                        f"**Horizon:** {factor_horizon} days"
                    )
                    st.metric("Pearson correlation", f"{corr:.3f}")

                    st.altair_chart(factor_chart, use_container_width=True)
                    st.altair_chart(scatter, use_container_width=True)

        # =================================================================
        # PERFORMANCE SUMMARY TABLES (ADVANCED)
        # =================================================================
        st.subheader("Model performance summary – directional (up / down)")
        perf_rows = []
        for h, m in sorted(metrics_all.items()):
            perf_rows.append(
                {
                    "horizon_days": h,
                    "test_accuracy": m.get("test_accuracy", np.nan),
                    "test_auc": m.get("test_auc", np.nan),
                    "test_brier": m.get("test_brier", np.nan),
                    "best_bal_thr": m.get("best_bal_acc_thr", np.nan),
                    "best_bal_acc": m.get("best_bal_acc", np.nan),
                    "best_f1_thr": m.get("best_f1_thr", np.nan),
                    "best_f1": m.get("best_f1", np.nan),
                }
            )
        if perf_rows:
            st.dataframe(pd.DataFrame(perf_rows))

        st.subheader("Model performance summary – trend-change (bull / bear)")
        trend_rows = []
        for direction, metrics_dir in trend_metrics.items():
            for h, m in sorted(metrics_dir.items()):
                trend_rows.append(
                    {
                        "direction": direction,
                        "horizon_days": h,
                        "test_accuracy": m.get("test_accuracy", np.nan),
                        "test_auc": m.get("test_auc", np.nan),
                        "best_bal_thr": m.get("best_bal_acc_thr", np.nan),
                        "best_bal_acc": m.get("best_bal_acc", np.nan),
                        "best_f1_thr": m.get("best_f1_thr", np.nan),
                        "best_f1": m.get("best_f1", np.nan),
                    }
                )
        if trend_rows:
            st.dataframe(pd.DataFrame(trend_rows))

        # =================================================================
        # RAW DATA PREVIEW (ADVANCED)
        # =================================================================
        st.subheader("Raw data preview")
        st.dataframe(df.tail(20))


if __name__ == "__main__":
    main()
