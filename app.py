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
    compute_directional_backtest,
)

PROCESSED_PATH = Path("data/processed/btc_dataset.parquet")
RAW_PATH = Path("data/raw/btc_price_daily.csv")


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
        horizons=[7, 30],
        test_size_days=2922,
    )


@st.cache_resource(show_spinner=False)
def train_all_trend_models(df: pd.DataFrame):
    """
    Train bull/bear trend-change models for horizons 7, 30, 90 days.
    """
    return fit_all_trend_change_models(
        df,
        horizons=[7, 30, 90],
        test_size_days=2922,
    )


# ---------------------------------------------------------------------
# BACKTEST LONG / FLAT (NOU I CORREGIT)
# ---------------------------------------------------------------------

def run_long_flat_backtest(
    df: pd.DataFrame,
    horizon: int,
    model,
    threshold: float = 0.55,
    cost_bps_per_side: float = 0.0,
    test_days: int = 2922,
):
    """
    Long/flat backtest utilitzant el model direccional, amb passos NO solapats.

    Cada pas representa un bloc de `horizon` dies:
      - y_ret_{h}d = log(P_{t+h} / P_t)
      - si P(up) >= threshold -> LONG durant aquest bloc; si no -> FLAT (cash)
      - buy&hold acumula TOTS els blocs igualment (sense filtrar per senyal)

    Retorna dict amb:
      - dates                 (dates del final de cada bloc)
      - strategy_equity       (equity normalitzada, bloc a bloc)
      - buyhold_equity
      - total_return_strategy
      - total_return_buyhold
      - cagr_strategy
      - hit_rate, n_trades
      - ann_vol, sharpe
      - max_dd_strategy, max_dd_buyhold
      - threshold, cost_bps_per_side
    """
    ret_col = f"y_ret_{horizon}d"
    if ret_col not in df.columns:
        return {}

    # 1) Construïm matriu de features per a aquest horitzó (per tenir les dates exactes)
    X_all, y_cls, dates_all, feature_cols, target_col = build_feature_matrix(
        df, horizon=horizon
    )
    if X_all is None or X_all.empty:
        return {}

    dates_all = pd.to_datetime(dates_all).reset_index(drop=True)
    X_all = X_all.reset_index(drop=True)

    # 2) Alineem els log-returns y_ret_{h}d amb aquestes dates
    df_ret = df.copy()
    df_ret["date"] = pd.to_datetime(df_ret["date"])
    s_ret = df_ret.set_index("date")[ret_col]

    # reindex per tenir un log-return per cada fila de X_all
    rets_aligned = s_ret.reindex(dates_all)

    # filtrem files sense retorn definit (per exemple, les últimes h-1)
    mask_valid = np.isfinite(rets_aligned.values)
    X_all = X_all.loc[mask_valid].reset_index(drop=True)
    dates_all = dates_all.loc[mask_valid].reset_index(drop=True)
    rets_aligned = rets_aligned.loc[mask_valid].reset_index(drop=True)

    if len(X_all) < horizon * 3:
        # no hi ha prou història amb y_ret_h vàlid
        return {}

    # 3) Definim finestra de test en temps de calendari
    last_date = dates_all.max()
    test_start_date = last_date - pd.Timedelta(days=test_days + horizon)
    test_mask = dates_all >= test_start_date

    X_test = X_all.loc[test_mask].reset_index(drop=True)
    dates_test = dates_all.loc[test_mask].reset_index(drop=True)
    logret_test = rets_aligned.loc[test_mask].reset_index(drop=True)

    if len(X_test) < horizon * 2:
        return {}

    # 4) Índex NO solapats: 1 bloc cada `horizon` dies
    indices = list(range(0, len(X_test), horizon))
    if len(indices) < 2:
        return {}

    indices = [i for i in indices if i < len(X_test)]

    X_blocks = X_test.iloc[indices]
    dates_blocks = dates_test.iloc[indices].reset_index(drop=True)
    rets_blocks = logret_test.iloc[indices].astype(float).reset_index(drop=True)

    # 5) Probabilitats del model i posició (1 = long, 0 = flat)
    proba_up = model.predict_proba(X_blocks)[:, 1]
    signal = (proba_up >= threshold).astype(int)

    # 6) Retorns log de l'estratègia i del buy&hold per bloc
    cost = cost_bps_per_side / 10000.0
    log_cost_roundtrip = np.log(1.0 - 2.0 * cost) if cost > 0 else 0.0

    strat_log_rets = []
    bh_log_rets = []
    trade_dates = []
    n_trades = 0

    for date_start, r, pos in zip(dates_blocks, rets_blocks.values, signal):
        date_end = date_start + pd.Timedelta(days=horizon)
        trade_dates.append(date_end)

        r = float(r)
        bh_log_rets.append(r)

        if pos == 1:
            strat_log_rets.append(r + log_cost_roundtrip)
            n_trades += 1
        else:
            # flat: retorn 0 per aquest bloc
            strat_log_rets.append(0.0)

    strat_log_rets = np.array(strat_log_rets, dtype=float)
    bh_log_rets = np.array(bh_log_rets, dtype=float)

    # 7) Equity (normalitzada a 1) – blocs no solapats ⇒ compounding coherent
    equity_strategy = np.exp(np.cumsum(strat_log_rets))
    equity_buyhold = np.exp(np.cumsum(bh_log_rets))

    total_return_strategy = float(equity_strategy[-1] - 1.0)
    total_return_buyhold = float(equity_buyhold[-1] - 1.0)

    # 8) Metrics anualitzats
    n_blocks = len(trade_dates)
    total_days = n_blocks * horizon
    years = total_days / 365.0 if total_days > 0 else np.nan
    if years > 0:
        cagr_strategy = (1.0 + total_return_strategy) ** (1.0 / years) - 1.0
    else:
        cagr_strategy = np.nan

    # Hit rate: sobre blocs on realment estem long
    if n_trades > 0:
        hits_mask = signal == 1
        hits = (rets_blocks.values[hits_mask] > 0).sum()
        hit_rate = hits / n_trades
    else:
        hit_rate = np.nan

    # Volatilitat i Sharpe (per bloc, incloent blocs flat)
    mean_block_ret = float(strat_log_rets.mean())
    std_block_ret = float(strat_log_rets.std(ddof=0))
    steps_per_year = 365.0 / float(horizon) if horizon > 0 else np.nan

    if std_block_ret > 0 and not np.isnan(steps_per_year):
        ann_vol = std_block_ret * np.sqrt(steps_per_year)
        sharpe = (mean_block_ret * steps_per_year) / ann_vol
    else:
        ann_vol = np.nan
        sharpe = np.nan

    # 9) Max drawdown helper
    def _max_drawdown(equity: np.ndarray) -> float:
        running_max = np.maximum.accumulate(equity)
        dd = equity / running_max - 1.0
        return float(dd.min())  # negatiu (ex: -0.35 = -35%)

    max_dd_strategy = _max_drawdown(equity_strategy)
    max_dd_buyhold = _max_drawdown(equity_buyhold)

    return {
        "dates": pd.to_datetime(trade_dates),
        "strategy_equity": pd.Series(equity_strategy),
        "buyhold_equity": pd.Series(equity_buyhold),
        "total_return_strategy": total_return_strategy,
        "total_return_buyhold": total_return_buyhold,
        "cagr_strategy": cagr_strategy,
        "hit_rate": hit_rate,
        "n_trades": n_trades,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd_strategy": max_dd_strategy,
        "max_dd_buyhold": max_dd_buyhold,
        "threshold": threshold,
        "cost_bps_per_side": cost_bps_per_side,
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
        options=[7, 30],
        index=1,
        format_func=lambda h: f"{h} days",
        key="sidebar_horizon",
    )

    show_raw_regime = st.sidebar.checkbox("Show raw regime (debug)", value=False)

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

    # ---- Debug date range ----
    st.write(
        "DEBUG date range in df:",
        pd.to_datetime(df["date"]).min(),
        "→",
        pd.to_datetime(df["date"]).max(),
    )

    # ---- Source info ----
    st.caption(
        f"**Source:** "
        f"{'LIVE (CoinDesk Data API + factors)' if source == 'live' else 'LOCAL CSV + factors'}"
        f" · Rows: {len(df):,} · "
        f"Date range: {df['date'].min().date()} → {df['date'].max().date()}"
    )

    # ---- Debug expanders ----
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

        def _regime_label(x: float) -> str:
            if x >= 0.5:
                return "Bull"
            elif x <= -0.5:
                return "Bear"
            else:
                return "Sideways"

        df_reg["regime_label"] = df_reg["regime_smooth"].apply(_regime_label)

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

        # Current regime summary
        latest_row = df_reg.iloc[-1]
        latest_regime = latest_row["regime_smooth"]
        latest_label = _regime_label(latest_regime)
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

    else:
        st.info(
            "Trend regime features (`regime_smooth`) are not available in the dataset. "
            "Make sure `trend_regime.add_trend_regime_block` is applied in build_dataset."
        )

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
                latest_idx = len(X_all) - 1
                latest_date = dates_all.iloc[latest_idx]
                proba_up = models[horizon].predict_proba(
                    X_all.iloc[[latest_idx]]
                )[0, 1]

                m = metrics_all[horizon]

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
    # BACKTEST – STRATEGY USING MODEL SIGNAL
    # -----------------------------------------------------------------
    st.subheader("Backtest – strategy using model signal (long / flat)")

    if horizon not in models:
        st.info("No trained model available for this horizon.")
    else:
        bt = run_long_flat_backtest(
            df=df,
            horizon=horizon,
            model=models[horizon],
            threshold=trade_threshold,
            cost_bps_per_side=trade_cost_bps,
            test_days=2922,  # últim any aproximadament
        )

        if not bt:
            st.info("Not enough data to run backtest for this horizon.")
        else:
            # Primera fila de KPIs
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

            # Segona fila: risc
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
                f"Backtest run on the last ~{365} days with a {horizon}-day horizon. "
                f"Threshold = {bt['threshold']:.2f}, trading cost per side = "
                f"{bt['cost_bps_per_side']:.1f} bps."
            )

    # =================================================================
    # FACTOR EXPLORER
    # =================================================================
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

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

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
            [7, 30],
            format_func=lambda h: f"{h} days",
            key="factor_horizon_selector",
        )
        factor_ret_col = f"y_ret_{factor_horizon}d"

        if factor_ret_col not in df.columns:
            st.warning(f"Return column '{factor_ret_col}' not found in dataset.")
        else:
            df_factor = df[["date", factor_name, factor_ret_col]].dropna().copy()
            if df_factor.empty:
                st.warning("No overlapping data for this factor and return horizon.")
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
                        tooltip=["date:T", f"{factor_name}:Q", f"{factor_ret_col}:Q"],
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
    # PERFORMANCE SUMMARY TABLES
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
    # RAW DATA PREVIEW
    # =================================================================
    st.subheader("Raw data preview")
    st.dataframe(df.tail(20))


if __name__ == "__main__":
    main()
