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

def backtest_long_flat(df, horizon, model, threshold=0.55, test_size_days=365):
    """
    Simple long/flat backtest for a given horizon.

    Estratègia:
      - Cada dia t:
          * El model dona p = P(preu t+h > preu t)
          * Si p > threshold → LONG durant h dies
          * Si p <= threshold → FLAT durant h dies
      - El retorn real de cada trade el prenem de y_ret_{h}d.
    El backtest es fa sobre els últims `test_size_days` de dates.
    """

    ret_col = f"y_ret_{horizon}d"

    # 1) Features + dates alineats amb el model
    X_all, y_label, dates_all, feature_cols, target_col = build_feature_matrix(
        df, horizon=horizon
    )
    dates_all = pd.to_datetime(dates_all)

    # 2) Sèrie de log-returns y_ret_{h}d alineada per data
    df_ret = df.copy()
    df_ret["date"] = pd.to_datetime(df_ret["date"])
    s_ret = df_ret.set_index("date")[ret_col]

    # reindexem perquè cada fila de X_all tingui el seu log-return
    logret_all = s_ret.reindex(dates_all).values

    # filtrem qualsevol fila sense retorn definit
    mask = np.isfinite(logret_all)
    X_all = X_all.loc[mask].reset_index(drop=True)
    dates_all = dates_all.loc[mask].reset_index(drop=True)
    logret_all = logret_all[mask]

    if len(X_all) < test_size_days + 30:
        raise ValueError("Not enough data to run backtest for this horizon.")

    # 3) Zona de test = últims `test_size_days`
    test_start_idx = max(0, len(X_all) - test_size_days)
    X_test = X_all.iloc[test_start_idx:].reset_index(drop=True)
    dates_test = dates_all.iloc[test_start_idx:].reset_index(drop=True)
    logret_test = logret_all[test_start_idx:]

    # 4) Probabilitats de pujada
    proba_test = model.predict_proba(X_test)[:, 1]

    # 5) Trades no solapats cada `horizon` dies
    n = len(X_test)
    step = max(1, horizon)
    indices = list(range(0, n, step))
    if indices and indices[-1] >= n:
        indices = indices[:-1]

    equity_bh = [1.0]
    equity_strat = [1.0]
    series_dates = []
    n_trades = 0
    n_wins = 0

    for idx in indices:
        if idx >= len(logret_test):
            break

        r = float(logret_test[idx])      # log-return real sobre h dies
        p = float(proba_test[idx])       # probabilitat model
        date_t = dates_test.iloc[idx]
        date_end = date_t + pd.Timedelta(days=horizon)

        # Buy & hold: sempre exposat
        equity_bh.append(equity_bh[-1] * np.exp(r))

        # Estratègia long/flat
        if p > threshold:
            equity_strat.append(equity_strat[-1] * np.exp(r))
            n_trades += 1
            if r > 0:
                n_wins += 1
        else:
            equity_strat.append(equity_strat[-1])

        series_dates.append(date_end)

    if not series_dates:
        raise ValueError("No test windows generated for backtest.")

    df_bt = pd.DataFrame(
        {
            "date": series_dates,
            "buyhold_equity": equity_bh[1:],
            "strategy_equity": equity_strat[1:],
        }
    )

    total_ret_strat = df_bt["strategy_equity"].iloc[-1] - 1.0
    total_ret_bh = df_bt["buyhold_equity"].iloc[-1] - 1.0

    # CAGR sobre el període de test
    n_days = (df_bt["date"].iloc[-1] - df_bt["date"].iloc[0]).days
    years = max(n_days / 365.25, 1e-9)
    cagr_strat = df_bt["strategy_equity"].iloc[-1] ** (1.0 / years) - 1.0

    hit_rate = (n_wins / n_trades) if n_trades > 0 else 0.0

    results = {
        "total_return_strategy": float(total_ret_strat),
        "total_return_buyhold": float(total_ret_bh),
        "cagr_strategy": float(cagr_strat),
        "hit_rate": float(hit_rate),
        "n_trades": int(n_trades),
        "test_days": int(n_days),
    }
    return results, df_bt

def run_long_flat_backtest(
    df: pd.DataFrame,
    horizon: int,
    model,
    threshold: float = 0.55,
    cost_bps_per_side: float = 0.0,
    test_days: int = 365,
):
    """
    Long/flat backtest utilitzant el model direccional.

    Estratègia:
      - Cada dia de la finestra de test construïm features per a horitzó `horizon`.
      - Si P(up) >= threshold -> LONG; si no -> FLAT.
      - Retorn log d'estratègia per pas = position * y_ret_{horizon}d.
      - Apliquem cost de trading per side en entrades i sortides.

    Retorna dict amb:
      - dates
      - strategy_equity, buyhold_equity
      - total_return_strategy, total_return_buyhold
      - cagr_strategy
      - hit_rate, n_trades
      - ann_vol, sharpe
      - max_dd_strategy, max_dd_buyhold
      - threshold, cost_bps_per_side
    """
    # Construïm matriu de features i targets per a aquest horitzó
    X_all, y_all, dates_all, feature_cols, target_col = build_feature_matrix(
        df, horizon=horizon
    )
    if X_all is None or X_all.empty:
        return {}

    dates_all = pd.to_datetime(dates_all)

    # Finestra de test: últims `test_days`
    last_date = dates_all.max()
    test_start = last_date - pd.Timedelta(days=test_days)
    mask = dates_all >= test_start

    X_test = X_all.loc[mask]
    y_test = y_all.loc[mask]
    dates_test = dates_all.loc[mask]

    if X_test.empty:
        return {}

    # Probabilitat de pujada
    proba_up = model.predict_proba(X_test)[:, 1]
    signal = (proba_up >= threshold).astype(int)

    # Sèries amb índex temporal
    position = pd.Series(signal, index=dates_test, name="position")
    ret = pd.Series(y_test.values, index=dates_test, name="log_ret")

    # Retorns de l'estratègia (log)
    strategy_ret = position * ret

    # Costos de trading per side (entrada + sortida)
    if cost_bps_per_side > 0:
        cost = cost_bps_per_side / 10000.0
        log_cost = np.log(1.0 - cost)

        pos_prev = position.shift(1).fillna(0)
        entries = (position == 1) & (pos_prev == 0)
        exits = (position == 0) & (pos_prev == 1)

        # Apliquem cost a entries i exits
        strategy_ret.loc[entries | exits] += log_cost

    # Equity (normalitzada a 1)
    equity_strategy = np.exp(strategy_ret.cumsum())
    equity_buyhold = np.exp(ret.cumsum())

    # Total return (en %)
    total_ret_strategy = float(equity_strategy.iloc[-1] - 1.0)
    total_ret_buyhold = float(equity_buyhold.iloc[-1] - 1.0)

    # Passos per any (aprox) segons horitzó
    steps_per_year = 365.0 / float(horizon)

    mean_step_ret = float(strategy_ret.mean())
    # CAGR aproximant distribució d'aquests passos
    cagr_strategy = np.exp(mean_step_ret * steps_per_year) - 1.0

    # Hit rate: % de trades LONG amb retorn positiu
    trades_mask = position == 1
    n_trades = int(trades_mask.sum())
    if n_trades > 0:
        hits = int((ret[trades_mask] > 0).sum())
        hit_rate = hits / n_trades
    else:
        hit_rate = np.nan

    # Volatilitat anualitzada & Sharpe aproximat
    step_vol = float(strategy_ret.std())
    if step_vol > 0:
        ann_vol = step_vol * np.sqrt(steps_per_year)
        sharpe = (mean_step_ret * steps_per_year) / ann_vol
    else:
        ann_vol = np.nan
        sharpe = np.nan

    # Max drawdown helper
    def _max_drawdown(equity: pd.Series) -> float:
        running_max = equity.cummax()
        dd = equity / running_max - 1.0
        return float(dd.min())  # número negatiu (ex: -0.35 = -35%)

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

        st.altair_chart(base_regime_chart + bull_points + bear_points, use_container_width=True)
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
                
                # ---- Backtest del model direccional ----
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
                    - Trades NO solapats: 1 trade cada `horizon` dies
                    - Long si proba_up >= threshold, flat si no
                    - Usa y_ret_{h}d com a log-return per trade
                
                    Retorna:
                        backtest_df: DataFrame amb sèries d'equity i senyals
                        stats: dict amb total_return, buyhold_return, cagr, hit_rate, n_trades, threshold, max_drawdown
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
                
                    # Període de test segons metrics['test_start']
                    test_start = metrics.get("test_start", None)
                    if test_start is not None:
                        test_start = pd.to_datetime(test_start)
                        base_mask = dates >= test_start
                    else:
                        # fallback: últim 30%
                        n = len(X)
                        split_idx = int(n * 0.7)
                        base_mask = np.zeros(n, dtype=bool)
                        base_mask[split_idx:] = True
                
                    idx_all = np.where(base_mask)[0]
                    if len(idx_all) == 0:
                        return None, {}
                
                    # 👉 Mostres NO solapades: agafem un cada `horizon`
                    idx_sampled = idx_all[::horizon]
                
                    X_test = X.iloc[idx_sampled]
                    dates_start = dates.iloc[idx_sampled].reset_index(drop=True)
                    rets_test = rets.iloc[idx_sampled].reset_index(drop=True)
                
                    # Assignem la data del trade al FINAL de l'horitzó (t + h)
                    dates_trade = dates_start + pd.to_timedelta(horizon, unit="D")
                
                    if len(X_test) == 0:
                        return None, {}
                
                    proba = model.predict_proba(X_test)[:, 1]
                
                    if threshold_type == "f1":
                        thr = metrics.get("best_f1_thr", 0.5)
                    else:
                        thr = metrics.get("best_bal_acc_thr", 0.5)
                
                    thr = float(thr if thr is not None else 0.5)
                
                    # Estratègia long / flat sobre trades discretitzats
                    pos = (proba >= thr).astype(float)
                    strat_log_ret = pos * rets_test.values
                    buyhold_log_ret = rets_test.values
                
                    cum_strat = np.cumsum(strat_log_ret)
                    cum_buy = np.cumsum(buyhold_log_ret)
                
                    equity_strat = np.exp(cum_strat)
                    equity_buy = np.exp(cum_buy)
                
                    backtest_df = pd.DataFrame(
                        {
                            "date": dates_trade.values,
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
                
                    n_trades = int((pos != 0).sum())
                
                    # CAGR sobre el temps real cobert pels trades
                    n_periods = len(dates_trade)
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
                
                    # Max drawdown de l'estratègia
                    peak = np.maximum.accumulate(equity_strat)
                    dd = (equity_strat / peak) - 1.0
                    max_dd = float(dd.min())
                
                    stats = {
                        "total_return": total_return_strat,
                        "buyhold_return": total_return_buy,
                        "cagr": float(cagr),
                        "hit_rate": hit_rate,
                        "n_trades": n_trades,
                        "threshold": thr,
                        "max_drawdown": max_dd,
                    }
                
                    return backtest_df, stats



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
    # BACKTEST – STRATEGY USING MODEL SIGNAL (LONG / FLAT)
    # -----------------------------------------------------------------
    if horizon in models:
        st.subheader("Backtest – strategy using model signal (long / flat)")

        # Intentem agafar el millor llindar que vam calcular al training
        m = metrics_all.get(horizon, {})
        thr = 0.5
        if isinstance(m, dict):
            thr = m.get("opt_threshold_bal", 0.5)

        try:
            bt_results, df_bt = backtest_long_flat(
                df=df,
                horizon=horizon,
                model=models[horizon],
                threshold=thr,
                test_size_days=365,
            )
        except Exception as e:
            st.info(f"Backtest not available for this horizon: {e}")
        else:
            # KPIs de l'estratègia
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Total return (strategy)",
                f"{bt_results['total_return_strategy'] * 100:,.1f} %",
            )
            c2.metric(
                "Total return (buy & hold)",
                f"{bt_results['total_return_buyhold'] * 100:,.1f} %",
            )
            c3.metric(
                "CAGR (strategy)",
                f"{bt_results['cagr_strategy'] * 100:,.1f} %",
            )
            c4.metric(
                "Hit rate (on trades)",
                f"{bt_results['hit_rate'] * 100:,.1f} %",
            )

            # Equity curve
            df_bt_long = df_bt.melt(
                "date", var_name="series", value_name="equity"
            )

            bt_chart = (
                alt.Chart(df_bt_long)
                .mark_line()
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("equity:Q", title="Equity (normalized)"),
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(
                            domain=["buyhold_equity", "strategy_equity"],
                            range=["#1f77b4", "#aec7e8"],
                        ),
                        title="Series",
                    ),
                    tooltip=["date:T", "series:N", "equity:Q"],
                )
                .properties(height=400)
            )
            st.altair_chart(bt_chart, use_container_width=True)


    # -----------------------------------------------------------------
    # BACKTEST – STRATEGY USING MODEL SIGNAL
    # -----------------------------------------------------------------
    st.subheader("Backtest – strategy using model signal (long / flat)")

    if horizon not in models:
        st.info("No trained model available for this horizon.")
    else:
        bt_res = run_long_flat_backtest(
            df=df,
            horizon=horizon,
            model=models[horizon],
            threshold=trade_threshold,
            cost_bps_per_side=trade_cost_bps,
            test_days=365,  # últim any
        )

        if not bt_res:
            st.info("Not enough data to run backtest for this horizon.")
        else:
            bt = bt_res

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
                f"Backtest run on the last 365 days with a {horizon}-day horizon. "
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
