import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys
from pathlib import Path

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
)

PROCESSED_PATH = Path("data/processed/btc_dataset.parquet")
RAW_PATH = Path("data/raw/btc_price_daily.csv")


# ---------------------------------------------------------------------
# Dataset loader (live + offline fallback), cachejat
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def load_dataset(use_live: bool = True):
    """
    Load or build the BTC dataset.

    Returns:
        df: DataFrame
        source: "live", "offline" or "none"
    """
    # 1) LIVE MODE: CoinDesk Data API + live factors
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

    # 2) OFFLINE MODE: build from local CSV
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
# Model training (cachejat)
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def train_all_models(df: pd.DataFrame):
    """
    Train directional models for 1d / 7d / 30d / 90d horizons.

    Returns:
        models: dict[horizon -> sklearn Pipeline]
        metrics_all: dict[horizon -> metrics dict]
        metas: dict[horizon -> ModelMeta]
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

    Returns:
        trend_models: {"bull": {h: model}, "bear": {h: model}}
        trend_metrics: {"bull": {h: metrics}, "bear": {h: metrics}}
        trend_metas: {"bull": {h: meta}, "bear": {h: meta}}
    """
    return fit_all_trend_change_models(
        df,
        horizons=[7, 30, 90],
        test_size_days=365,
    )


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Bitcoin Predictive Model – Data & Trend Explorer",
        layout="wide",
    )

    st.title("Bitcoin Predictive Model – Data & Trend Explorer")

    # ---- Sidebar (top) ----
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

    # ---- Load dataset ----
    with st.spinner("Building dataset..."):
        df, source = load_dataset(use_live=use_live)

    if df is None or df.empty:
        st.error("Dataset is empty.")
        st.stop()

    # ---- Train models (directional up/down) ----
    with st.spinner("Training directional models (1d / 7d / 30d / 90d)..."):
        models, metrics_all, metas = train_all_models(df)

    # ---- Train trend-change models ----
    with st.spinner("Training trend-change models (bull/bear)..."):
        trend_models, trend_metrics, trend_metas = train_all_trend_models(df)

    # --- Safety: ensure trend_models & trend_metrics always have 'bull'/'bear' keys ---
    if trend_models is None or not isinstance(trend_models, dict):
        trend_models = {}
    if "bull" not in trend_models:
        trend_models["bull"] = {}
    if "bear" not in trend_models:
        trend_models["bear"] = {}

    if trend_metrics is None or not isinstance(trend_metrics, dict):
        trend_metrics = {}
    if "bull" not in trend_metrics:
        trend_metrics["bull"] = {}
    if "bear" not in trend_metrics:
        trend_metrics["bear"] = {}

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

    # ---- Debug expanders (optional) ----
    with st.expander("Debug: first rows of dataset"):
        st.dataframe(df.head(10))

    with st.expander("Debug: last rows of dataset"):
        st.dataframe(df.tail(10))

    with st.expander("Debug: columns & dtypes"):
        st.write(df.dtypes)

    # ---- Sidebar (footer info) ----
    st.sidebar.markdown("---")
    st.sidebar.write(f"Rows in dataset: **{len(df):,}**")

    # -----------------------------------------------------------------
    # PRICE CHART
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # TREND REGIME & TURNING POINTS
    # -----------------------------------------------------------------
    if "regime_smooth" in df.columns:
        st.subheader("Trend regime & turning points")

        df_reg = df[["date", "close", "regime_smooth"]].dropna().copy()
        df_reg = df_reg.sort_values("date").reset_index(drop=True)

        # Identify regime changes
        df_reg["regime_change"] = df_reg["regime_smooth"].diff().fillna(0)

        # Bull turns: regime_smooth becomes 1 from <= 0
        bull_turns = df_reg[
            (df_reg["regime_smooth"] == 1)
            & (df_reg["regime_change"] > 0)
        ]

        # Bear turns: regime_smooth becomes -1 from >= 0
        bear_turns = df_reg[
            (df_reg["regime_smooth"] == -1)
            & (df_reg["regime_change"] < 0)
        ]

        # Map regime to label for tooltips
        def _regime_label(x: float) -> str:
            if x >= 0.5:
                return "Bull"
            elif x <= -0.5:
                return "Bear"
            else:
                return "Sideways"

        df_reg["regime_label"] = df_reg["regime_smooth"].apply(_regime_label)

        # Base chart: price colored by regime
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
                tooltip=[
                    "date:T",
                    "close:Q",
                    "regime_label:N",
                ],
            )
            .properties(height=400)
        )

        # Markers for turning points
        bull_points = (
            alt.Chart(bull_turns)
            .mark_point(shape="triangle-up", size=80, filled=True, color="#2ca02c")
            .encode(
                x="date:T",
                y="close:Q",
                tooltip=["date:T", "close:Q"],
            )
        )

        bear_points = (
            alt.Chart(bear_turns)
            .mark_point(shape="triangle-down", size=80, filled=True, color="#d62728")
            .encode(
                x="date:T",
                y="close:Q",
                tooltip=["date:T", "close:Q"],
            )
        )

        st.altair_chart(
            base_regime_chart + bull_points + bear_points,
            use_container_width=True,
        )
        st.caption(
            "Green ▲ = start of **bull** regime · Red ▼ = start of **bear** regime"
        )

        # ---- Current regime summary ----
        latest_row = df_reg.iloc[-1]
        latest_regime = latest_row["regime_smooth"]
        latest_label = _regime_label(latest_regime)
        latest_date = latest_row["date"]

        # Find start date of current regime
        last_change_idx = (
            df_reg.index[df_reg["regime_change"] != 0].max()
            if (df_reg["regime_change"] != 0).any()
            else None
        )

        if last_change_idx is None:
            start_date = df_reg["date"].min()
        else:
            # regime started on the next row after the last change
            start_idx = min(last_change_idx + 1, len(df_reg) - 1)
            start_date = df_reg.loc[start_idx, "date"]

        days_in_regime = (latest_date - start_date).days

        color_map = {
            "Bull": "#2ca02c",     # verd
            "Bear": "#d62728",     # vermell
            "Sideways": "#7f7f7f", # gris
        }
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

    # -----------------------------------------------------------------
    # TREND-CHANGE SIGNALS (MODELS)
    # -----------------------------------------------------------------
    st.subheader("Trend-change signals (bull / bear)")

    # Use same horizon from sidebar, but enforce minimum 7 days for trend models
    trend_h = max(horizon, 7)

    if "regime_smooth" not in df.columns:
        st.info(
            "Trend-change models require 'regime_smooth' in the dataset. "
            "Make sure trend_regime.add_trend_regime_block() is applied."
        )
    else:
        # Build feature matrix for bull / bear turn for this horizon
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

            # Latest state
            idx_bull = len(X_bull) - 1
            idx_bear = len(X_bear) - 1

            date_ref = dates_bull.iloc[idx_bull]

            p_bull = bull_model.predict_proba(X_bull.iloc[[idx_bull]])[0, 1]
            p_bear = bear_model.predict_proba(X_bear.iloc[[idx_bear]])[0, 1]

            m_bull = trend_metrics["bull"].get(trend_h, {})
            m_bear = trend_metrics["bear"].get(trend_h, {})

            c1, c2 = st.columns(2)

            c1.metric(
                label=f"Prob. start of BULL regime in next {trend_h} days",
                value=f"{p_bull * 100:.1f} %",
            )
            if m_bull:
                c1.write(
                    f"Test accuracy: {m_bull.get('test_accuracy', float('nan')):.3f} · "
                    f"AUC: {m_bull.get('test_auc', float('nan')):.3f}"
                )

            c2.metric(
                label=f"Prob. start of BEAR regime in next {trend_h} days",
                value=f"{p_bear * 100:.1f} %",
            )
            if m_bear:
                c2.write(
                    f"Test accuracy: {m_bear.get('test_accuracy', float('nan')):.3f} · "
                    f"AUC: {m_bear.get('test_auc', float('nan')):.3f}"
                )

            st.caption(
                f"Reference date for probabilities: {date_ref.date()}. "
                f"Targets are: bull_turn_{trend_h}d / bear_turn_{trend_h}d."
            )

    # -----------------------------------------------------------------
    # HORIZON STATS + FORECAST RANGE + MODEL PROBABILITY
    # -----------------------------------------------------------------
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

            # Last actual price
            last_row = df.dropna(subset=["close"]).iloc[-1]
            last_date = last_row["date"]
            last_price = last_row["close"]

            # Forecast endpoint
            forecast_end_date = last_date + pd.Timedelta(days=horizon)
            expected_price = last_price * np.exp(mean_ret)

            # 95% price range based on historical log-return distribution
            low_log = mean_ret - 1.96 * std_ret
            high_log = mean_ret + 1.96 * std_ret
            price_low = last_price * np.exp(low_log)
            price_high = last_price * np.exp(high_log)

    if mean_ret is None or valid is None or valid.empty:
        st.warning(f"Column `{ret_col}` not found or not enough data.")
    else:
        cols = st.columns(4)

        cols[0].metric(
            "Mean log-return",
            f"{mean_ret:.4f}",
        )
        cols[1].metric(
            "Std. of log-return",
            f"{std_ret:.4f}",
        )
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

                st.markdown("### Model-based directional signal")
                st.metric(
                    label=f"Prob. BTC higher in {horizon} days (from {latest_date.date()})",
                    value=f"{proba_up * 100:.1f} %",
                )
                st.write(
                    f"**Test accuracy:** {m['test_accuracy']:.3f}  |  "
                    f"**Test AUC:** {m['test_auc']:.3f}  |  "
                    f"**Test Brier:** {m['test_brier']:.3f}"
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
    # MODEL PERFORMANCE SUMMARY (DIRECTIONAL)
    # -----------------------------------------------------------------
    st.subheader("Model performance summary – directional (up / down)")

    dir_rows = []
    for h in sorted(metrics_all.keys()):
        m = metrics_all[h]
        dir_rows.append(
            {
                "horizon_days": h,
                "test_accuracy": m.get("test_accuracy", np.nan),
                "test_auc": m.get("test_auc", np.nan),
                "test_brier": m.get("test_brier", np.nan),
            }
        )

    if dir_rows:
        df_dir_metrics = pd.DataFrame(dir_rows)
        st.dataframe(df_dir_metrics.style.format(
            {
                "test_accuracy": "{:.3f}",
                "test_auc": "{:.3f}",
                "test_brier": "{:.3f}",
            }
        ))
    else:
        st.info("No directional model metrics available.")


    # -----------------------------------------------------------------
    # MODEL PERFORMANCE SUMMARY (TREND-CHANGE)
    # -----------------------------------------------------------------
    st.subheader("Model performance summary – trend-change (bull / bear)")

    trend_rows = []
    for direction in ["bull", "bear"]:
        models_dict = trend_models.get(direction, {})
        metrics_dict = trend_metrics.get(direction, {})
        for h, m in metrics_dict.items():
            trend_rows.append(
                {
                    "direction": direction,
                    "horizon_days": h,
                    "test_accuracy": m.get("test_accuracy", np.nan),
                    "test_auc": m.get("test_auc", np.nan),
                }
            )

    if trend_rows:
        df_trend_metrics = pd.DataFrame(trend_rows)
        df_trend_metrics = df_trend_metrics.sort_values(
            ["direction", "horizon_days"]
        ).reset_index(drop=True)
        st.dataframe(df_trend_metrics.style.format(
            {
                "test_accuracy": "{:.3f}",
                "test_auc": "{:.3f}",
            }
        ))
    else:
        st.info(
            "No trend-change model metrics available. "
            "This can happen if there are very few bull/bear turn events "
            "for the selected horizons."
        )



    # -----------------------------------------------------------------
    # FACTOR EXPLORER
    # -----------------------------------------------------------------
    st.subheader("Factor Explorer")

    # Identify numeric columns
    ignore_exact = {
        "date",
        "open", "high", "low", "close", "volume", "quote_volume",
        "log_ret_1d", "log_ret_3d", "log_ret_7d",
        "vol_7d", "vol_30d",
        "ma_20", "ma_50", "ma_90",
        "price_over_ma20", "price_over_ma50", "price_over_ma90",
        "drawdown_90d",
        "regime_raw", "regime_smooth",
        "days_since_last_halving", "days_to_next_halving",
    }
    ignore_prefixes = (
        "y_ret_", "up_",
        "bull_turn_", "bear_turn_",
    )

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
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
                st.warning("No overlapping data for this factor and return horizon.")
            else:
                # Time series chart of the factor
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

                # Scatter vs future returns
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

    # -----------------------------------------------------------------
    # RAW DATA PREVIEW
    # -----------------------------------------------------------------
    st.subheader("Raw data preview")
    st.dataframe(df.tail(20))


if __name__ == "__main__":
    main()
