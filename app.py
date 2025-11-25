import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys
from pathlib import Path

# 🟢 Make sure Python can import from src/
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from build_dataset import (
    build_btc_dataset_live,
    build_btc_dataset_from_csv,
)

PROCESSED_PATH = Path("data/processed/btc_dataset.parquet")
RAW_PATH = Path("data/raw/btc_price_daily.csv")

# Make sure we can import from src/
sys.path.append("src")

@st.cache_data(ttl=3600)
def load_dataset():
    """
    Try to build the BTC dataset using live APIs (DIA, etc.).
    If that fails, fall back to local parquet/CSV if available.
    Cached for 1 hour.
    """
    # 1) Try live mode
    try:
        df = build_btc_dataset_live(price_days=365 * 5)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.warning(
            f"Live data fetch failed: {e}\n\n"
            "Falling back to local dataset if available."
        )

    # 2) Fallback: processed parquet
    if PROCESSED_PATH.exists():
        df = pd.read_parquet(PROCESSED_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # 3) Fallback: raw CSV -> build offline dataset
    if RAW_PATH.exists():
        df = build_btc_dataset_from_csv(
            price_csv_path=str(RAW_PATH),
            output_path=str(PROCESSED_PATH),
        )
        df["date"] = pd.to_datetime(df["date"])
        return df

    # 4) Nothing worked
    st.error(
        "Could not load dataset from live APIs nor from local files.\n\n"
        "Please check:\n"
        "- DIA API availability,\n"
        "- local CSV at `data/raw/btc_price_daily.csv` (with `date` and `close`)."
    )
    return None


def main():
    st.set_page_config(
        page_title="BTC Predictive Model – Data & Trend Explorer",
        layout="wide",
    )

    st.title("Bitcoin Predictive Model – Data & Trend Explorer (Live Data)")

    with st.spinner("Fetching live data and building dataset..."):
        df = load_dataset()

    if df is None or df.empty:
        st.stop()

    # --- DEBUG: basic info ---
    st.caption(
        f"Rows: {len(df):,} | "
        f"date range: {df['date'].min().date()} → {df['date'].max().date()}"
    )

    with st.expander("Debug: first rows of dataset"):
        st.dataframe(df.head(10))

    with st.expander("Debug: last rows of dataset"):
        st.dataframe(df.tail(10))

    with st.expander("Debug: columns & dtypes"):
        st.write(df.dtypes)


    # ---- Sidebar ----
    st.sidebar.header("Options")

    horizon = st.sidebar.selectbox(
        "Prediction horizon:",
        options=[1, 7, 30, 90],
        index=1,
        format_func=lambda h: f"{h} days",
    )

    show_raw_regime = st.sidebar.checkbox("Show raw regime", value=False)

    st.sidebar.markdown("---")
    st.sidebar.write(f"Rows in dataset: **{len(df):,}**")

    # ---- Compute stats & forecast for selected horizon ----
    ret_col = f"y_ret_{horizon}d"
    up_col = f"up_{horizon}d"

    mean_ret = std_ret = up_ratio = None
    forecast_line = None
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

            # Line from last real point to forecast endpoint
            forecast_line = pd.DataFrame(
                {
                    "date": [last_date, forecast_end_date],
                    "price": [last_price, expected_price],
                }
            )

    # ---- Price & Regime chart + forecast line ----
    #st.subheader("Price & Trend Regime")

    # ---- Price chart (minimal version) ----
    st.subheader("BTC price")

    df_price = df[["date", "close"]].dropna().copy()

    if df_price.empty:
        st.warning("No price data available to plot.")
    else:
        basic_price_chart = (
            alt.Chart(df_price)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("close:Q", title="BTC Price (close)"),
                tooltip=["date:T", "close:Q"],
            )
            .properties(height=400)
        )

        st.altair_chart(basic_price_chart, use_container_width=True)


    # ---- Targets summary + price range ----
    st.subheader(f"Targets for {horizon}-day horizon")

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
                "Up probability",
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

    # ---- Factor explorer ----
    st.subheader("Factor Explorer")

    # Identify numeric columns
    ignore_exact = {
        "date",
        "open", "high", "low", "close", "volume",
        "ret_1d", "ret_7d", "ret_30d",
        "mom_30d",
        "ma_7", "ma_30", "ma_90",
        "price_over_ma30", "price_over_ma90", "ma_ratio_30_90",
        "rv_7d", "rv_30d",
        "drawdown_90d",
        "days_since_halving", "cycle_position",
        "regime_raw", "regime_smooth",
        "bull_turn", "bear_turn",
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

    # Debug: show which numeric columns we consider as factors
    with st.expander("Debug: candidate factor columns", expanded=False):
        st.write(factor_cols)

    if not factor_cols:
        st.info(
            "No external factor columns detected yet.\n\n"
            "This is expected if live factor APIs (Fear & Greed, on-chain activity, ETF flows, "
            "equities) are not implemented or failed to fetch. "
            "Right now only BTC price & trend features may be available."
        )
    else:
        factor_name = st.selectbox(
            "Choose a factor to inspect:",
            sorted(factor_cols),
        )

        horizon = st.selectbox(
            "Prediction horizon:",
            [1, 7, 30, 90],
            format_func=lambda h: f"{h} days",
        )
        ret_col = f"y_ret_{horizon}d"

        if ret_col not in df.columns:
            st.warning(f"Return column '{ret_col}' not found in dataset.")
        else:
            df_factor = df[["date", factor_name, ret_col]].dropna().copy()
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
                        y=alt.Y(f"{ret_col}:Q", title=f"log-return {horizon}d ahead"),
                        tooltip=["date:T", f"{factor_name}:Q", f"{ret_col}:Q"],
                    )
                    .properties(height=250)
                )

                corr = df_factor[factor_name].corr(df_factor[ret_col])

                st.write(f"**Factor:** `{factor_name}` | **Horizon:** {horizon} days")
                st.metric("Pearson correlation", f"{corr:.3f}")

                st.altair_chart(factor_chart, use_container_width=True)
                st.altair_chart(scatter, use_container_width=True)


    # ---- Raw data preview ----
    st.subheader("Raw data preview")
    st.dataframe(df.tail(20))


if __name__ == "__main__":
    main()
