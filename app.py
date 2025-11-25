import streamlit as st
import pandas as pd
from pathlib import Path
import altair as alt
import numpy as np
import sys

# Make sure we can import from src/
sys.path.append("src")
from build_dataset import build_btc_dataset  # our dataset builder


RAW_PATH = Path("data/raw/btc_price_daily.csv")          # adjust name if needed
PROCESSED_PATH = Path("data/processed/btc_dataset.parquet")


@st.cache_data
def load_dataset():
    """
    Load the processed dataset if it exists.
    If not, try to build it from raw price data using build_btc_dataset.
    """
    # 1) If processed already exists, just load it
    if PROCESSED_PATH.exists():
        df = pd.read_parquet(PROCESSED_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # 2) If no processed file but we DO have raw prices, build it now
    if RAW_PATH.exists():
        build_btc_dataset(
            price_csv_path=str(RAW_PATH),
            output_path=str(PROCESSED_PATH),
            live=True
        )
        df = pd.read_parquet(PROCESSED_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # 3) If neither exists, we can’t do anything
    return None


def main():
    st.set_page_config(
        page_title="BTC Predictive Model – Data & Trend Explorer",
        layout="wide",
    )

    st.title("Bitcoin Predictive Model – Data & Trend Explorer")

    df = load_dataset()

    if df is None:
        st.error(
            "Could not find either:\n"
            "- processed dataset at `data/processed/btc_dataset.parquet`, nor\n"
            "- raw prices at `data/raw/btc_price_daily.csv`.\n\n"
            "Please add a raw BTC price CSV with columns `date` and `close`."
        )
        st.stop()

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
    st.subheader("Price & Trend Regime")

    df_plot = df[["date", "close", "regime_smooth"]].copy()
    df_plot["regime_label"] = df_plot["regime_smooth"].map(
        {1: "Bull", 0: "Neutral", -1: "Bear"}
    )

    price_chart = (
        alt.Chart(df_plot)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("close:Q", title="BTC Price (close)"),
            color=alt.Color(
                "regime_label:N",
                title="Regime",
                scale=alt.Scale(
                    domain=["Bear", "Neutral", "Bull"],
                    range=["#d62728", "#7f7f7f", "#2ca02c"],
                ),
            ),
            tooltip=["date:T", "close:Q", "regime_label:N"],
        )
        .properties(height=400)
    )

    if forecast_line is not None:
        forecast_chart = (
            alt.Chart(forecast_line)
            .mark_line(color="#ff7f0e", strokeWidth=3)
            .encode(
                x="date:T",
                y=alt.Y("price:Q", title="BTC Price (forecast)"),
                tooltip=["date:T", "price:Q"],
            )
        )
        st.altair_chart(price_chart + forecast_chart, use_container_width=True)
    else:
        st.altair_chart(price_chart, use_container_width=True)

    # ---- Optional: raw vs smoothed regime ----
    if show_raw_regime and "regime_raw" in df.columns:
        st.subheader("Raw vs Smoothed Regime (last 200 days)")

        recent = df.tail(200).copy()
        recent["regime_raw_label"] = recent["regime_raw"].map(
            {1: "Bull", 0: "Neutral", -1: "Bear"}
        )
        recent["regime_smooth_label"] = recent["regime_smooth"].map(
            {1: "Bull", 0: "Neutral", -1: "Bear"}
        )

        raw_chart = (
            alt.Chart(recent)
            .mark_circle(size=40)
            .encode(
                x="date:T",
                y=alt.value(0),
                color=alt.Color(
                    "regime_raw_label:N",
                    scale=alt.Scale(
                        domain=["Bear", "Neutral", "Bull"],
                        range=["#d62728", "#7f7f7f", "#2ca02c"],
                    ),
                ),
                tooltip=["date:T", "regime_raw_label:N"],
            )
            .properties(title="Raw regime", height=120)
        )

        smooth_chart = (
            alt.Chart(recent)
            .mark_circle(size=40)
            .encode(
                x="date:T",
                y=alt.value(0),
                color=alt.Color(
                    "regime_smooth_label:N",
                    scale=alt.Scale(
                        domain=["Bear", "Neutral", "Bull"],
                        range=["#d62728", "#7f7f7f", "#2ca02c"],
                    ),
                ),
                tooltip=["date:T", "regime_smooth_label:N"],
            )
            .properties(title="Smoothed regime", height=120)
        )

        st.altair_chart(raw_chart, use_container_width=True)
        st.altair_chart(smooth_chart, use_container_width=True)

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
    st.subheader("Factor explorer")

    # Columns that are *not* factors (structural / internal stuff)
    ignore_exact = {
        "date", "close",
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

    # Any numeric column that is not in the ignore list is treated as a factor
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

    if not factor_cols:
        st.info(
            "No external factor columns detected yet.\n\n"
            "If you are using live APIs, make sure they are actually merged in "
            "build_btc_dataset (fear & greed, activity, ETF flows, equities, etc.)."
        )
    else:
        selected_factor = st.selectbox(
            "Select factor to visualize:",
            options=sorted(factor_cols),
        )

        factor_df = df[["date", selected_factor]].dropna()

        if factor_df.empty:
            st.warning(f"No non-NaN data for factor `{selected_factor}`.")
        else:
            factor_chart = (
                alt.Chart(factor_df)
                .mark_line()
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y(selected_factor + ":Q", title=selected_factor),
                    tooltip=["date:T", selected_factor + ":Q"],
                )
                .properties(height=300)
            )
            st.altair_chart(factor_chart, use_container_width=True)

            # Optional: correlation with future returns for the selected horizon
            if ret_col in df.columns:
                merged = df[["date", ret_col]].merge(
                    factor_df, on="date", how="inner"
                ).dropna()
                if not merged.empty:
                    corr = merged[ret_col].corr(merged[selected_factor])
                    st.caption(
                        f"Correlation between `{selected_factor}` and `{ret_col}` "
                        f"(where both are available): **{corr:.3f}**"
                    )

    # ---- Raw data preview ----
    st.subheader("Raw data preview")
    st.dataframe(df.tail(20))


if __name__ == "__main__":
    main()
