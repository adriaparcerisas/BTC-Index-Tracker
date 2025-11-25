import streamlit as st
import pandas as pd
from pathlib import Path
import altair as alt
import sys

# Make sure we can import from src/
sys.path.append("src")
from build_dataset import build_btc_dataset  # function we wrote earlier

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
        # You’ll see a spinner in the UI while it builds
        build_btc_dataset(
            price_csv_path=str(RAW_PATH),
            output_path=str(PROCESSED_PATH),
            # other args left as default (threshold, horizons, etc.)
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
            "Please add a raw BTC price CSV to `data/raw/` with columns `date` and `close`."
        )
        st.stop()

    # Sidebar controls
    st.sidebar.header("Options")

    horizon = st.sidebar.selectbox(
        "Prediction horizon (for targets):",
        options=[1, 7, 30, 90],
        index=1,
        format_func=lambda h: f"{h} days",
    )

    show_raw_regime = st.sidebar.checkbox("Show raw regime", value=False)

    st.sidebar.markdown("---")
    st.sidebar.write(f"Rows in dataset: **{len(df):,}**")

    # ----- Price & Regime chart -----
    st.subheader("Price & Trend Regime")

    df_plot = df[["date", "close", "regime_smooth"]].copy()
    df_plot["regime_label"] = df_plot["regime_smooth"].map(
        {1: "Bull", 0: "Neutral", -1: "Bear"}
    )

    # Price line colored by regime
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

    st.altair_chart(price_chart, use_container_width=True)

    if show_raw_regime and "regime_raw" in df.columns:
        st.subheader("Raw vs Smoothed Regime (recent 200 days)")

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

    # ----- Horizon targets summary -----
    st.subheader(f"Targets for {horizon}-day horizon")

    ret_col = f"y_ret_{horizon}d"
    up_col = f"up_{horizon}d"

    if ret_col not in df.columns:
        st.warning(f"Column `{ret_col}` not found in dataset.")
    else:
        # Basic stats
        valid = df.dropna(subset=[ret_col])
        mean_ret = valid[ret_col].mean()
        std_ret = valid[ret_col].std()
        up_ratio = valid[up_col].mean() if up_col in valid.columns else None

        cols = st.columns(3)
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

        # Distribution chart
        st.caption("Distribution of future log-returns")
        hist = (
            alt.Chart(valid)
            .mark_bar()
            .encode(
                x=alt.X(ret_col, bin=alt.Bin(maxbins=50), title=f"log-return over {horizon} days"),
                y=alt.Y("count()", title="Count"),
            )
            .properties(height=300)
        )
        st.altair_chart(hist, use_container_width=True)

    # ----- Raw data preview -----
    st.subheader("Raw data preview")
    st.dataframe(df.tail(20))


if __name__ == "__main__":
    main()
