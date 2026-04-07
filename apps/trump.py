import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Post–Market Reaction Explorer", layout="wide")

st.markdown("""
<style>
.big-title {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.subtle {
    color: #666;
    font-size: 14px;
}
.card {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #f8f9fa;
    border: 1px solid #e1e4e8;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>Post–Market Reaction Explorer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtle'>Analyze how public statements correlate with index movements (descriptive only, no predictions).</div>", unsafe_allow_html=True)
st.markdown("---")

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV of posts (timestamp,text)",
    type=["csv"],
    help="Must contain 'timestamp' and 'text' columns."
)

default_tickers = ["SPY", "QQQ", "DIA", "^VIX"]
tickers = st.sidebar.multiselect(
    "Market indices / ETFs",
    default_tickers,
    default=default_tickers
)

lookahead_minutes = st.sidebar.selectbox(
    "Window after post to measure reaction",
    [5, 15, 30, 60, 240, 1440],
    index=3,
    format_func=lambda x: f"{x} minutes" if x < 60 else (
        f"{x//60} hours" if x < 1440 else "1 day"
    )
)

start_date = st.sidebar.date_input("Start date", value=datetime(2024, 1, 1))
end_date = st.sidebar.date_input("End date", value=datetime.now().date())

st.sidebar.markdown("---")
st.sidebar.caption("This app is for historical, descriptive analysis only. It does not predict or recommend trades.")

# =========================
# HELPERS
# =========================
@st.cache_data
def load_posts(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "timestamp" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain 'timestamp' and 'text' columns.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "text"])
    df = df.sort_values("timestamp")
    return df

@st.cache_data
def fetch_market_data(tickers, start, end):
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers,
        start=start,
        end=end + timedelta(days=1),
        interval="1m" if (end - start).days <= 7 else "5m"
    )
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    data = data.tz_localize("UTC", level=None, nonexistent="shift_forward", ambiguous="NaT", errors="ignore")
    return data

def sentiment_score(text: str) -> float:
    return TextBlob(str(text)).sentiment.polarity

def compute_reactions(posts_df, market_df, lookahead_minutes):
    rows = []
    delta = timedelta(minutes=lookahead_minutes)

    for _, row in posts_df.iterrows():
        ts = row["timestamp"]
        text = row["text"]
        end_ts = ts + delta

        # Slice market window
        window = market_df.loc[(market_df.index >= ts) & (market_df.index <= end_ts)]
        if window.empty:
            continue

        start_prices = market_df.loc[market_df.index >= ts].head(1)
        if start_prices.empty:
            continue

        start_prices = start_prices.iloc[0]
        end_prices = window.iloc[-1]

        changes = (end_prices - start_prices) / start_prices * 100.0

        rows.append({
            "timestamp": ts,
            "text": text,
            "sentiment": sentiment_score(text),
            **{f"{col}_pct_change": changes[col] for col in changes.index}
        })

    return pd.DataFrame(rows)

# =========================
# MAIN LOGIC
# =========================
col_main, col_side = st.columns([3, 1])

with col_main:
    st.subheader("1. Load Posts")

    if uploaded_file is None:
        st.info("Upload a CSV of posts in the sidebar to begin. Example columns: `timestamp,text`.")
        st.stop()

    try:
        posts_df = load_posts(uploaded_file)
    except Exception as e:
        st.error(f"Error loading posts: {e}")
        st.stop()

    st.markdown("**Sample of loaded posts:**")
    st.dataframe(posts_df.head(10), use_container_width=True)

    st.subheader("2. Fetch Market Data")
    if not tickers:
        st.warning("Select at least one ticker in the sidebar.")
        st.stop()

    with st.spinner("Downloading market data..."):
        market_df = fetch_market_data(tickers, start_date, end_date)

    if market_df.empty:
        st.error("No market data returned. Try adjusting date range or tickers.")
        st.stop()

    st.markdown("**Market data preview:**")
    st.dataframe(market_df.head(10), use_container_width=True)

    st.subheader("3. Align Posts with Market Reactions")

    with st.spinner("Computing reactions..."):
        reactions_df = compute_reactions(posts_df, market_df, lookahead_minutes)

    if reactions_df.empty:
        st.warning("No overlapping data between posts and market window. Check dates and tickers.")
        st.stop()

    st.markdown("**Post–reaction dataset:**")
    st.dataframe(reactions_df.head(20), use_container_width=True)

    st.subheader("4. Sentiment vs. Market Reaction")

    numeric_cols = [c for c in reactions_df.columns if c.endswith("_pct_change")]
    if not numeric_cols:
        st.warning("No percentage change columns found.")
    else:
        selected_metric = st.selectbox(
            "Choose a market reaction metric",
            numeric_cols,
            index=0
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=reactions_df,
            x="sentiment",
            y=selected_metric,
            ax=ax
        )
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"Sentiment vs. {selected_metric}")
        ax.set_xlabel("Sentiment (TextBlob polarity)")
        ax.set_ylabel("Percent change (%)")
        st.pyplot(fig)

        corr = reactions_df["sentiment"].corr(reactions_df[selected_metric])
        st.markdown(f"**Correlation between sentiment and {selected_metric}:** `{corr:.3f}` (descriptive only)")

    st.subheader("5. Time Series View Around Posts")

    # Plot one ticker around posts
    ts_ticker = st.selectbox("Time series ticker", tickers, index=0)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    market_df[ts_ticker].plot(ax=ax2, color="tab:blue", alpha=0.8)
    ax2.set_title(f"{ts_ticker} Close Price with Post Markers")
    ax2.set_ylabel("Price")
    for ts in reactions_df["timestamp"]:
        ax2.axvline(ts, color="red", alpha=0.3, linestyle="--")
    st.pyplot(fig2)

with col_side:
    st.subheader("Summary")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"**Total posts loaded:** {len(posts_df)}")
    st.markdown(f"**Posts with matched market windows:** {len(reactions_df)}")
    st.markdown(f"**Tickers analyzed:** {', '.join(tickers)}")
    st.markdown(f"**Lookahead window:** {lookahead_minutes} minutes")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Notes**")
    st.markdown("- This app is descriptive, not predictive.")
    st.markdown("- Correlations do not imply causation.")
    st.markdown("- Market reactions can be driven by many factors.")
    st.markdown("</div>", unsafe_allow_html=True)
