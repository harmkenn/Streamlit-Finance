import os
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import yfinance as yf
from textblob import TextBlob

# ============================================================
# CONFIG v1.3
# ============================================================
st.set_page_config(page_title="Post–Market Reaction Explorer", layout="wide")

CACHE_FILE = "trump_posts.csv"

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
st.markdown("<div class='subtle'>Analyze how public statements correlate with index movements (descriptive only).</div>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# LOCAL CACHE FUNCTIONS
# ============================================================
def load_local_posts() -> pd.DataFrame:
    if not os.path.exists(CACHE_FILE):
        return pd.DataFrame(columns=["timestamp", "text"])
    df = pd.read_csv(CACHE_FILE)
    if "timestamp" not in df.columns or "text" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "text"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def save_local_posts(df: pd.DataFrame) -> None:
    df.to_csv(CACHE_FILE, index=False)


# ============================================================
# FETCH NEW POSTS (placeholder)
# Replace this with a real Truth Social fetcher later
# ============================================================
def fetch_new_truth_posts(latest_timestamp: pd.Timestamp | None) -> pd.DataFrame:
    # Always use timezone-aware UTC
    now = datetime.now(timezone.utc)

    # Normalize latest_timestamp to timezone-aware UTC
    if latest_timestamp is not None and latest_timestamp.tzinfo is None:
        latest_timestamp = latest_timestamp.tz_localize("UTC")

    # Simulate a new post if no posts yet or last post older than 1 hour
    if latest_timestamp is None or (now - latest_timestamp).total_seconds() > 3600:
        return pd.DataFrame(
            [
                {
                    "timestamp": now.isoformat(),
                    "text": "Example new post for testing incremental updates.",
                }
            ]
        )

    return pd.DataFrame(columns=["timestamp", "text"])


# ============================================================
# MERGE OLD + NEW POSTS
# ============================================================
def update_posts_cache() -> pd.DataFrame:
    local_df = load_local_posts()

    latest_ts = local_df["timestamp"].max() if not local_df.empty else None
    new_posts = fetch_new_truth_posts(latest_ts)

    if new_posts.empty:
        return local_df

    new_posts["timestamp"] = pd.to_datetime(new_posts["timestamp"], utc=True)

    merged = pd.concat([local_df, new_posts], ignore_index=True)
    merged = merged.drop_duplicates(subset=["timestamp", "text"])
    merged = merged.sort_values("timestamp")

    save_local_posts(merged)
    return merged


# ============================================================
# MARKET DATA + ANALYSIS
# ============================================================
@st.cache_data
def fetch_market_data(tickers, start, end) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers,
        start=start,
        end=end + timedelta(days=1),
        interval="5m",
    )

    if data.empty:
        return data

    # If MultiIndex (Open/High/Low/Close), select Close
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]

    # Ensure index is timezone-aware UTC
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")

    return data


def sentiment_score(text: str) -> float:
    return TextBlob(str(text)).sentiment.polarity


def compute_reactions(
    posts_df: pd.DataFrame, market_df: pd.DataFrame, lookahead_minutes: int
) -> pd.DataFrame:
    if posts_df.empty or market_df.empty:
        return pd.DataFrame(columns=["timestamp", "text", "sentiment"])

    rows = []
    delta = timedelta(minutes=lookahead_minutes)

    for _, row in posts_df.iterrows():
        ts = row["timestamp"]
        text = row["text"]
        end_ts = ts + delta

        # Market window
        window = market_df.loc[(market_df.index >= ts) & (market_df.index <= end_ts)]
        if window.empty:
            continue

        # Starting price
        start_prices = market_df.loc[market_df.index >= ts].head(1)
        if start_prices.empty:
            continue

        start_prices = start_prices.iloc[0]
        end_prices = window.iloc[-1]

        changes = (end_prices - start_prices) / start_prices * 100.0

        row_dict = {
            "timestamp": ts,
            "text": text,
            "sentiment": sentiment_score(text),
        }
        for col in changes.index:
            row_dict[f"{col}_pct_change"] = changes[col]

        rows.append(row_dict)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "text", "sentiment"])

    return pd.DataFrame(rows)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Settings")

# Define tickers
ticker_list = [t.strip().upper() for t in st.session_state.get("tickers", "").split(",") if t.strip()]
#ticker_list = ["MSTY", "MAIN"]
tickers = st.multiselect("Select Tickers to Compare",options=ticker_list,default=ticker_list[:7])

lookahead_minutes = st.sidebar.selectbox(
    "Window after post to measure reaction (minutes)",
    [5, 15, 30, 60, 240, 1440],
    index=3,
)

start_date = st.sidebar.date_input("Start date", value=datetime(2024, 1, 1))
end_date = st.sidebar.date_input("End date", value=datetime.now().date())

st.sidebar.markdown("---")
st.sidebar.caption("This app performs descriptive analysis only. No predictions or trading signals.")

# ============================================================
# MAIN APP
# ============================================================
col_main, col_side = st.columns([3, 1])

with col_main:
    # 1. Load / update posts
    st.subheader("1. Loading Posts")

    with st.spinner("Checking for new posts…"):
        posts_df = update_posts_cache()

    st.success(f"Loaded {len(posts_df)} total posts.")
    if not posts_df.empty:
        st.dataframe(posts_df.tail(10), use_container_width=True)
    else:
        st.info("No posts available yet.")
        st.stop()

    # 2. Fetch market data
    st.subheader("2. Fetching Market Data")

    with st.spinner("Downloading market data…"):
        market_df = fetch_market_data(tickers, start_date, end_date)

    if market_df.empty:
        st.warning("No market data returned. Check tickers and date range.")
        st.stop()

    st.dataframe(market_df.head(10), use_container_width=True)

    # 3. Compute reactions
    st.subheader("3. Aligning Posts with Market Reactions")

    with st.spinner("Computing reactions…"):
        reactions_df = compute_reactions(posts_df, market_df, lookahead_minutes)

    st.write("Reactions columns:", reactions_df.columns.tolist())
    st.write("Reactions shape:", reactions_df.shape)

    if reactions_df.empty:
        st.warning("No reactions computed. Check date ranges, tickers, and lookahead window.")
    else:
        st.dataframe(reactions_df.head(20), use_container_width=True)

    # 4. Sentiment vs Market Reaction
    st.subheader("4. Sentiment vs Market Reaction")

    numeric_cols = [c for c in reactions_df.columns if c.endswith("_pct_change")]

    if reactions_df.empty or "sentiment" not in reactions_df.columns or not numeric_cols:
        st.info("Not enough data to compute sentiment vs reaction.")
    else:
        metric = st.selectbox("Choose metric", numeric_cols)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=reactions_df, x="sentiment", y=metric, ax=ax)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_xlabel("Sentiment (TextBlob polarity)")
        ax.set_ylabel("Percent change (%)")
        ax.set_title(f"Sentiment vs {metric}")
        st.pyplot(fig)

        corr = reactions_df["sentiment"].corr(reactions_df[metric])
        st.markdown(f"**Correlation:** `{corr:.3f}` (descriptive only)")

    # 5. Time series view
    st.subheader("5. Time Series View")

    if market_df.empty:
        st.info("No market data to plot.")
    else:
        ts_ticker = st.selectbox("Ticker", tickers)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        market_df[ts_ticker].plot(ax=ax2, color="tab:blue", alpha=0.8)
        ax2.set_title(f"{ts_ticker} Close Price with Post Markers")
        ax2.set_ylabel("Price")

        if not reactions_df.empty and "timestamp" in reactions_df.columns:
            for ts in reactions_df["timestamp"]:
                ax2.axvline(ts, color="red", alpha=0.3, linestyle="--")

        st.pyplot(fig2)

with col_side:
    st.subheader("Summary")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"**Total posts loaded:** {len(posts_df)}")
    st.markdown(f"**Tickers analyzed:** {', '.join(tickers) if tickers else 'None'}")
    st.markdown(f"**Lookahead window:** {lookahead_minutes} minutes")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Notes**")
    st.markdown("- Descriptive analysis only.")
    st.markdown("- Correlation ≠ causation.")
    st.markdown("- Market moves are driven by many factors.")
    st.markdown("</div>", unsafe_allow_html=True)
