import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# ============================================================
# CONFIG
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
def load_local_posts():
    if not os.path.exists(CACHE_FILE):
        return pd.DataFrame(columns=["timestamp", "text"])
    df = pd.read_csv(CACHE_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

def save_local_posts(df):
    df.to_csv(CACHE_FILE, index=False)

# ============================================================
# FETCH NEW POSTS (placeholder)
# Replace this with a real Truth Social fetcher later
# ============================================================
def fetch_new_truth_posts(latest_timestamp):
    # TODO: Replace with real scraper/API
    # For now, simulate a new post only if latest_timestamp is old
    now = datetime.utcnow()

    if latest_timestamp is None or (now - latest_timestamp).total_seconds() > 3600:
        return pd.DataFrame([
            {
                "timestamp": now.isoformat(),
                "text": "Example new post for testing incremental updates."
            }
        ])
    return pd.DataFrame(columns=["timestamp", "text"])

# ============================================================
# MERGE OLD + NEW POSTS
# ============================================================
def update_posts_cache():
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
def fetch_market_data(tickers, start, end):
    data = yf.download(
        tickers,
        start=start,
        end=end + timedelta(days=1),
        interval="5m"
    )
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    data = data.tz_localize("UTC", level=None, nonexistent="shift_forward", ambiguous="NaT", errors="ignore")
    return data

def sentiment_score(text):
    return TextBlob(str(text)).sentiment.polarity

def compute_reactions(posts_df, market_df, lookahead_minutes):
    rows = []
    delta = timedelta(minutes=lookahead_minutes)

    for _, row in posts_df.iterrows():
        ts = row["timestamp"]
        text = row["text"]
        end_ts = ts + delta

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

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Settings")

tickers = [t.strip().upper() for t in st.session_state.get("tickers", "").split(",") if t.strip()]

lookahead_minutes = st.sidebar.selectbox(
    "Window after post to measure reaction",
    [5, 15, 30, 60, 240, 1440],
    index=3
)

start_date = st.sidebar.date_input("Start date", value=datetime(2024, 1, 1))
end_date = st.sidebar.date_input("End date", value=datetime.utcnow().date())

st.sidebar.markdown("---")
st.sidebar.caption("This app performs descriptive analysis only.")

# ============================================================
# MAIN APP
# ============================================================
st.subheader("1. Loading Posts")

with st.spinner("Checking for new posts…"):
    posts_df = update_posts_cache()

st.success(f"Loaded {len(posts_df)} total posts.")
st.dataframe(posts_df.tail(10), use_container_width=True)

st.subheader("2. Fetching Market Data")

with st.spinner("Downloading market data…"):
    market_df = fetch_market_data(tickers, start_date, end_date)

st.dataframe(market_df.head(10), use_container_width=True)

st.subheader("3. Aligning Posts with Market Reactions")

with st.spinner("Computing reactions…"):
    reactions_df = compute_reactions(posts_df, market_df, lookahead_minutes)

st.dataframe(reactions_df.head(20), use_container_width=True)

# ============================================================
# SENTIMENT VS MARKET REACTION
# ============================================================
st.subheader("4. Sentiment vs Market Reaction")

numeric_cols = [c for c in reactions_df.columns if c.endswith("_pct_change")]

if numeric_cols:
    metric = st.selectbox("Choose metric", numeric_cols)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=reactions_df, x="sentiment", y=metric, ax=ax)
    ax.axvline(0, color="gray", linestyle="--")
    st.pyplot(fig)

    corr = reactions_df["sentiment"].corr(reactions_df[metric])
    st.markdown(f"**Correlation:** `{corr:.3f}` (descriptive only)")

# ============================================================
# TIME SERIES VIEW
# ============================================================
st.subheader("5. Time Series View")

ts_ticker = st.selectbox("Ticker", tickers)

fig2, ax2 = plt.subplots(figsize=(10, 4))
market_df[ts_ticker].plot(ax=ax2)
for ts in reactions_df["timestamp"]:
    ax2.axvline(ts, color="red", alpha=0.3, linestyle="--")
st.pyplot(fig2)
