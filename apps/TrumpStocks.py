import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import timedelta


# ---------------------------------------------------------
# Helpers v1.3 — includes timestamp cleaning
# ---------------------------------------------------------

def detect_timestamp_column(df):
    """
    Auto-detect a timestamp column in the uploaded CSV.
    Looks for columns containing 'date', 'time', or 'stamp'.
    """
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "stamp"])]
    if not candidates:
        raise ValueError("No timestamp-like column found. Expected something like 'timestamp' or 'date'.")

    # Prefer common names
    for preferred in ["timestamp", "date", "datetime", "created_at"]:
        for c in candidates:
            if c.lower() == preferred:
                return c

    return candidates[0]


def clean_timestamp(ts):
    """
    Convert strings like:
        'April 8, 2026 @ 10:56 PM ET'
    into a pandas-parsable datetime.
    """
    if pd.isna(ts):
        return pd.NaT

    ts = str(ts)

    # Remove " @ " and "ET"
    ts = ts.replace(" @ ", " ")
    ts = ts.replace("ET", "")
    ts = ts.strip()

    return pd.to_datetime(ts, errors="coerce")


def fetch_daily_prices(tickers, start_date, end_date):
    """
    Fetch daily OHLCV for a list of tickers using yfinance.
    Returns a DataFrame with columns like 'TQQQ', 'UPRO', etc.
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    # yfinance returns a multi-index (OHLCV). We only want Close.
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]

    data.index = pd.to_datetime(data.index)
    return data


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.title("Trump Tweets + Daily Market Prices (yfinance)")

st.write(
    "Upload a CSV of Trump tweets, and I'll match them with **daily prices** "
    "for **TQQQ, UPRO, UDOW, XOP, and ^VIX** using yfinance."
)

uploaded = st.file_uploader("Upload Trump Tweet CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("Preview of Uploaded Tweets")
    st.dataframe(df.head())

    # Detect timestamp column
    try:
        ts_col = detect_timestamp_column(df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.success(f"Detected timestamp column: **{ts_col}**")

    # Clean + convert timestamps
    df[ts_col] = df[ts_col].apply(clean_timestamp)
    df = df.dropna(subset=[ts_col])

    if df.empty:
        st.error("All timestamps failed to parse. Check the timestamp format.")
        st.stop()

    min_date = df[ts_col].min().date()
    max_date = df[ts_col].max().date()

    # Pad the range slightly
    start_date = min_date - timedelta(days=2)
    end_date = max_date + timedelta(days=2)

    st.write(f"**Tweet date range:** {min_date} → {max_date}")
    st.write(f"**Fetching market data for:** {start_date} → {end_date}")

    if st.button("Fetch Daily Prices"):
        with st.spinner("Fetching daily prices from yfinance…"):
            tickers = ["TQQQ", "UPRO", "UDOW", "XOP", "^VIX"]
            prices = fetch_daily_prices(tickers, start_date, end_date)

        st.subheader("Daily Price Data (Close)")
        st.dataframe(prices.tail())

        # Merge tweets + prices by date
        df["date"] = df[ts_col].dt.date
        prices["date"] = prices.index.date

        merged = df.merge(prices, on="date", how="left")

        st.subheader("Merged Tweet + Price Dataset")
        st.dataframe(merged.head())

        # Download merged CSV
        csv = merged.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Merged CSV",
            csv,
            "tweets_with_daily_prices.csv",
            "text/csv"
        )
