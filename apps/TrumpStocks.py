import streamlit as st
import pandas as pd
from datetime import timedelta
from dateutil import parser

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import yfinance as yf

#1.1
def parse_timestamp_column(df):
    """
    Try to find and parse a timestamp column in the uploaded Trump CSV.
    Assumes a column name like 'timestamp', 'date', 'time', etc.
    """
    candidate_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if not candidate_cols:
        raise ValueError("Could not find a timestamp/date column in the uploaded file.")

    # Prefer 'timestamp' if present
    for name in ["timestamp", "date", "datetime"]:
        for c in candidate_cols:
            if c.lower() == name:
                ts_col = c
                break
        else:
            continue
        break
    else:
        ts_col = candidate_cols[0]

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if df[ts_col].isna().all():
        raise ValueError(f"Could not parse any valid datetimes from column '{ts_col}'.")

    return df, ts_col


def get_alpaca_daily_prices(api_key, api_secret, symbols, start_date, end_date):
    client = StockHistoricalDataClient(api_key, api_secret)

    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
    )

    bars = client.get_stock_bars(req)
    df = bars.df  # MultiIndex: (symbol, timestamp)

    # Flatten to wide format: one column per symbol, index = date
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.date
    price_pivot = df.pivot_table(
        index="timestamp",
        columns="symbol",
        values="close",
        aggfunc="last",
    ).sort_index()

    price_pivot.index = pd.to_datetime(price_pivot.index)
    return price_pivot


def get_vix_daily(start_date, end_date):
    vix = yf.download("^VIX", start=start_date, end=end_date, interval="1d")
    if vix.empty:
        return pd.DataFrame()
    vix = vix[["Adj Close"]].rename(columns={"Adj Close": "^VIX"})
    vix.index = vix.index.tz_localize(None)
    return vix


def main():
    st.title("Trump Tweets + Market Data (Daily)")

    st.markdown(
        "Upload your Trump tweet CSV, and I’ll pull **daily prices** for "
        "`TQQQ, UPRO, UDOW, XOP` from **Alpaca**, plus `^VIX` from **yfinance**."
    )

    st.sidebar.header("Alpaca Credentials")
    api_key = st.sidebar.text_input("Alpaca API Key", type="password")
    api_secret = st.sidebar.text_input("Alpaca API Secret", type="password")

    uploaded_file = st.file_uploader("Upload Trump tweets CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            tweets_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

        st.write("Preview of uploaded tweets:")
        st.dataframe(tweets_df.head())

        try:
            tweets_df, ts_col = parse_timestamp_column(tweets_df)
        except ValueError as e:
            st.error(str(e))
            return

        min_ts = tweets_df[ts_col].min()
        max_ts = tweets_df[ts_col].max()

        if pd.isna(min_ts) or pd.isna(max_ts):
            st.error("Could not determine a valid date range from the tweet timestamps.")
            return

        # Pad the range a bit
        start_date = (min_ts - timedelta(days=2)).date()
        end_date = (max_ts + timedelta(days=2)).date()

        st.markdown(
            f"**Detected tweet date range:** {min_ts} → {max_ts}  \n"
            f"**Fetching market data for:** {start_date} → {end_date}"
        )

        if not api_key or not api_secret:
            st.warning("Enter your Alpaca API key and secret in the sidebar to fetch prices.")
            return

        if st.button("Fetch Daily Prices"):
            with st.spinner("Fetching daily prices from Alpaca and yfinance..."):
                try:
                    symbols = ["TQQQ", "UPRO", "UDOW", "XOP"]
                    alpaca_prices = get_alpaca_daily_prices(
                        api_key, api_secret, symbols, start_date, end_date
                    )
                except Exception as e:
                    st.error(f"Error fetching data from Alpaca: {e}")
                    return

                vix = get_vix_daily(start_date, end_date)

                # Merge Alpaca + VIX
                if not vix.empty:
                    combined = alpaca_prices.join(vix, how="outer")
                else:
                    combined = alpaca_prices

                combined = combined.sort_index()

                st.subheader("Daily Price Data")
                st.dataframe(combined.tail(20))

                csv = combined.to_csv(index=True).encode("utf-8")
                st.download_button(
                    label="Download combined daily prices as CSV",
                    data=csv,
                    file_name="trump_markets_daily.csv",
                    mime="text/csv",
                )


main()
