import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Top 20 Gainers", layout="wide")
st.title("📈 Top 20 Stock Gainers Right Now")

# Refresh button
if st.button("🔄 Refresh"):
    st.experimental_rerun()

# Pull top gainers from Yahoo Finance
gainers = yf.get_day_gainers()

# Select columns you want
df = gainers[["symbol", "price", "change_percent", "volume"]]

# Rename columns
df.columns = ["Ticker", "Price", "% Change", "Volume"]

# Take top 20
df = df.head(20)

st.dataframe(df, use_container_width=True)
