import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Top 20 Gainers", layout="wide")
st.title("📈 Top 20 Stock Gainers Right Now")

# Refresh button
if st.button("🔄 Refresh"):
    st.experimental_rerun()

# Yahoo Finance Screener API (public)
url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?count=100&scrIds=day_gainers"

response = requests.get(url).json()

results = response["finance"]["result"][0]["quotes"]

# Build DataFrame
df = pd.DataFrame(results)

# Select columns
df = df[["symbol", "regularMarketPrice", "regularMarketChangePercent", "regularMarketVolume"]]

# Rename columns
df.columns = ["Ticker", "Price", "% Change", "Volume"]

# Take top 20
df = df.head(20)

st.dataframe(df, use_container_width=True)
