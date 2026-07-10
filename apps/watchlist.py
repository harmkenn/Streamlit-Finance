import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Top 20 Gainers", layout="wide")
st.title("📈 Top 20 Stock Gainers Right Now")

if st.button("🔄 Refresh"):
    st.experimental_rerun()

url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?count=100&scrIds=day_gainers"

headers = { "User-Agent": "Mozilla/5.0" }
response = requests.get(url, headers=headers)

try:
    data = response.json()
except ValueError:
    st.error("⚠️ Yahoo Finance returned invalid data. Please refresh again.")
    st.stop()

results = data["finance"]["result"][0]["quotes"]
df = pd.DataFrame(results)

df = df[["symbol", "regularMarketPrice", "regularMarketChangePercent", "regularMarketVolume"]]
df.columns = ["Ticker", "Price", "% Change", "Volume"]
df = df.head(20)

st.dataframe(df, use_container_width=True)
