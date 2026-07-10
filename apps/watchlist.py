import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Top Gainers Watchlist", layout="wide")

st.title("📈 Biggest Stock Gainers Right Now")

# Universe of tickers to scan (you can expand this list)
# For example: S&P 500, NASDAQ 100, or your own list
tickers = [
    "AAPL","MSFT","TSLA","AMZN","NVDA","META","GOOGL","AMD","NFLX","PLTR",
    "TQQQ","UPRO","UDOW","XOP","^VIX"
]

# Refresh button
if st.button("🔄 Refresh Data"):
    st.experimental_rerun()

def get_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "Ticker": ticker,
            "Price": info.get("regularMarketPrice"),
            "Change %": info.get("regularMarketChangePercent"),
            "Volume": info.get("regularMarketVolume"),
        }
    except Exception:
        return None

# Fetch data
rows = []
for t in tickers:
    data = get_data(t)
    if data:
        rows.append(data)

df = pd.DataFrame(rows)

# Sort by biggest gainers
df = df.sort_values("Change %", ascending=False)

st.subheader("🔥 Top Gainers")
st.dataframe(df, use_container_width=True)
