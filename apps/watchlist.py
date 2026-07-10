import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Market Movers", layout="wide")
st.title("📈 Market Movers")

if st.button("🔄 Refresh"):
    st.experimental_rerun()

def fetch_screener(scr_id):
    url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?count=50&scrIds={scr_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    try:
        data = response.json()
    except ValueError:
        return None
    
    return data["finance"]["result"][0]["quotes"]

# ---- Top Gainers ----
gainers = fetch_screener("day_gainers")
if gainers:
    df_gainers = pd.DataFrame(gainers)
    df_gainers = df_gainers[["symbol", "regularMarketPrice", "regularMarketChangePercent", "regularMarketVolume"]]
    df_gainers.columns = ["Ticker", "Price", "% Change", "Volume"]
    df_gainers = df_gainers.head(20)

    st.subheader("🔥 Top 20 Gainers")
    st.dataframe(df_gainers, use_container_width=True)
else:
    st.error("⚠️ Unable to load Top Gainers. Try again later.")

# ---- Premarket Movers ----
premarket = fetch_screener("day_gainers_premarket")
if premarket:
    df_pre = pd.DataFrame(premarket)
    df_pre = df_pre[["symbol", "preMarketPrice", "preMarketChangePercent", "preMarketVolume"]]
    df_pre.columns = ["Ticker", "Premarket Price", "% Change", "Volume"]

    st.subheader("🌅 Premarket Movers")
    st.dataframe(df_pre, use_container_width=True)
else:
