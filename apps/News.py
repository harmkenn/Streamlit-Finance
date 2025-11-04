import streamlit as st
import yfinance as yf
import requests

st.set_page_config(page_title="Stock News", layout="wide")
st.title("📰 Stock News Dashboard")

# --- Input for tickers ---
st.markdown("Enter a comma-separated list of stock symbols (e.g. AAPL, MSFT, NVDA):")
tickers_input = st.text_input("Tickers", value=st.session_state.get("tickers", "AAPL, MSFT, NVDA"))
st.session_state["tickers"] = tickers_input
tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

refresh_button = st.button("🔄 Fetch News")

# --- Function to simulate news fetch (replace with real API) ---
def get_stock_news(ticker):
    # Placeholder: Replace with real API call (e.g., NewsAPI, Bing News Search, etc.)
    return [
        {"title": f"{ticker} hits new highs amid market rally", "url": f"https://news.example.com/{ticker}/1"},
        {"title": f"Analysts upgrade {ticker} after earnings beat", "url": f"https://news.example.com/{ticker}/2"},
        {"title": f"{ticker} faces regulatory scrutiny over new product", "url": f"https://news.example.com/{ticker}/3"},
    ]

# --- Display News ---
if tickers_list and refresh_button:
    cols = st.columns(len(tickers_list))
    for i, t in enumerate(tickers_list):
        with cols[i]:
            try:
                yf_t = yf.Ticker(t)
                info = yf_t.info
                st.subheader(f"🧾 {t} News")
                news_items = get_stock_news(t)
                for item in news_items:
                    st.markdown(f"- [{item['title']}]({item['url']})")
            except Exception as e:
                st.error(f"Error retrieving news for {t}: {e}")
else:
    st.info("👆 Enter tickers and click 'Fetch News' to view headlines.")
