import os

import pandas as pd
import streamlit as st
import yfinance as yf

TICKERS_FILE = os.path.join(os.path.dirname(__file__), "..", "tickers.txt")


def load_tickers() -> list[str]:
    if os.path.exists(TICKERS_FILE):
        with open(TICKERS_FILE, "r", encoding="utf-8") as f:
            return [t.strip().upper() for t in f.read().split(",") if t.strip()]
    return ["TQQQ", "UPRO", "UDOW", "^VIX"]


st.set_page_config(page_title="Watchlist", layout="wide")
st.title("Watchlist")

selected_tickers = load_tickers()
if not selected_tickers:
    st.info("No tickers found in tickers.txt.")
    st.stop()

rows = []
for ticker in selected_tickers:
    try:
        data = yf.Ticker(ticker)
        info = data.fast_info or {}
        price = info.get("last_price") or info.get("lastPrice")
        change = info.get("last_change") or info.get("lastChange")
        percent_change = info.get("last_change_pct") or info.get("lastChangePercent")
        market_cap = info.get("market_cap") or info.get("marketCap")

        if price is None:
            hist = data.history(period="5d")
            if not hist.empty:
                price = hist["Close"].iloc[-1]
                change = price - hist["Close"].iloc[-2] if len(hist) > 1 else 0
                percent_change = (change / hist["Close"].iloc[-2]) * 100 if len(hist) > 1 and hist["Close"].iloc[-2] else 0

        rows.append({
            "Ticker": ticker,
            "Price": round(float(price), 2) if price is not None else None,
            "Change": round(float(change), 2) if change is not None else None,
            "% Change": round(float(percent_change), 2) if percent_change is not None else None,
            "Market Cap": market_cap,
        })
    except Exception:
        rows.append({
            "Ticker": ticker,
            "Price": None,
            "Change": None,
            "% Change": None,
            "Market Cap": None,
        })

if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No ticker data available.")
