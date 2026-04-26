import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os

DATA_FILE = "data/watchlist.json"

def load_watchlist():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def save_watchlist(watchlist):
    os.makedirs("data", exist_ok=True)
    with open(DATA_FILE, 'w') as f:
        json.dump(watchlist, f)

def validate_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return 'regularMarketPrice' in info and info['regularMarketPrice'] is not None
    except:
        return False

def get_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current = info.get('regularMarketPrice')
        if not current:
            return None
        opening = info.get('open')
        peg = info.get('pegRatio')
        ps = info.get('priceToSalesTrailing12Months')
        pb = info.get('priceToBook')
        short_pct = info.get('shortPercentOfFloat')
        # historical
        data = yf.download(ticker, period="1mo", interval="1d", auto_adjust=True, progress=False)
        if data.empty:
            return None
        close = data['Close']
        # 1 day gain
        yesterday = close.iloc[-2] if len(close) > 1 else current
        day_gain = (current - yesterday) / yesterday * 100 if yesterday else 0
        # 1 week gain
        if len(close) >= 7:
            week_price = close.iloc[-8] if len(close) > 7 else close.iloc[0]
            week_gain = (current / week_price - 1) * 100
        else:
            week_gain = None
        # 1 month gain
        if len(close) >= 30:
            month_price = close.iloc[-31] if len(close) > 30 else close.iloc[0]
            month_gain = (current / month_price - 1) * 100
        else:
            month_gain = None
        return {
            'Ticker': ticker.upper(),
            'Current Price': round(current, 2),
            'Opening Price': round(opening, 2) if opening else None,
            'PEG Ratio': round(peg, 2) if peg else None,
            'PS Ratio': round(ps, 2) if ps else None,
            'PB Ratio': round(pb, 2) if pb else None,
            'Short %': round(short_pct * 100, 2) if short_pct else None,
            '1 Day % Gain': round(day_gain, 2),
            '1 Week % Gain': round(week_gain, 2) if week_gain else None,
            '1 Month % Gain': round(month_gain, 2) if month_gain else None
        }
    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {e}")
        return None

st.title("Stock Watchlist")

watchlist = load_watchlist()

# Display table
if watchlist:
    st.subheader("Your Watchlist")
    metrics = []
    for ticker in watchlist:
        m = get_metrics(ticker)
        if m:
            metrics.append(m)
    if metrics:
        df = pd.DataFrame(metrics)
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No data available for the tickers.")
else:
    st.write("Your watchlist is empty. Add some tickers below.")

# Add ticker
st.subheader("Add Ticker")
ticker_input = st.text_input("Enter stock ticker (e.g., AAPL)", key="add_ticker")
if st.button("Add Ticker"):
    if ticker_input:
        ticker = ticker_input.upper().strip()
        if ticker in watchlist:
            st.warning("Ticker already in watchlist.")
        elif validate_ticker(ticker):
            watchlist.append(ticker)
            save_watchlist(watchlist)
            st.success(f"Added {ticker} to watchlist.")
            st.rerun()
        else:
            st.error("Invalid ticker. Please check the symbol.")
    else:
        st.warning("Please enter a ticker.")

# Delete tickers
if watchlist:
    st.subheader("Delete Tickers")
    selected_to_delete = []
    for ticker in watchlist:
        if st.checkbox(f"Select {ticker} to delete", key=f"delete_{ticker}"):
            selected_to_delete.append(ticker)
    if st.button("Delete Selected"):
        if selected_to_delete:
            watchlist = [t for t in watchlist if t not in selected_to_delete]
            save_watchlist(watchlist)
            st.success(f"Deleted {', '.join(selected_to_delete)} from watchlist.")
            st.rerun()
        else:
            st.warning("No tickers selected for deletion.")