import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os

st.set_page_config(layout="wide")

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
        price = safe_get(info, 'regularMarketPrice')
        return price is not None
    except:
        return False

def safe_get(info, key):
    try:
        val = info[key]
    except KeyError:
        return None
    if isinstance(val, (pd.DataFrame, pd.Series)):
        if hasattr(val, 'empty') and val.empty:
            return None
        if hasattr(val, 'item'):
            try:
                return val.item()
            except ValueError:
                pass
        if hasattr(val, 'iloc'):
            try:
                return val.iloc[0] if isinstance(val, pd.Series) else val.iloc[0, 0]
            except:
                pass
        return None
    return val

def get_metrics(ticker):
    # Mock data for demo
    mock_data = {
        'AAPL': {
            'Ticker': 'AAPL',
            'Current Price': 150.0,
            'Opening Price': 148.0,
            'PEG Ratio': 1.5,
            'PS Ratio': 5.0,
            'PB Ratio': 10.0,
            'Short %': 0.5,
            '1 Day % Gain': 1.2,
            '1 Week % Gain': 3.5,
            '1 Month % Gain': 7.8
        },
        'MXL': {
            'Ticker': 'MXL',
            'Current Price': 25.0,
            'Opening Price': 24.5,
            'PEG Ratio': 2.0,
            'PS Ratio': 3.0,
            'PB Ratio': 2.5,
            'Short %': 1.0,
            '1 Day % Gain': -0.5,
            '1 Week % Gain': 2.0,
            '1 Month % Gain': 5.0
        },
        'TQQQ': {
            'Ticker': 'TQQQ',
            'Current Price': 45.0,
            'Opening Price': 44.0,
            'PEG Ratio': None,
            'PS Ratio': None,
            'PB Ratio': None,
            'Short %': 2.0,
            '1 Day % Gain': 4.0,
            '1 Week % Gain': 10.0,
            '1 Month % Gain': 15.0
        },
        'NVTS': {
            'Ticker': 'NVTS',
            'Current Price': 5.0,
            'Opening Price': 4.8,
            'PEG Ratio': None,
            'PS Ratio': 20.0,
            'PB Ratio': 5.0,
            'Short %': 3.0,
            '1 Day % Gain': 2.0,
            '1 Week % Gain': 8.0,
            '1 Month % Gain': 12.0
        }
    }
    if ticker.upper() in mock_data:
        return mock_data[ticker.upper()]
    try:
        stock = yf.Ticker(ticker)
        try:
            info = stock.info
        except:
            return None
        current = safe_get(info, 'regularMarketPrice')
        opening = safe_get(info, 'open')
        peg = safe_get(info, 'pegRatio')
        ps = safe_get(info, 'priceToSalesTrailing12Months')
        pb = safe_get(info, 'priceToBook')
        short_pct = safe_get(info, 'shortPercentOfFloat')
        # historical data for % gains
        try:
            data = yf.download(ticker, period="1mo", interval="1d", auto_adjust=True, progress=False)
            if not isinstance(data, pd.DataFrame) or data.empty:
                day_gain = None
                week_gain = None
                month_gain = None
            else:
                close = data['Close']
                yesterday = close.iloc[-2] if len(close) > 1 else current
                day_gain = (current - yesterday) / yesterday * 100 if yesterday and yesterday != 0 else None
                if len(close) >= 7:
                    week_price = close.iloc[-8] if len(close) > 7 else close.iloc[0]
                    week_gain = (current / week_price - 1) * 100 if week_price and week_price != 0 else None
                else:
                    week_gain = None
                if len(close) >= 30:
                    month_price = close.iloc[-31] if len(close) > 30 else close.iloc[0]
                    month_gain = (current / month_price - 1) * 100 if month_price and month_price != 0 else None
                else:
                    month_gain = None
        except:
            day_gain = None
            week_gain = None
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