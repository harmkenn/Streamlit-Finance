import streamlit as st
from yahooquery import Ticker
import pandas as pd
from datetime import datetime, timedelta

st.title("Dividend Information App")

col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=2 * 365))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())
with col3:
    tickers_list = [t.strip().upper() for t in st.session_state.get("tickers", "").split(",") if t.strip()]
    ticker = st.selectbox("Select Stock Ticker", tickers_list) if tickers_list else st.text_input("Enter ticker").upper()

if ticker:
    try:
        tk = Ticker(ticker)
        
        # Fetch historical prices and dividends (yahooquery returns combined history with 'dividends' column)
        hist = tk.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), actions=True)
        
        if hist.empty:
            st.error(f"No data found for {ticker} in the selected date range.")
        else:
            # If multiple tickers, extract relevant ticker
            if isinstance(hist.index, pd.MultiIndex):
                data = hist.xs(ticker, level=0)
            else:
                data = hist
            
            data = data.copy()
            # 'dividends' column may not exist if no dividends, so ensure it does
            if 'dividends' not in data.columns:
                data['dividends'] = 0.0

            # Filter rows with dividends
            div_data = data[data['dividends'] != 0][['close', 'dividends']].copy()
            if div_data.empty:
                st.info("No dividend payouts found in the selected date range.")
            else:
                div_data['Dividend Yield (%)'] = (div_data['dividends'] / div_data['close']) * 100
                div_data = div_data.rename(columns={'close': 'Close Price', 'dividends': 'Dividend'})
                div_data = div_data.iloc[::-1]  # reverse to show recent first
                
                st.write("Daily Investment Values, Dividend Payouts, and Stock Prices:")
                st.dataframe(div_data)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
else:
    st.info("Please enter or select a stock ticker symbol.")
