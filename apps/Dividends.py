import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.title("Dividend Information App")

st.header("Input Stock Ticker and Date Range")
col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("Start Date",
                               datetime.today() - timedelta(days=2 * 365))
with col2:
    end_date = st.date_input("End Date", datetime.today())
with col3:
    # Ticker input
    ticker = st.text_input("Enter Stock Ticker", "MSTY").upper()

stock = yf.Ticker(ticker)
historical_data = stock.history(start=start_date, end=end_date, actions=True)
dividends = historical_data['Dividends']
historical_prices = historical_data['Close']
div_data = historical_data[['Close', 'Dividends']]
div_data['Dividend Yield'] = (div_data['Dividends'] / div_data['Close']) * 100
div_data = div_data[div_data['Dividends'] != 0]

st.write("Daily Investment Values, Dividend Payouts, and Stock Prices:")
st.dataframe(div_data)