import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.title("Dividend Information App")

col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=2 * 365))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())
with col3:
    ticker = st.text_input("Enter Stock Ticker", value="MSTY").upper()

stock = yf.Ticker(ticker)
historical_data = stock.history(start=start_date, end=end_date, actions=True)

# Extract relevant data
dividends = historical_data['Dividends']
historical_prices = historical_data['Close']

# Create dividend data frame
div_data = historical_data[['Close', 'Dividends']].copy()
div_data['Dividend Yield'] = (div_data['Dividends'] / div_data['Close']) * 100
div_data = div_data[div_data['Dividends'] != 0]

st.write("Daily Investment Values, Dividend Payouts, and Stock Prices:")
st.dataframe(div_data)