import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import time

st.title("Stock Earnings and Options Calendar")

st.header("Enter a stock ticker:")

ticker = st.text_input("Ticker", value="AAPL")

if ticker:
    time.sleep(1)  # wait for 1 second before making the request
    data = yf.Ticker(ticker)

    # Get earnings dates for the next 12 months
    earnings_dates = []
    for date in data.calendar:
        if date > datetime.today() and date < datetime.today() + timedelta(days=365):
            earnings_dates.append(date)

    # Get option expiration dates for the next 12 months
    options_dates = []
    for date in data.options:
        if date > datetime.today() and date < datetime.today() + timedelta(days=365):
            options_dates.append(date)

    st.header("Earnings Dates:")
    for date in earnings_dates:
        st.write(date.strftime("%Y-%m-%d"))

    st.header("Option Expiration Dates:")
    for date in options_dates:
        st.write(date.strftime("%Y-%m-%d"))