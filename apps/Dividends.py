import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.title("Dividend Information App")

st.header("Input Stock Ticker and Date Range")
col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("Start Date",
                               datetime.today() - timedelta(days=5 * 365))
with col2:
    end_date = st.date_input("End Date", datetime.today())
with col3:
    # Ticker input
    ticker = st.text_input("Enter Stock Ticker", "TQQQ").upper()

if st.button("Get Dividend Information"):
    stock_data = yf.Ticker(ticker)
    dividends = stock_data.dividends[start_date:end_date]
    if dividends is not None:
        dividend_yield = stock_data.info["trailingAnnualDividendRate"]
        dividend_df = pd.DataFrame(dividends).reset_index()
        dividend_df.columns = ["Date", "Dividend Amount"]
        dividend_df["Dividend Yield"] = dividend_yield
        st.write(dividend_df)
    else:
        st.write("No dividend data available for the specified date range.")