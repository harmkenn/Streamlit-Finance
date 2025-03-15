import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Dividend Information App")

st.header("Input Stock Ticker and Date Range")
col1,col2,col3 = st.columns(3)
with col1:
    ticker = st.text_input("Enter stock ticker:", value="AAPL")
with col2:
    start_date = st.date_input("Enter start date:", value=pd.to_datetime("2024-01-01"))
with col3:
    end_date = st.date_input("Enter end date:", value=pd.to_datetime("today"))

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