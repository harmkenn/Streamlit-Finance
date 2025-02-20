import streamlit as st
import yfinance as yf
import pandas as pd

# Function to fetch stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.option_chain('all')

# Streamlit app
st.title("Option Sell Dates for Next 12 Months")
ticker = st.text_input("Enter Stock Ticker")
if st.button("Fetch Options"):
    stock_data = get_stock_data(ticker)
    # Process and display the data
    # (You'll need to add logic to filter and display the sell dates for the next 12 months)
    st.write(stock_data)
