import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

st.title("Intraday Stock Prices with Extended Hours")

# User input for stock symbol
stock_symbol = st.text_input("Enter stock symbol (e.g. AAPL, GOOG, MSFT):", "NVDA").upper()

if stock_symbol:
    try:
        # Fetch stock data (5-minute interval for 5 days to capture extended hours)
        ticker = yf.Ticker(stock_symbol)
        data = ticker.history(period="5d", interval="5m", prepost=True)  # Include pre/after-market

        if data.empty:
            st.error(f"No data found for {stock_symbol}. Please check the symbol and try again.")
        else:
            # Convert timestamps to Eastern Time
            data = data.tz_convert("America/New_York")

            # Extract time for session classification
            data["Time"] = data.index.strftime("%H:%M")

            # Define session time ranges
            early_premarket = ("04:00", "07:00")
            regular_premarket = ("07:00", "09:30")
            regular_hours = ("09:30", "16:00")
            after_hours_
