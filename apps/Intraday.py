import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
from datetime import datetime

st.title("Intraday Stock Prices (Including Pre-market & After-hours)")

# User input for stock symbol
col1, col2 = st.columns(2)
with col1:
    stock_symbol = st.text_input("Enter stock symbol (e.g. AAPL, GOOG, MSFT):", "NVDY").upper()
    
if stock_symbol:
    try:
        # Fetch stock data (5-minute interval for 5 days to capture extended hours)
        ticker = yf.Ticker(stock_symbol)
        data = ticker.history(period="5d", interval="5m", prepost=True)  # Include pre/after-market

        if data.empty:
            st.error(f"No data found for {stock_symbol}. Please check the symbol and try again.")
        else:
            with col2:
                # Get the latest closing price (most recent data point)
                latest_price = data["Close"].iloc[-1]
                
                # Display the current price at the top of the page
                st.markdown(f"### Current Price: ${latest_price:.2f}")
            # Convert timestamps to Eastern Time
            data = data.tz_convert("America/New_York")

            # Create subplots for price and volume
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Plot single-colored line for price
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                name="Stock Price",
                line=dict(color="blue")  # Set single color
            ), secondary_y=False)

            # Volume as grey bars
            fig.add_trace(go.Bar(
                x=data.index,
                y=data["Volume"],
                name="Volume",
                marker=dict(color="grey")
            ), secondary_y=True)

            # Update layout
            fig.update_layout(
                title=f"{stock_symbol} Intraday Prices (Including Pre-market & After-hours)",
                xaxis_title="Time",
                yaxis_title="Price",
                legend_title="Market Data"
            )
            fig.update_yaxes(title_text="Volume", secondary_y=True)

            # Display chart
            st.plotly_chart(fig)

            # Show raw data
            st.write(data[["Close", "Volume"]])

    except Exception as e:
        st.error(f"Error fetching data: {e}")
