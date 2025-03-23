import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

st.title("Intraday Stock Prices with Pre-market & After-hours")

# User input for stock symbol
stock_symbol = st.text_input("Enter stock symbol (e.g. AAPL, GOOG, MSFT):", "NVDA").upper()

if stock_symbol:
    try:
        # Fetch stock data (1-minute interval for 1 day to capture extended hours)
        ticker = yf.Ticker(stock_symbol)
        data = ticker.history(period="5d", interval="5m", prepost=True)  # Include pre/after-market

        if data.empty:
            st.error(f"No data found for {stock_symbol}. Please check the symbol and try again.")
        else:
            # Determine pre-market and after-hours sessions
            market_open = "09:30"
            market_close = "16:00"

            data["Time"] = data.index.strftime("%H:%M")
            data["Session"] = "Regular Hours"
            data.loc[data["Time"] < market_open, "Session"] = "Pre-market"
            data.loc[data["Time"] > market_close, "Session"] = "After-hours"

            # Create subplots for price and volume
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Plot regular hours in blue, pre-market in green, after-hours in red
            for session, color in [("Regular Hours", "blue"), ("Pre-market", "green"), ("After-hours", "red")]:
                session_data = data[data["Session"] == session]
                fig.add_trace(go.Scatter(x=session_data.index, y=session_data["Close"], mode="lines", name=session, line=dict(color=color)), secondary_y=False)

            # Volume as bars (grey)
            fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume", marker=dict(color="grey")), secondary_y=True)

            # Update layout
            fig.update_layout(title=f"{stock_symbol} Intraday Prices (Including Pre-market & After-hours)", xaxis_title="Time", yaxis_title="Price")
            fig.update_yaxes(title_text="Volume", secondary_y=True)

            # Display chart
            st.plotly_chart(fig)

            # Show raw data
            st.write(data[["Close", "Volume", "Session"]])

    except Exception as e:
        st.error(f"Error fetching data: {e}")
