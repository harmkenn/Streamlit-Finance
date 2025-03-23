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
            after_hours = ("16:00", "20:00")

            # Assign session labels
            data["Session"] = "Regular Hours"
            data.loc[data["Time"].between(*early_premarket), "Session"] = "Early Pre-market"
            data.loc[data["Time"].between(*regular_premarket), "Session"] = "Regular Pre-market"
            data.loc[data["Time"].between(*after_hours), "Session"] = "After-hours"

            # Define session colors
            session_colors = {
                "Early Pre-market": "purple",
                "Regular Pre-market": "green",
                "Regular Hours": "blue",
                "After-hours": "red"
            }

            # Create subplots for price and volume
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Plot each session with a distinct color (without gaps)
            for session, color in session_colors.items():
                session_data = data[data["Session"] == session]
                fig.add_trace(go.Scatter(
                    x=session_data.index,
                    y=session_data["Close"],
                    mode="lines",
                    name=session,
                    line=dict(color=color),
                    connectgaps=False  # Prevents lines from connecting different sessions
                ), secondary_y=False)

            # Volume as grey bars
            fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume", marker=dict(color="grey")), secondary_y=True)

            # Update layout
            fig.update_layout(
                title=f"{stock_symbol} Intraday Prices (Including Pre-market & After-hours)",
                xaxis_title="Time",
                yaxis_title="Price",
                legend_title="Market Session"
            )
            fig.update_yaxes(title_text="Volume", secondary_y=True)

            # Display chart
            st.plotly_chart(fig)

            # Show raw data with session labels
            st.write(data[["Close", "Volume", "Session"]])

    except Exception as e:
        st.error(f"Error fetching data: {e}")
