import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime

st.title("Intraday Stock Prices (Including Pre-market & After-hours)")

# User input for stock symbol
col1, col2, col3 = st.columns(3)
with col1:
    # Get tickers from session state and split into a list
    tickers_list = [t.strip().upper() for t in st.session_state.get("tickers", "").split(",") if t.strip()]

    # Ticker selector
    ticker = st.selectbox("Select Stock Ticker", tickers_list) if tickers_list else ""

with col3:
    refresh_button = st.button("Refresh")

if ticker:
    try:
        # Fetch stock data (5-minute interval for 5 days to capture extended hours)
        yf_ticker = yf.Ticker(ticker)
        data = yf_ticker.history(period="5d", interval="5m", prepost=True)

        if data.empty:
            st.error(f"No data found for {ticker}. Please check the symbol and try again.")
        else:
            # Convert timestamps to Eastern Time for session filtering
            data = data.tz_convert("America/New_York")

        with col2:
            # Get the latest price
            latest_price = data["Close"].iloc[-1]

            # Filter only regular trading hours
            regular_hours = data.between_time("09:30", "16:00")

            # Group by date and get the last close of each day
            grouped_days = regular_hours.groupby(regular_hours.index.date)

            if len(grouped_days) >= 3:
                # Most recent full close (yesterday)
                prev_close = grouped_days.last().iloc[-2]["Close"]
                # Close before that
                prev_prev_close = grouped_days.last().iloc[-3]["Close"]
            elif len(grouped_days) == 2:
                prev_close = grouped_days.last().iloc[-2]["Close"]
                prev_prev_close = prev_close  # Fallback
            else:
                prev_close = latest_price
                prev_prev_close = prev_close  # Fallback

            # Change from previous close to current
            percent_change = ((latest_price - prev_close) / prev_close) * 100
            change_color = "green" if percent_change >= 0 else "red"

            # Change from prior-to-previous close to previous close
            prev_change = ((prev_close - prev_prev_close) / prev_prev_close) * 100
            prev_color = "green" if prev_change >= 0 else "red"

            # Display previous closing price and its change
            st.markdown(
                f"### Previous Close: ${prev_close:.2f} "
                f"<span style='color:{prev_color}; font-size:18px'>({prev_change:+.2f}%)</span>",
                unsafe_allow_html=True
            )

            # Display current price and percent change from previous close
            st.markdown(
                f"### Current Price: ${latest_price:.2f} "
                f"<span style='color:{change_color}; font-size:20px'>({percent_change:+.2f}%)</span>",
                unsafe_allow_html=True
            )


        # Create subplots for price and volume
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Plot price
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Stock Price",
            line=dict(color="blue")
        ), secondary_y=False)

        # Plot volume
        fig.add_trace(go.Bar(
            x=data.index,
            y=data["Volume"],
            name="Volume",
            marker=dict(color="grey")
        ), secondary_y=True)

        # Update layout
        fig.update_layout(
            title=f"{ticker} Intraday Prices (Including Pre-market & After-hours)",
            xaxis_title="Time",
            yaxis_title="Price",
            legend_title="Market Data"
        )
        fig.update_yaxes(title_text="Volume", secondary_y=True)

        # Display chart
        st.plotly_chart(fig)

        # Show raw data (reversed for most recent on top)
        st.write(data[["Close", "Volume"]][::-1])

        # Refresh button logic
        if refresh_button:
            st.experimental_rerun()

    except Exception as e:
        st.error(f"Error fetching data: {e}")
