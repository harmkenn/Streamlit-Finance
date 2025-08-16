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
            daily_closes = regular_hours.groupby(regular_hours.index.date).last()

            if len(daily_closes) >= 4:
                # Get the last 4 closes (most recent 3 for display + 1 to compare from)
                recent_closes = daily_closes.tail(4)
            else:
                recent_closes = daily_closes

            close_dates = recent_closes.index.tolist()
            close_values = recent_closes["Close"].tolist()

            # Display previous 3 closes and changes
            for i in range(1, len(close_values)):
                date_str = close_dates[i].strftime("%Y-%m-%d")
                close_price = close_values[i]
                prev_close_price = close_values[i - 1]
                price_change = close_price - prev_close_price
                percent_change = (price_change / prev_close_price) * 100 if prev_close_price != 0 else 0
                color = "green" if percent_change >= 0 else "red"

                st.markdown(
                    f"### {date_str}: ${close_price:.2f} "
                    f"<span style='color:{color}; font-size:16px'>({price_change:+.2f}, {percent_change:+.2f}%)</span>",
                    unsafe_allow_html=True
                )

            # Display current price and change from most recent close
            last_close_price = close_values[-1]
            price_diff = latest_price - last_close_price
            percent_diff = (price_diff / last_close_price) * 100 if last_close_price != 0 else 0
            color = "green" if percent_diff >= 0 else "red"

            st.markdown(
                f"### Current Price: ${latest_price:.2f} "
                f"<span style='color:{color}; font-size:20px'>({percent_diff:+.2f}%)</span>",
                unsafe_allow_html=True
            )



        # --- Price Chart ---
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Price",
            line=dict(color="blue")
        ))
        price_fig.update_layout(
            title=f"{ticker} Intraday Price (Including Pre-market & After-hours)",
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True
        )
        st.plotly_chart(price_fig)

        # --- Volume Chart ---
        volume_fig = go.Figure()
        volume_fig.add_trace(go.Bar(
            x=data.index,
            y=data["Volume"],
            name="Volume",
            marker=dict(color="grey")
        ))
        volume_fig.update_layout(
            title=f"{ticker} Intraday Volume (Including Pre-market & After-hours)",
            xaxis_title="Time",
            yaxis_title="Volume",
            showlegend=True
        )
        st.plotly_chart(volume_fig)


        # Show raw data (reversed for most recent on top)
        st.write(data[["Close", "Volume"]][::-1])

        # Refresh button logic
        if refresh_button:
            st.experimental_rerun()

    except Exception as e:
        st.error(f"Error fetching data: {e}")

st.sidebar.header("Current Prices")

if tickers_list:
    for t in tickers_list:
        try:
            yf_t = yf.Ticker(t)
            
            # Get today's intraday with pre/post
            t_data = yf_t.history(period="1d", interval="1m", prepost=True)
            
            # Get last 5 days to pick yesterday's close
            daily_data = yf_t.history(period="5d", interval="1d")
            
            if not t_data.empty and not daily_data.empty:
                latest = t_data["Close"].iloc[-1]
                
                # Yesterday's close (last row excluding today)
                if len(daily_data) > 1:
                    prev_close = daily_data["Close"].iloc[-2]
                else:
                    prev_close = daily_data["Close"].iloc[-1]
                
                price_diff = latest - prev_close
                percent_diff = (price_diff / prev_close) * 100 if prev_close != 0 else 0
                color = "green" if percent_diff >= 0 else "red"
                
                st.sidebar.markdown(
                    f"**{t}**: ${latest:.2f} "
                    f"<span style='color:{color}'>({percent_diff:+.2f}%)</span>",
                    unsafe_allow_html=True
                )
            else:
                st.sidebar.write(f"**{t}**: No data")
        except Exception as e:
            st.sidebar.write(f"**{t}**: Error ({e})")