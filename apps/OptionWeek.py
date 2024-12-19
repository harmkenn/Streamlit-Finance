import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta, datetime
import pandas as pd

st.title("Stock OHLC Plot with Moving Averages")

# Date inputs
col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("Start Date", date.today() - timedelta(days=5 * 365))
with col2:
    end_date = st.date_input("End Date", date.today())
with col3:
    # Ticker input
    ticker = st.text_input("Enter Stock Ticker", "SOXL").upper()

if ticker:
    if start_date > end_date:
        st.error("Error: Start date must be before end date.")
    else:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)

            if data.empty:
                st.error(f"No data found for {ticker} within the selected date range.")
            else:
                # Calculate moving averages
                data['50-day MA'] = data['Close'].rolling(window=50).mean()
                data['200-day MA'] = data['Close'].rolling(window=200).mean()

                # --- Highlight Third Fridays ---
                third_fridays = []
                for year in range(start_date.year, end_date.year + 1):
                    for month in [3, 6, 9, 12]:  # March, June, Sept, Dec
                        # Find the first Friday of the month
                        first_day = datetime(year, month, 1)
                        first_friday = first_day + timedelta((4 - first_day.weekday() + 7) % 7)

                        # Find the third Friday (add 2 weeks)
                        third_friday = first_friday + timedelta(weeks=2)
                        third_fridays.append(third_friday.date())

                # Convert the list of dates to pandas DatetimeIndex for efficient filtering
                third_fridays_index = pd.to_datetime(third_fridays)

                # Create a boolean mask to select rows corresponding to third Fridays
                third_fridays_mask = data.index.isin(third_fridays_index)

                # --- Plotting ---
                fig = go.Figure()

                # Add candlestick trace
                fig.add_trace(go.Candlestick(x=data.index,
                                            open=data['Open'],
                                            high=data['High'],
                                            low=data['Low'],
                                            close=data['Close'], name="OHLC"))

                # Add moving average traces
                fig.add_trace(go.Scatter(x=data.index, y=data['50-day MA'],
                                        mode='lines', name='50-day MA', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=data.index, y=data['200-day MA'],
                                        mode='lines', name='200-day MA', line=dict(color='red')))

                fig.update_layout(title=f"{ticker} OHLC Chart ({start_date} - {end_date})",
                                  xaxis_title="Date", yaxis_title="Price",
                                  xaxis_rangeslider_visible=False)

                st.plotly_chart(fig)

                st.subheader("Key Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", data['Close'][-1].round(2) if not data.empty else "N/A")
                with col2:
                    st.metric("High", data['High'].max().round(2) if not data.empty else "N/A")
                with col3:
                    st.metric("Low", data['Low'].min().round(2) if not data.empty else "N/A")

                with st.expander("Show raw data"):
                    # Highlight the third Fridays in the dataframe
                    styled_df = data.style.apply(lambda x: ['background-color: yellow' if third_fridays_mask[i] else '' for i in range(len(x))], axis=1)
                    st.dataframe(styled_df)

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please enter a stock ticker symbol.")