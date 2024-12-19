import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

st.title("Stock OHLC Plot")



# Date inputs
col1, col2, col3= st.columns(3)
with col1:
    start_date = st.date_input("Start Date", date.today() - timedelta(days=5 * 365))  # Default 5 years ago
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
                fig = go.Figure(data=[go.Candlestick(x=data.index,
                                                    open=data['Open'],
                                                    high=data['High'],
                                                    low=data['Low'],
                                                    close=data['Close'])])

                fig.update_layout(title=f"{ticker} OHLC Chart ({start_date} - {end_date})",
                                  xaxis_title="Date",
                                  yaxis_title="Price",
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
                    st.dataframe(data)

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please enter a stock ticker symbol.")