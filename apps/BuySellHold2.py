import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import datetime as dt


# Set up the Streamlit interface
st.title("Closing Price Chart")


# Set the ticker symbol
ticker = st.text_input("Enter ticker symbol", value="AAPL")


# Set the start and end dates
start_date = st.date_input("Select start date", value=dt.date(2020, 1, 1), min_value=dt.date(2010, 1, 1), max_value=dt.date.today())
end_date = st.date_input("Select end date", value=dt.date.today(), min_value=dt.date(2010, 1, 1), max_value=dt.date.today())


# Fetch historical data for the selected ticker and date range
data = yf.download(ticker, start=start_date, end=end_date, interval='1d')


# Create the closing price chart
fig = go.Figure()


# Add closing price data
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Closing Price'))


# Update layout
fig.update_layout(title=f'Closing Price Chart for {ticker}',xaxis_title='Date',yaxis_title='Price')


# Streamlit app layout
st.plotly_chart(fig)