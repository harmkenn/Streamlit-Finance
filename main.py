import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf

def get_soxl_data(start_date, end_date):
  """Fetches historical data for SOXL between given dates."""
  try:
    return yf.download("SOXL", start=start_date, end=end_date)
  except Exception as e:
    st.error(f"Error downloading data: {e}")
    return None

def plot_ohlc(weekly_data):
  """Creates a Plotly candlestick chart for the OHLC data."""
  fig = go.Figure(data=[go.Candlestick(x=weekly_data.index,
                                       open=weekly_data['open'],
                                       high=weekly_data['high'],
                                       low=weekly_data['low'],
                                       close=weekly_data['close'])])
  fig.update_layout(xaxis_title="Date", yaxis_title="Price ($)")
  st.plotly_chart(fig)

st.title("SOXL OHLC Data by Week")

# Date range selection
start_date = st.date_input("Start Date", value=datetime(2018, 1, 1))
end_date = st.date_input("End Date", value=datetime(2023, 1, 1))

# Data download and processing
data = get_soxl_data(start_date, end_date)
if data is not None:
  weekly_data = data['Close'].resample('W').ohlc()
  plot_ohlc(weekly_data)