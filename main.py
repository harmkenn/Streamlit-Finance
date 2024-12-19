import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf

# Fetch historical data for SOXL
data = yf.download("SOXL", start="2018-01-01", end="2023-01-01")


# Resample data to weekly frequency and get OHLC values
weekly_data = data['Close'].resample('W').ohlc()


def plot_ohlc(weekly_data):
    fig = go.Figure(data=[go.Candlestick(x=weekly_data.index,
                                         open=weekly_data['open'],
                                         high=weekly_data['high'],
                                         low=weekly_data['low'],
                                         close=weekly_data['close'])])
    st.plotly_chart(fig)

st.title("SOXL OHLC Data by Week")
st.write("Displaying weekly OHLC data for SOXL over the last 5 years")

plot_ohlc(weekly_data)
