import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.title("Stock OHLC Graph")

stock_symbol = st.text_input("Enter stock symbol (e.g. AAPL, GOOG, etc.)")

if stock_symbol:
    data = yf.download(stock_symbol, period="5y")

    fig = go.Figure(data=[go.Ohlc(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])

    fig.update_layout(title=f"{stock_symbol} OHLC Graph", xaxis_title="Date", yaxis_title="Price")

    st.plotly_chart(fig, use_container_width=True)