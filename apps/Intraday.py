import streamlit as st
import yfinance as yf
import plotly.express as px

st.title("Intraday Stock Prices")

stock_symbol = st.text_input("Enter stock symbol (e.g. AAPL, GOOG, MSFT):")

if stock_symbol:
    ticker = yf.Ticker(stock_symbol)
    data = ticker.history(period="1d", interval="1m")

    fig = px.line(data, x=data.index, y="Close")
    fig.update_layout(title=f"{stock_symbol} Intraday Prices", xaxis_title="Time", yaxis_title="Price")

    volume_fig = px.bar(data, x=data.index, y="Volume", color_discrete_sequence=["red"])

    fig.add_trace(volume_fig.data[0], secondary_y=True)

    fig.update_layout(yaxis_title="Price", secondary_y_title="Volume", secondary_y=True)

    st.plotly_chart(fig)

    st.write(data)