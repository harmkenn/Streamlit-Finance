import streamlit as st
import yfinance as yf
import plotly.express as px
from plotly.subplots import make_subplots

st.title("Intraday Stock Prices")

stock_symbol = st.text_input("Enter stock symbol (e.g. AAPL, GOOG, MSFT):")

if stock_symbol:
    refresh_button = st.button("Refresh")

    if refresh_button:
        ticker = yf.Ticker(stock_symbol)
        data = ticker.history(period="5d", interval="1m")

        # Reverse the order of the data to get the newest row first
        data = data.iloc[::-1]

        # Select only the Close, Volume, and Dividends columns
        data = data[["Close", "Volume", "Dividends"]]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(px.line(data, x=data.index, y="Close").data[0], secondary_y=False)

        fig.add_trace(px.bar(data, x=data.index, y="Volume").data[0], secondary_y=True)

        fig.data[1].marker.color = 'red'

        fig.update_layout(title=f"{stock_symbol} Intraday Prices", xaxis_title="Time", yaxis_title="Price")
        fig.update_yaxes(title_text="Volume", secondary_y=True)

        st.plotly_chart(fig)

        st.write(data)
    else:
        ticker = yf.Ticker(stock_symbol)
        data = ticker.history(period="5d", interval="1m")

        # Reverse the order of the data to get the newest row first
        data = data.iloc[::-1]

        # Select only the Close, Volume, and Dividends columns
        data = data[["Close", "Volume", "Dividends"]]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(px.line(data, x=data.index, y="Close").data[0], secondary_y=False)

        fig.add_trace(px.bar(data, x=data.index, y="Volume").data[0], secondary_y=True)

        fig.data[1].marker.color = 'red'

        fig.update_layout(title=f"{stock_symbol} Intraday Prices", xaxis_title="Time", yaxis_title="Price")
        fig.update_yaxes(title_text="Volume", secondary_y=True)

        st.plotly_chart(fig)

        st.write(data)