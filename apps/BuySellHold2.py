import streamlit as st
import yfinance as yf
import plotly.express as px

st.title("Stock OHLC Graph")

stock_symbol = st.text_input("Enter stock symbol (e.g. AAPL, GOOG, etc.)")

if stock_symbol:
    data = yf.download(stock_symbol, period="5y")

    fig = px.line(data, x=data.index, y=['Open', 'High', 'Low', 'Close'], 
                  title=f"{stock_symbol} OHLC Graph", 
                  labels={'value': 'Price', 'index': 'Date'})

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{'visible': [True, True, True, True]}],
                        label="All",
                        method="update"
                    ),
                    dict(
                        args=[{'visible': [True, False, False, False]}],
                        label="Open",
                        method="update"
                    ),
                    dict(
                        args=[{'visible': [False, True, False, False]}],
                        label="High",
                        method="update"
                    ),
                    dict(
                        args=[{'visible': [False, False, True, False]}],
                        label="Low",
                        method="update"
                    ),
                    dict(
                        args=[{'visible': [False, False, False, True]}],
                        label="Close",
                        method="update"
                    ),
                ]),
            ]
        )

    st.plotly_chart(fig, use_container_width=True)