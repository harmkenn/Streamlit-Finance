import streamlit as st
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Stock OHLC Graph")

stock_symbol = st.text_input("Enter stock symbol (e.g. AAPL, GOOG, etc.)")

if stock_symbol:
    data = yf.download(stock_symbol, period="5y")

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.lineplot(x=data.index, y=data['Open'], ax=ax, label='Open')
    sns.lineplot(x=data.index, y=data['High'], ax=ax, label='High')
    sns.lineplot(x=data.index, y=data['Low'], ax=ax, label='Low')
    sns.lineplot(x=data.index, y=data['Close'], ax=ax, label='Close')

    ax.set_title(f"{stock_symbol} OHLC Graph")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    st.pyplot(fig)