import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd


def plot_ohlc(tickerSymbol):
    tickerData = yf.Ticker(tickerSymbol)
    tickerDf = tickerData.history(period='2y')

    # Calculate moving averages
    tickerDf['MA50'] = tickerDf['Close'].rolling(window=50).mean()
    tickerDf['MA200'] = tickerDf['Close'].rolling(window=200).mean()

    # Create additional plots for moving averages
    addplot = [
        mpf.make_addplot(tickerDf['MA50'], color='g'),
        mpf.make_addplot(tickerDf['MA200'], color='r')
    ]

    fig, ax = mpf.plot(tickerDf, type='candle', style='yahoo', addplot=addplot, returnfig=True)
    st.pyplot(fig)


st.title('OHLC Chart')
ticker_symbol = st.text_input('Enter a ticker symbol (e.g., AAPL, GOOGL)')


if ticker_symbol:
    plot_ohlc(ticker_symbol)