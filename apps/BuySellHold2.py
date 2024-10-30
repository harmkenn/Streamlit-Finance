import streamlit as st
import yfinance as yf
import mplfinance as mpf

def plot_ohlc(tickerSymbol):
    tickerData = yf.Ticker(tickerSymbol)
    tickerDf = tickerData.history(period='1y')

    fig, ax = mpf.plot(tickerDf, type='candle', style='yahoo', returnfig=True)
    st.pyplot(fig)

st.title('OHLC Chart')
ticker_symbol = st.text_input('Enter a ticker symbol (e.g., AAPL, GOOGL)')

if ticker_symbol:
    plot_ohlc(ticker_symbol)