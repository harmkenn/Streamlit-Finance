import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import datetime as dt

# Set up the Streamlit interface
st.title("Buy Sell Hold Strategy 2")

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


# Set the ticker symbol
c1, c2, c3, c4 = st.columns(4)
with c1:
    ticker_symbol = st.selectbox("Select Ticker", ['ARGT','EDEN','INCO','KBWP','SMIN','SOXL','SPXL','SSO','TECL','TQQQ','UPRO'])
with c2:
    bound = st.number_input("Bound", min_value=0.0, max_value=10.0, value=1.1, step=0.1)
with c4:
    start_date = st.date_input("Select start date", value=dt.date(2020, 1, 1), min_value=dt.date(2010, 1, 1), max_value=dt.date.today())  # replace with your desired start date
    end_date = st.date_input("Select end date", value=dt.date.today(), min_value=dt.date(2010, 1, 1), max_value=dt.date.today())  # replace with your desired start da
if ticker_symbol:
    plot_ohlc(ticker_symbol)