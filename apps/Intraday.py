import streamlit as st
import yfinance as yf
import plotly.express as px
from plotly.subplots import make_subplots

st.title("Intraday Stock Prices")

# User input for stock symbol
stock_symbol = st.text_input("Enter stock symbol (e.g. AAPL, GOOG, MSFT):", "NVDA").upper()

if stock_symbol:
    try:
        # Fetch stock data
        ticker = yf.Ticker(stock_symbol)
        data = ticker.history(period="5d", interval="1m")

        if data.empty:
            st.error(f"No data found for {stock_symbol}. Please check the symbol and try again.")
        else:
            # Keep the newest row first
            data = data.iloc[::-1]

            # Select relevant columns
            data = data[["Close", "Volume"]]

            # Create subplots for price and volume
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(px.line(data, x=data.index, y="Close").data[0], secondary_y=False)
            fig.add_trace(px.bar(data, x=data.index, y="Volume").data[0], secondary_y=True)

            # Style updates
            fig.data[1].marker.color = 'red'
            fig.update_layout(title=f"{stock_symbol} Intraday Prices", xaxis_title="Time", yaxis_title="Price")
            fig.update_yaxes(title_text="Volume", secondary_y=True)

            # Display chart
            st.plotly_chart(fig)

            # Show raw data
            st.write(data)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
