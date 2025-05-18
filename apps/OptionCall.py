import streamlit as st
import yfinance as yf
import pandas as pd

# Create a title for the app
st.title("Option Calls App")

# Create an input field for the stock ticker
stock_ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA)", value="AAPL")

# Get the current stock price
if stock_ticker:
    stock = yf.Ticker(stock_ticker)
    current_price = stock.info["regularMarketPrice"]
    st.write(f"Current Stock Price: ${current_price:.2f}")

    # Get the options data
    options = stock.options
    if options:
        calls = stock.option_chain(options[0]).calls
        # Create a pandas dataframe with the required columns
        df = pd.DataFrame({
            "Call Strike Price": calls["strike"],
            "Call Premium": calls["lastPrice"],
            "Delta": calls["delta"]
        })
        # Display the dataframe
        st.write(df)
    else:
        st.write("No options data available for this stock.")