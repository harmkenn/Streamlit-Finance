import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Option Calls App")

# Set up three columns
col1, col2, col3 = st.columns([1, 1, 2])  # Adjust column width as needed

# Column 1: Stock Ticker input
with col1:
    stock_ticker = st.text_input("Ticker", value="TQQQ")

# Proceed only if ticker is entered
if stock_ticker:
    try:
        stock = yf.Ticker(stock_ticker)
        current_price = stock.info.get("regularMarketPrice", None)
        options = stock.options

        if current_price is None:
            st.error("Could not retrieve current stock price.")
        elif not options:
            st.error("No options data available for this stock.")
        else:
            # Column 2: Show current price
            with col2:
                st.metric(label="Current Price", value=f"${current_price:.2f}")

            # Column 3: Expiration date selector
            with col3:
                selected_date = st.selectbox("Expiration Date", options)

            # Load call options for selected expiration
            calls = stock.option_chain(selected_date).calls

            # Display call options table
            df = pd.DataFrame({
                "Strike": calls["strike"],
                "Premium": calls["lastPrice"],
                "Bid": calls["bid"],
                "Ask": calls["ask"],
                "Open Interest": calls["openInterest"],
                "Volume": calls["volume"],
                "IV": calls["impliedVolatility"]
            })

            st.subheader("Call Option Chain")
            st.dataframe(df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
