import streamlit as st
import yfinance as yf
import pandas as pd

# Create a title for the app
st.title("Option Calls App")

# Create an input field for the stock ticker
stock_ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA)", value="TQQQ")

# Get the current stock price
if stock_ticker:
    try:
        stock = yf.Ticker(stock_ticker)
        current_price = stock.info.get("regularMarketPrice", None)
        
        if current_price is None:
            st.error("Could not retrieve current stock price.")
        else:
            st.write(f"Current Stock Price: ${current_price:.2f}")

            # Show expiration dates for user to pick
            options = stock.options
            if options:
                selected_date = st.selectbox("Select expiration date", options)

                calls = stock.option_chain(selected_date).calls
                df = pd.DataFrame({
                    "Call Strike Price": calls["strike"],
                    "Call Premium": calls["lastPrice"],
                    "Bid": calls["bid"],
                    "Ask": calls["ask"],
                    "Open Interest": calls["openInterest"],
                    "Volume": calls["volume"],
                    "Implied Volatility": calls["impliedVolatility"]
                })
                # Display the dataframe
                st.dataframe(df)
            else:
                st.write("No options data available for this stock.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
