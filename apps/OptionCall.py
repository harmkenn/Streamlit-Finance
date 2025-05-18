import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Option Calls App")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    stock_ticker = st.text_input("Ticker", value="TQQQ")

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
            with col2:
                st.metric(label="Current Price", value=f"${current_price:.2f}")
            with col3:
                selected_date = st.selectbox("Expiration Date", options)

            calls = stock.option_chain(selected_date).calls

            # Placeholder for Delta (not available via yfinance)
            df = pd.DataFrame({
                "Strike": calls["strike"],
                "Premium": calls["lastPrice"],
                "Open Interest": calls["openInterest"],
                "Volume": calls["volume"],
                "IV (%)": (calls["impliedVolatility"] * 100).round(2),
                "Delta": "N/A"  # Placeholder
            })

            st.subheader("Call Option Chain")
            st.dataframe(df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
