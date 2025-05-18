import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Option Calls App")

# Set up columns for input layout
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

            # Load call options for selected expiration
            calls = stock.option_chain(selected_date).calls

            # Filter strikes within 90% to 120% of current price
            lower_bound = 0.9 * current_price
            upper_bound = 1.2 * current_price
            filtered_calls = calls[(calls["strike"] >= lower_bound) & (calls["strike"] <= upper_bound)]

            # Create the DataFrame
            df = pd.DataFrame({
                "Strike": filtered_calls["strike"],
                "Premium": filtered_calls["lastPrice"],
                "Open Interest": filtered_calls["openInterest"],
                "Volume": filtered_calls["volume"],
                "IV (%)": (filtered_calls["impliedVolatility"] * 100).round(2),
                "Delta": "N/A"  # Placeholder unless you compute Delta separately
            })

            st.subheader("Filtered Call Option Chain (90%–120% Strike Range)")
            st.dataframe(df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
