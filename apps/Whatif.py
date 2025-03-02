import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def calculate_etf_value(ticker, initial_investment):
    """Calculates the current value of an ETF investment with reinvested dividends."""

    try:
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        stock = yf.Ticker(ticker)
        historical_data = stock.history(start=start_date, end=end_date, actions=True)
        dividends = historical_data['Dividends']
        historical_prices = historical_data['Close']

        if historical_prices.empty:
            return "No historical price data found for the given ticker."
        if dividends.empty:
          return "No dividend data found."

        # Calculate initial shares
        initial_price = historical_prices.iloc[0]
        shares = initial_investment / initial_price

        # Reinvest dividends
        for date, dividend in dividends.items():
            if dividend > 0:
                price_at_dividend = historical_prices.asof(date)
                if pd.isna(price_at_dividend):
                    #if there is no price on the dividend date, use the next available price.
                    price_at_dividend = historical_prices[historical_prices.index > date].iloc[0]

                shares += (shares * dividend) / price_at_dividend

        # Calculate current value
        current_price = historical_prices.iloc[-1]
        current_value = shares * current_price

        return f"The current value of your investment is: ${current_value:.2f}"

    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit app
st.title("ETF Growth Calculator")
st.write("Enter an ETF ticker symbol and initial investment to calculate its current value.")

ticker = st.text_input("ETF Ticker Symbol (e.g., SPY, VOO, MSTY):")
initial_investment = st.number_input("Initial Investment ($):", value=10000.0)

if st.button("Calculate"):
    if ticker:
        result = calculate_etf_value(ticker, initial_investment)
        st.write(result)
    else:
        st.write("Please enter an ETF ticker symbol.")