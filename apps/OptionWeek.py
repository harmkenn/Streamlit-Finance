import streamlit as st
import yfinance as yf
import pandas as pd

# Function to get earnings dates for the next 12 months
def get_earnings_dates(ticker):
    stock = yf.Ticker(ticker)
    earnings_dates = stock.earnings_dates
    
    # Convert to DataFrame for easier manipulation
    earnings_df = pd.DataFrame(earnings_dates).reset_index()
    
    # Filter earnings dates within the next 12 months
    current_date = pd.to_datetime("today")
    future_dates = earnings_df[earnings_df['Earnings Date'] > current_date]
    
    # Only keep earnings dates for the next 12 months
    future_12_months = future_dates[future_dates['Earnings Date'] <= current_date + pd.DateOffset(months=12)]
    
    return future_12_months[['Earnings Date']]

# Streamlit UI setup
st.title("Stock Earnings Dates")
st.write("Enter a stock ticker symbol to get earnings dates for the next 12 months.")

# Input for the stock ticker
ticker_input = st.text_input("Stock Ticker (e.g. AAPL, TSLA)")

if ticker_input:
    # Get earnings dates for the input ticker
    try:
        earnings_dates = get_earnings_dates(ticker_input)
        if not earnings_dates.empty:
            st.write(f"Earnings Dates for {ticker_input} in the next 12 months:")
            st.write(earnings_dates)
        else:
            st.write(f"No earnings dates found for {ticker_input} in the next 12 months.")
    except Exception as e:
        st.write(f"Error fetching data for {ticker_input}: {str(e)}")
