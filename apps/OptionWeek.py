import streamlit as st
import yfinance as yf
import pandas as pd

# Function to get earnings dates for the next 12 months
def get_earnings_dates(ticker):
    stock = yf.Ticker(ticker)
    try:
        earnings_dates = stock.earnings_dates
        earnings_df = pd.DataFrame(earnings_dates).reset_index()
        earnings_df['Earnings Date'] = pd.to_datetime(earnings_df['Earnings Date']).dt.tz_localize(None)
        current_date = pd.to_datetime("today")
        future_dates = earnings_df[earnings_df['Earnings Date'] > current_date]
        future_12_months = future_dates[future_dates['Earnings Date'] <= current_date + pd.DateOffset(months=12)]
        return future_12_months[['Earnings Date']]
    except Exception:
        return pd.DataFrame(columns=["Earnings Date"])

# Function to get options expiration dates for the next 12 months
def get_options_dates(ticker):
    stock = yf.Ticker(ticker)
    try:
        options_dates = stock.options
        if not options_dates:
            return pd.DataFrame(columns=["Options Expiration Date"])
        options_df = pd.DataFrame(options_dates, columns=["Options Expiration Date"])
        options_df['Options Expiration Date'] = pd.to_datetime(options_df['Options Expiration Date']).dt.tz_localize(None)
        current_date = pd.to_datetime("today")
        future_dates = options_df[options_df['Options Expiration Date'] > current_date]
        future_12_months = future_dates[future_dates['Options Expiration Date'] <= current_date + pd.DateOffset(months=12)]
        return future_12_months[['Options Expiration Date']]
    except Exception:
        return pd.DataFrame(columns=["Options Expiration Date"])

# Function to get dividend payout amounts and dates for the last 12 months
def get_past_dividend_payments(ticker):
    stock = yf.Ticker(ticker)
    try:
        dividends = stock.dividends
        if dividends.empty:
            return pd.DataFrame(columns=["Dividend Pay Date", "Dividend Amount"])
        dividends_df = pd.DataFrame(dividends).reset_index()
        dividends_df.rename(columns={'Date': 'Dividend Pay Date', 'Dividends': 'Dividend Amount'}, inplace=True)
        dividends_df['Dividend Pay Date'] = pd.to_datetime(dividends_df['Dividend Pay Date']).dt.tz_localize(None)
        current_date = pd.to_datetime("today")
        past_12_months = dividends_df[dividends_df['Dividend Pay Date'] >= current_date - pd.DateOffset(months=12)]
        past_12_months['Dividend Yield'] = (past_12_months['Dividend Amount'] / past_12_months['Close']) * 100
        return past_12_months[['Dividend Pay Date', 'Dividend Amount','Dividend Yield']]
    except Exception:
        return pd.DataFrame(columns=["Dividend Pay Date", "Dividend Amount", "Dividend Yield"])

# Streamlit UI setup
st.title("Stock Earnings, Options Expiration, and Dividend Data")
st.write("Enter a stock ticker symbol to get earnings, options expiration, and past 12 months dividend data.")

# Input for the stock ticker
ticker_input = st.text_input("Stock Ticker (e.g. AAPL, TSLA)",'MSTY')

if ticker_input:
    # Get all dates
    earnings_dates = get_earnings_dates(ticker_input)
    options_dates = get_options_dates(ticker_input)
    dividend_data = get_past_dividend_payments(ticker_input)

    col1, col2, col3 = st.columns(3)
    # Display earnings dates
    with col1:
        if not earnings_dates.empty:
            st.write(f"Earnings Dates for {ticker_input} in the next 12 months:")
            st.write(earnings_dates)
        else:
            st.write(f"No earnings dates found for {ticker_input} in the next 12 months.")

    # Display options expiration dates
    with col2:
        if not options_dates.empty:
            st.write(f"Options Expiration Dates for {ticker_input} in the next 12 months:")
            st.write(options_dates)
        else:
            st.write(f"No options expiration dates found for {ticker_input} in the next 12 months.")

    # Display dividend dates and amounts
    with col3:
        if not dividend_data.empty:
            st.write(f"Dividend Payments for {ticker_input} in the last 12 months:")
            st.write(dividend_data)
        else:
            st.write(f"No dividend payments found for {ticker_input} in the last 12 months.")