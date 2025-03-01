import streamlit as st
import yfinance as yf
import pandas as pd

# Function to get earnings dates for the next 12 months
def get_earnings_dates(ticker):
    stock = yf.Ticker(ticker)
    earnings_dates = stock.earnings_dates
    
    # Convert to DataFrame for easier manipulation
    earnings_df = pd.DataFrame(earnings_dates).reset_index()
    
    # Convert 'Earnings Date' to datetime and handle timezone issues
    earnings_df['Earnings Date'] = pd.to_datetime(earnings_df['Earnings Date']).dt.tz_localize(None)
    
    # Filter earnings dates within the next 12 months
    current_date = pd.to_datetime("today")
    future_dates = earnings_df[earnings_df['Earnings Date'] > current_date]
    
    # Only keep earnings dates for the next 12 months
    future_12_months = future_dates[future_dates['Earnings Date'] <= current_date + pd.DateOffset(months=12)]
    
    return future_12_months[['Earnings Date']]

# Function to get options expiration dates for the next 12 months
def get_options_dates(ticker):
    stock = yf.Ticker(ticker)
    try:
        options_dates = stock.options
    except:
        return pd.DataFrame(columns=["Options Expiration Date"])
    
    # Convert to DataFrame and filter for dates within the next 12 months
    if not options_dates:
        return pd.DataFrame(columns=["Options Expiration Date"])
    
    options_df = pd.DataFrame(options_dates, columns=["Options Expiration Date"])
    options_df['Options Expiration Date'] = pd.to_datetime(options_df['Options Expiration Date']).dt.tz_localize(None)
    
    # Filter options expiration dates within the next 12 months
    current_date = pd.to_datetime("today")
    future_dates = options_df[options_df['Options Expiration Date'] > current_date]
    
    # Only keep options expiration dates for the next 12 months
    future_12_months = future_dates[future_dates['Options Expiration Date'] <= current_date + pd.DateOffset(months=12)]
    
    return future_12_months[['Options Expiration Date']]

# Function to get dividend pay dates for the next 12 months
def get_dividend_dates(ticker):
    stock = yf.Ticker(ticker)
    try:
        dividends = stock.dividends
    except:
        return pd.DataFrame(columns=["Dividend Pay Date"])
    
    if dividends.empty:
      return pd.DataFrame(columns=["Dividend Pay Date"])

    dividends_df = pd.DataFrame(dividends).reset_index()
    dividends_df.rename(columns={'Date': 'Dividend Pay Date'}, inplace=True)
    dividends_df['Dividend Pay Date'] = pd.to_datetime(dividends_df['Dividend Pay Date']).dt.tz_localize(None)
    
    current_date = pd.to_datetime("today")
    future_dates = dividends_df[dividends_df['Dividend Pay Date'] > current_date]
    future_12_months = future_dates[future_dates['Dividend Pay Date'] <= current_date + pd.DateOffset(months=12)]
    
    return future_12_months[['Dividend Pay Date']]

# Streamlit UI setup
st.title("Stock Earnings, Options Expiration, and Dividend Dates")
st.write("Enter a stock ticker symbol to get earnings, options expiration, and dividend dates for the next 12 months.")

# Input for the stock ticker
ticker_input = st.text_input("Stock Ticker (e.g. AAPL, TSLA)")

if ticker_input:
    # Get earnings dates for the input ticker
    try:
        earnings_dates = get_earnings_dates(ticker_input)
        options_dates = get_options_dates(ticker_input)
        dividend_dates = get_dividend_dates(ticker_input)
        
        # Display earnings dates
        if not earnings_dates.empty:
            st.write(f"Earnings Dates for {ticker_input} in the next 12 months:")
            st.write(earnings_dates)
        else:
            st.write(f"No earnings dates found for {ticker_input} in the next 12 months.")
        
        # Display options expiration dates
        if not options_dates.empty:
            st.write(f"Options Expiration Dates for {ticker_input} in the next 12 months:")
            st.write(options_dates)
        else:
            st.write(f"No options expiration dates found for {ticker_input} in the next 12 months.")

        # Display dividend dates
        if not dividend_dates.empty:
            st.write(f"Dividend Pay Dates for {ticker_input} in the next 12 months:")
            st.write(dividend_dates)
        else:
            st.write(f"No dividend pay dates found for {ticker_input} in the next 12 months.")
    
    except Exception as e:
        st.write(f"Error fetching data for {ticker_input}: {str(e)}")