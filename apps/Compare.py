import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf

# Fetch SOXL ETF data for the past week
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, interval="1m", prepost=True)
    data.reset_index(inplace=True)
    return data

# Create Streamlit app
def main():
    st.title("SOXL ETF Price Graph (Including Pre-market and Post-market Hours)")

    # Define date range
    start_date = st.date_input("Start Date", pd.to_datetime("today") - pd.Timedelta(days=7))
    end_date = st.date_input("End Date", pd.to_datetime("today"))

    if start_date and end_date:
        data = fetch_data("SOXL", start_date, end_date)
        fig = px.line(data, x="Datetime", y="Close", title="SOXL ETF Price")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
