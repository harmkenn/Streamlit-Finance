import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf

# Fetch SOXL ETF data
def fetch_data():
    ticker = "SOXL"
    data = yf.download(ticker, period="1d", interval="1m", prepost=True)
    data.reset_index(inplace=True)
    return data

# Create Streamlit app
def main():
    st.title("SOXL ETF Price Graph (Including Pre-market and Post-market Hours)")
    data = fetch_data()

    fig = px.line(data, x="Datetime", y="Close", title="SOXL ETF Price")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
