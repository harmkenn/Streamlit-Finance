import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.title("SOXL Weekly High-Low Volatility")

try:
    # Date range for the last 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)

    # Download SOXL data using yfinance
    soxl = yf.download("SOXL", start=start_date, end=end_date)

    if soxl.empty:
        st.error("Could not retrieve SOXL data. Please check the ticker symbol or your internet connection.")
        st.stop()

    # Resample to weekly data, taking the high and low
    weekly_high = soxl['High'].resample('W').max()
    weekly_low = soxl['Low'].resample('W').min()

    # Calculate the weekly high-low range
    weekly_range = weekly_high - weekly_low

    # Calculate the standard deviation of the weekly range
    std_dev = weekly_range.std()

    # Display the results
    st.write(f"Standard Deviation of Weekly High-Low Range (last 5 years): {std_dev:.2f}")

    # Optional: Display a chart of the weekly range
    st.subheader("Weekly High-Low Range")
    chart_data = pd.DataFrame({'Weekly Range': weekly_range})
    st.line_chart(chart_data)

    # Optional: Display the raw weekly range data
    if st.checkbox("Show Raw Data"):
        st.subheader("Weekly Range Data")
        st.dataframe(weekly_range)

except Exception as e:
    st.error(f"An error occurred: {e}")