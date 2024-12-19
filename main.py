import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.title("SOXL Weekly High-Low Volatility (3rd Friday vs. Others)")

try:
    # Date range for the last 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)

    # Download SOXL data using yfinance
    soxl = yf.download("SOXL", start=start_date, end=end_date)

    if soxl.empty:
        st.error("Could not retrieve SOXL data. Please check the ticker symbol or your internet connection.")
        st.stop()

    # Add a 'Week' column and a 'Third Friday' boolean column
    soxl['Week'] = soxl.index.to_period('W')
    soxl['Third Friday'] = soxl.index.day == 21 #Approximate, see explanation below.

    # Resample to weekly data, taking the high and low
    weekly_high = soxl.groupby('Week')['High'].max()
    weekly_low = soxl.groupby('Week')['Low'].min()
    weekly_third_friday = soxl.groupby('Week')['Third Friday'].any()

    # Calculate the weekly high-low range
    weekly_range = weekly_high - weekly_low

    # Separate third Friday weeks and other weeks
    third_friday_weeks = weekly_range[weekly_third_friday]
    other_weeks = weekly_range[~weekly_third_friday]

    # Calculate standard deviations
    std_dev_third_friday = third_friday_weeks.std()
    std_dev_other_weeks = other_weeks.std()

    # Display the results
    st.write(f"Standard Deviation of Weekly High-Low Range (3rd Friday Weeks): {std_dev_third_friday:.2f}")
    st.write(f"Standard Deviation of Weekly High-Low Range (Other Weeks): {std_dev_other_weeks:.2f}")

    # Display as a table
    st.subheader("Standard Deviations")
    st.table(pd.DataFrame({
        "Group": ["3rd Friday Weeks", "Other Weeks"],
        "Standard Deviation": [std_dev_third_friday, std_dev_other_weeks]
    }))


    # Optional: Display histograms
    st.subheader("Distribution of Weekly Ranges")
    col1, col2 = st.columns(2)
    with col1:
        st.write("3rd Friday Weeks")
        st.bar_chart(third_friday_weeks)
    with col2:
        st.write("Other Weeks")
        st.bar_chart(other_weeks)

except Exception as e:
    st.error(f"An error occurred: {e}")