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

    # Create 'Week' and 'Third Friday' columns using datetime indexes directly
    soxl['WeekStart'] = soxl.index - pd.to_timedelta(soxl.index.dayofweek, unit='D')
    soxl['ThirdFriday'] = (soxl.index.day >= 15) & (soxl.index.day <= 21) & (soxl.index.weekday == 4)

    # Group by week start date
    weekly_high = soxl.groupby('WeekStart')['High'].max()
    weekly_low = soxl.groupby('WeekStart')['Low'].min()
    weekly_third_friday = soxl.groupby('WeekStart')['ThirdFriday'].any()

    weekly_range = weekly_high - weekly_low

    # Remove NaT values *before* separating the data
    weekly_range = weekly_range.dropna()
    weekly_third_friday = weekly_third_friday[weekly_range.index] #align index after dropping NaN

    third_friday_weeks = weekly_range[weekly_third_friday]
    other_weeks = weekly_range[~weekly_third_friday]

    # Calculate standard deviations
    std_dev_third_friday = third_friday_weeks.std()
    std_dev_other_weeks = other_weeks.std()

    # ... (Display results as before)

    # Improved Histograms (using .to_numpy())

    st.subheader("Distribution of Weekly Ranges")
    col1, col2 = st.columns(2)

    with col1:
        st.write("3rd Friday Weeks")
        if not third_friday_weeks.empty:  # Check for empty data
            st.hist(third_friday_weeks.to_numpy(), bins=20)
        else:
            st.write("No data available for 3rd Friday Weeks.")

    with col2:
        st.write("Other Weeks")
        if not other_weeks.empty: # Check for empty data
            st.hist(other_weeks.to_numpy(), bins=20)
        else:
            st.write("No data available for Other Weeks.")

except Exception as e:
    st.error(f"An error occurred: {e}")
