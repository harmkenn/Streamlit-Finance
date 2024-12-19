import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_third_friday(year, month):
    first_day = datetime(year, month, 1)
    first_day_weekday = first_day.weekday()
    days_to_first_friday = (4 - first_day_weekday) % 7
    first_friday = first_day + timedelta(days=days_to_first_friday)
    third_friday = first_friday + timedelta(weeks=2)
    return third_friday.date()

def analyze_volatility(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

    if df.empty:
        st.warning("No data found for the given ticker and date range.")
        return None

    df['Weekly_Return'] = df['Adj Close'].resample('W').ffill().pct_change()
    df['Week_of_Month'] = df.index.to_series().apply(lambda x: (x.day - 1) // 7 + 1)

    df['Third_Friday_Week'] = False
    for year in range(df.index.min().year, df.index.max().year + 1):
        for month in range(1, 13):
            try:
                third_friday = get_third_friday(year, month)
                third_friday_week_start = third_friday - timedelta(days=third_friday.weekday())
                third_friday_week_end = third_friday_week_start + timedelta(days=6)
                df.loc[(df.index >= pd.to_datetime(third_friday_week_start)) & (df.index <= pd.to_datetime(third_friday_week_end)), 'Third_Friday_Week'] = True
            except ValueError:
                pass

    weekly_volatility = df.groupby('Week_of_Month')['Weekly_Return'].std()
    third_friday_volatility = df[df['Third_Friday_Week']]['Weekly_Return'].std()
    all_volatility = df['Weekly_Return'].std()

    st.write(f"Overall Annualized Volatility: {all_volatility*np.sqrt(52):.4f}")
    st.write("Annualized Volatility by Week of Month:")
    st.write(weekly_volatility*np.sqrt(52))
    st.write(f"Annualized Volatility during Third Friday Weeks: {third_friday_volatility*np.sqrt(52):.4f}")

    st.write("\nAverage Returns by week of the month")
    st.write(df.groupby('Week_of_Month')['Weekly_Return'].mean())
    st.write("\nAverage Return for Third Friday Weeks")
    st.write(df[df['Third_Friday_Week']]['Weekly_Return'].mean())
    return df

st.title("ETF Volatility Analysis")

ticker = st.text_input("Enter ETF Ticker", "SOXL").upper()
start_date = st.date_input("Start Date", datetime(2010, 1, 1))
end_date = st.date_input("End Date", datetime(2024, 1, 1))

if st.button("Analyze"):
    if ticker:
        df = analyze_volatility(ticker, start_date, end_date)
        if df is not None:
            st.dataframe(df.head())  # Display the first few rows of the DataFrame
            st.line_chart(df['Adj Close'])
            st.line_chart(df['Weekly_Return'])
            st.bar_chart(df.groupby('Week_of_Month')['Weekly_Return'].std()*np.sqrt(52)) #Bar chart of annualized volatility by week of the month
            st.bar_chart(df.groupby('Week_of_Month')['Weekly_Return'].mean()) #Bar chart of average return by week of month
            if not df[df['Third_Friday_Week']].empty: #Check to see if there are any third friday weeks in the data
                st.bar_chart(df[df['Third_Friday_Week']]['Weekly_Return'].mean()) #Bar chart of average return on third friday weeks

