import streamlit as st
from yahooquery import Ticker
import pandas as pd
from datetime import datetime, timedelta

st.title("Dividend Information App")

col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=2 * 365))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())
with col3:
    tickers_list = [t.strip().upper() for t in st.session_state.get("tickers", "").split(",") if t.strip()]
    ticker = st.selectbox("Select Stock Ticker", tickers_list) if tickers_list else st.text_input("Enter ticker").upper()

if ticker:
    try:
        tk = Ticker(ticker)

        # Fetch price history
        price_hist = tk.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if price_hist.empty:
            st.error(f"No price data found for {ticker} in the selected date range.")
        else:
            # If MultiIndex (multiple tickers), extract relevant ticker
            if isinstance(price_hist.index, pd.MultiIndex):
                price_hist = price_hist.xs(ticker, level=0)
            
            # Fetch dividends separately
            divs = tk.dividends
            if divs.empty:
                st.info("No dividend data found for this ticker.")
                divs = pd.DataFrame(columns=['date', 'dividends'])
            else:
                # dividends is a Series indexed by datetime, convert to DataFrame and filter dates
                divs = divs.reset_index()
                divs.columns = ['date', 'dividends']
                divs['date'] = pd.to_datetime(divs['date'])
                divs = divs[(divs['date'] >= pd.to_datetime(start_date)) & (divs['date'] <= pd.to_datetime(end_date))]
            
            # Merge dividend data onto price history by date
            price_hist = price_hist.reset_index()
            price_hist['date'] = pd.to_datetime(price_hist['date'])
            merged = pd.merge(price_hist, divs, how='left', on='date')
            merged['dividends'].fillna(0, inplace=True)

            # Calculate dividend yield (%)
            merged['Dividend Yield (%)'] = (merged['dividends'] / merged['close']) * 100

            # Filter only dividend payout days
            div_data = merged[merged['dividends'] != 0][['date', 'close', 'dividends', 'Dividend Yield (%)']]
            div_data = div_data.rename(columns={'close': 'Close Price', 'dividends': 'Dividend'}).iloc[::-1]

            if div_data.empty:
                st.info("No dividend payouts found in the selected date range.")
            else:
                st.write("Dividend Payouts and Corresponding Close Prices:")
                st.dataframe(div_data)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
else:
    st.info("Please enter or select a stock ticker symbol.")
