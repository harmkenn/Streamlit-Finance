import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import numpy as np
import datetime

st.set_page_config(layout="wide")
st.title("📈 Stock Comparison Tool (with 7-Day Prophet Forecast)")

# Split and clean tickers
ticker_list = [ticker.strip().upper() for ticker in st.session_state.get("tickers", "").split(',') if ticker.strip()]

col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("Start date", datetime.date(2024, 6, 1))
with col2:
    end_date = st.date_input("End date", datetime.date.today())
with col3:
    selected_tickers = st.multiselect(
        "Select Tickers to Compare",
        options=ticker_list,
        default=ticker_list[:3]  # Pre-select up to 3 tickers
    )

# Split and clean tickers
ticker_list = [ticker.strip().upper() for ticker in tickers.split(',') if ticker.strip()]

# Validate dates
if start_date >= end_date:
    st.error("End date must be after start date.")
    st.stop()

if not ticker_list:
    st.error("Please enter at least one valid ticker symbol.")
    st.stop()

# Download data
@st.cache_data
def load_data(ticker_list, start, end):
    raw_data = yf.download(ticker_list, start=start, end=end, group_by='ticker', auto_adjust=False)

    if len(ticker_list) == 1:
        ticker = ticker_list[0]
        if 'Adj Close' not in raw_data.columns:
            raise ValueError(f"No 'Adj Close' data for {ticker}")
        df = raw_data[['Adj Close']]
        df.columns = [ticker]
        return df.dropna()

    adj_close_data = pd.DataFrame()
    for ticker in ticker_list:
        try:
            ticker_data = raw_data[ticker]['Adj Close'].dropna()
            adj_close_data[ticker] = ticker_data
        except (KeyError, TypeError):
            st.warning(f"⚠️ No data found for {ticker}. It will be skipped.")

    if adj_close_data.empty:
        raise ValueError("No valid data returned for any tickers.")
    
    return adj_close_data

# Load and process data
data = load_data(ticker_list, start_date, end_date)
normalized_data = data / data.iloc[0] * 100

# Plot
st.subheader("Normalized Stock Prices (Start at 100)")
st.line_chart(normalized_data)

# Summary statistics
st.subheader("📋 Summary Statistics")
returns = data.pct_change().dropna()
stats = pd.DataFrame(index=data.columns)
stats["Start Price"] = data.iloc[0]
stats["End Price"] = data.iloc[-1]
stats["Total Return (%)"] = ((data.iloc[-1] / data.iloc[0]) - 1) * 100
num_years = (end_date - start_date).days / 365.25
stats["Annualized Return (%)"] = ((data.iloc[-1] / data.iloc[0]) ** (1 / num_years) - 1) * 100
stats["Volatility (Std Dev of Daily Returns)"] = returns.std()
stats["7-Day Predicted Price"] = np.nan

# Prophet-based 7-day prediction
for ticker in data.columns:
    try:
        df = data[[ticker]].dropna().reset_index()
        df.columns = ['ds', 'y']

        model = Prophet(daily_seasonality=True)
        model.fit(df)

        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)

        predicted_price = forecast.iloc[-1]['yhat']
        stats.loc[ticker, "7-Day Predicted Price"] = predicted_price

    except Exception as e:
        st.warning(f"Prophet prediction failed for {ticker}: {e}")

# Display stats table
st.dataframe(stats.style.format({
    "Start Price": "${:,.2f}",
    "End Price": "${:,.2f}",
    "Total Return (%)": "{:.2f}%",
    "Annualized Return (%)": "{:.2f}%",
    "Volatility (Std Dev of Daily Returns)": "{:.4f}",
    "7-Day Predicted Price": "${:,.2f}"
}))

# Optional: Show raw data
if st.checkbox("Show raw data"):
    st.write(data)
