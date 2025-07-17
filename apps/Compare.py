import streamlit as st
from yahooquery import Ticker
import pandas as pd
from prophet import Prophet
import numpy as np
import datetime

st.set_page_config(layout="wide")
st.title("📈 Stock Comparison Tool (with 7-Day Prophet Forecast)")

# Split and clean tickers from session state
ticker_list = [ticker.strip().upper() for ticker in st.session_state.get("tickers", "").split(',') if ticker.strip()]

# Date input and ticker selection
col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("Start date", datetime.date(2024, 6, 1))
with col2:
    end_date = st.date_input("End date", datetime.date.today())
with col3:
    selected_tickers = st.multiselect(
        "Select Tickers to Compare",
        options=ticker_list,
        default=ticker_list[:3]
    )

# Validate inputs
if start_date >= end_date:
    st.error("End date must be after start date.")
    st.stop()

if not selected_tickers:
    st.error("Please enter at least one valid ticker symbol.")
    st.stop()

# ✅ Load data using yahooquery and pivot to adjusted close
@st.cache_data
def load_data(tickers, start, end):
    tq = Ticker(tickers)
    history = tq.history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))

    if history.empty:
        raise ValueError("No data returned.")

    history = history.reset_index()
    history = history[['symbol', 'date', 'adjclose']]

    if history.empty or 'adjclose' not in history.columns:
        raise ValueError("Adjusted close prices not found.")

    data = history.pivot(index='date', columns='symbol', values='adjclose')
    data.index.name = 'Date'
    data.columns.name = None

    return data.dropna(how='all')

# Load and process data
data = load_data(selected_tickers, start_date, end_date)

# 📊 Plot actual adjusted close prices
st.subheader("Stock Prices (Adjusted Close)")
st.line_chart(data)

# 📋 Summary statistics
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

# 🔮 Prophet-based 7-day prediction
for ticker in data.columns:
    try:
        df = data[[ticker]].dropna().reset_index()
        df.columns = ['ds', 'y']
        model = Prophet(daily_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)
        stats.loc[ticker, "7-Day Predicted Price"] = forecast.iloc[-1]['yhat']
    except Exception as e:
        st.warning(f"Prophet prediction failed for {ticker}: {e}")

# 📑 Display summary table
st.dataframe(stats.style.format({
    "Start Price": "${:,.2f}",
    "End Price": "${:,.2f}",
    "Total Return (%)": "{:.2f}%",
    "Annualized Return (%)": "{:.2f}%",
    "Volatility (Std Dev of Daily Returns)": "{:.4f}",
    "7-Day Predicted Price": "${:,.2f}"
}))

# 🔍 Show raw data toggle
if st.checkbox("Show raw data"):
    st.write(data)
