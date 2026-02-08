import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

st.title("📈 TQQQ 7-Day Price Prediction")

ticker = st.sidebar.text_input("Enter ticker:", "TQQQ")
start_date = st.sidebar.date_input("Start date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.today())

# Download data
st.write(f"Fetching data for {ticker}...")
data = yf.download(ticker, start=start_date, end=end_date)

# Use 'Adj Close' if available, otherwise fallback to 'Close'
if "Adj Close" in data.columns:
    series = data["Adj Close"]
else:
    series = data["Close"]

st.line_chart(series, width='stretch')

# Fit ARIMA model
st.write("Training ARIMA model...")
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit()

# Forecast next 7 days
forecast = model_fit.forecast(steps=7)
forecast_dates = pd.date_range(datetime.today(), periods=7)

forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted Price": forecast})
st.write("### 📊 7-Day Forecast")
st.dataframe(forecast_df)

# Plot forecast
fig, ax = plt.subplots()
ax.plot(series.index, series, label="Historical")
ax.plot(forecast_dates, forecast, label="Forecast", color="red")
ax.set_title(f"{ticker} Price Forecast (Next 7 Days)")
ax.legend()
st.pyplot(fig)

st.write("⚠️ Disclaimer: This is a simple statistical forecast. TQQQ is highly volatile and leveraged. Predictions should not be used as financial advice.")
