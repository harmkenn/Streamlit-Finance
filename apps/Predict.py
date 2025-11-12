import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Nasdaq Prediction App",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Nasdaq Prediction App")
st.markdown("""
This app analyzes the Nasdaq Composite index using technical indicators, macroeconomic data, market sentiment, and external events to provide insights into potential market movement for the next 5 days.
""")

# Sidebar for user input
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Enter Ticker Symbol:", "^IXIC", help="Use ^IXIC for Nasdaq Composite")
    analysis_period = st.number_input("Analysis Period (Days):", min_value=5, max_value=60, value=30)
    risk_tolerance = st.selectbox("Risk Tolerance Level:", ["Low", "Medium", "High"])
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Enter the ticker symbol (e.g., ^IXIC for Nasdaq Composite).
    2. Specify the analysis period (e.g., last 30 days).
    3. Choose your risk tolerance level.
    4. Click 'Analyze Market' to get insights.
    """)

# Analyze button
if st.button("🚀 Analyze Market"):
    if not ticker:
        st.error("⚠️ Please enter a ticker symbol.")
    else:
        with st.spinner("🔄 Fetching data and analyzing..."):
            try:
                # Fetch historical data using yfinance
                stock = yf.Ticker(ticker)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=analysis_period)
                data = stock.history(start=start_date, end=end_date)

                if data.empty:
                    st.error("❌ No data found for the given ticker symbol. Please try another.")
                else:
                    # Calculate technical indicators
                    current_price = data['Close'][-1]
                    avg_price = data['Close'].mean()
                    high_price = data['High'].max()
                    low_price = data['Low'].min()

                    # RSI calculation
                    delta = data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))

                    # Moving averages
                    data['MA_10'] = data['Close'].rolling(window=10).mean()
                    data['MA_50'] = data['Close'].rolling(window=50).mean()

                    # Display results
                    st.success(f"✅ Analysis complete for {ticker.upper()}!")
                    st.markdown(f"### Current Price: ${current_price:.2f}")
                    st.markdown(f"### Key Metrics:")
                    st.markdown(f"- **Average Price (Last {analysis_period} Days)**: ${avg_price:.2f}")
                    st.markdown(f"- **High Price**: ${high_price:.2f}")
                    st.markdown(f"- **Low Price**: ${low_price:.2f}")
                    st.markdown(f"- **RSI**: {rsi[-1]:.2f} (Overbought > 70, Oversold < 30)")

                    # Plot historical prices and moving averages
                    st.markdown("### Price Trend:")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['MA_10'], mode='lines', name='10-Day MA'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], mode='lines', name='50-Day MA'))
                    fig.update_layout(title="Nasdaq Price Trend", xaxis_title="Date", yaxis_title="Price (USD)")
                    st.plotly_chart(fig)

                    # Fetch real macroeconomic data using FRED API
                    st.markdown("### Macroeconomic Data:")
                    st.write("Fetching economic data...")

                    # FRED API key (use Streamlit Secrets for security)
                    API_KEY = st.secrets["api_keys"]["fred_api_key"]

                    # FRED API URL for Inflation Rate (CPIAUCSL - Consumer Price Index for All Urban Consumers)
                    inflation_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={API_KEY}&file_type=json"

                    # Fetch inflation rate data
                    response = requests.get(inflation_url)
                    if response.status_code == 200:
                        inflation_data = response.json()
                        latest_observation = inflation_data["observations"][-1]
                        inflation_rate = float(latest_observation["value"])/100
                        inflation_date = latest_observation["date"]
                        st.markdown(f"- **Inflation Rate**: {inflation_rate:.2f}% (as of {inflation_date})")
                    else:
                        st.error("❌ Failed to fetch inflation rate data.")

                    # Example prediction logic (dummy data for demonstration)
                    st.markdown("### Prediction for the Next 5 Days:")
                    st.write("Based on the analysis:")
                    st.markdown("""
                    - **Bullish Indicators**: RSI is neutral, moving averages show upward momentum, and inflation data is favorable.
                    - **Bearish Risks**: Watch for potential geopolitical developments or unexpected economic data.
                    - **Overall Sentiment**: Moderate bullish outlook for the next 5 days.
                    """)

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit, FRED API, yFinance, and Plotly | Powered by Python</p>
</div>
""", unsafe_allow_html=True)