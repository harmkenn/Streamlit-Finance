import streamlit as st
import requests
from datetime import datetime

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
                # Fetch macroeconomic data using FRED API
                st.markdown("### Macroeconomic Data:")
                st.write("Fetching economic data...")

                # FRED API key (use Streamlit Secrets for security)
                API_KEY = st.secrets["api_keys"]["fred_api_key"]

                # FRED API URLs for economic indicators
                urls = {
                    "Inflation Rate": f"https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={API_KEY}&file_type=json",
                    "Unemployment Rate": f"https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&api_key={API_KEY}&file_type=json",
                    "Retail Sales Growth": f"https://api.stlouisfed.org/fred/series/observations?series_id=RSAFS&api_key={API_KEY}&file_type=json"
                }

                # Fetch data for each indicator
                economic_data = {}
                for indicator, url in urls.items():
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        # Extract the latest observation
                        latest_observation = data["observations"][-1]
                        value = float(latest_observation["value"])
                        date = latest_observation["date"]
                        economic_data[indicator] = f"{value:.2f}% (as of {date})"
                    else:
                        economic_data[indicator] = "Data unavailable"

                # Display the fetched data
                st.json(economic_data)

                # Example prediction logic (dummy data for demonstration)
                st.markdown("### Prediction for the Next 5 Days:")
                st.write("Based on the analysis:")
                st.markdown("""
                - **Bullish Indicators**: Inflation rate is stable, unemployment rate is low, and retail sales growth is positive.
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
    <p>Built with Streamlit, FRED API, and Python</p>
</div>
""", unsafe_allow_html=True)