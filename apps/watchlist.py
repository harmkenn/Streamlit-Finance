import streamlit as st
import pandas as pd
import requests

# Set up page config
st.set_page_config(page_title="Real-Time Stock Gainers", page_icon="📈", layout="wide")

st.title("📈 Real-Time Top 20 Stock Gainers (API Driven)")
st.write("Fetching live, official exchange data cleanly via REST endpoints.")

# Safely manage your API key
# For production, use Streamlit Secrets. For local testing, paste it here.
API_KEY = st.sidebar.text_input("Enter FMP API Key:", type="password")

# Session Selector Toggle
session = st.sidebar.radio(
    "Select Market Session:",
    ("Open Market Gainers", "Premarket Gainers")
)

@st.cache_data(ttl=15)  # Cache for 15 seconds to stay updated but respect API limits
def fetch_realtime_gainers(session_type, api_key):
    if not api_key:
        st.warning("Please enter your Financial Modeling Prep API Key in the sidebar.")
        return None
        
    # Dynamically select the official API endpoint based on your toggle choice
    if session_type == "Premarket Gainers":
        url = f"https://financialmodelingprep.com/api/v4/premarket-gainers?apikey={api_key}"
    else:
        url = f"https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey={api_key}"
        
    try:
        response = requests.get(url)
        if response.status_code == 200:
            json_data = response.json()
            df = pd.DataFrame(json_data)
            
            if df.empty:
                return df
                
            # Clean and filter to top 20
            # Common columns returned: symbol, name, price, change, changesPercentage
            df = df.head(20)
            return df
        else:
            st.error(f"API Error: Received status code {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to the data stream: {e}")
        return None

# Execution logic
if API_KEY:
    with st.spinner(f"Streaming live {session} details..."):
        data = fetch_realtime_gainers(session, API_KEY)
        
    if data is not None and not data.empty:
        st.subheader(f"Top 20 Live {session}")
        
        # Format columns for professional view if they exist in the dataset
        rename_dict = {
            'symbol': 'Ticker',
            'name': 'Company Name',
            'price': 'Current Price',
            'change': 'Net Change',
            'changesPercentage': '% Change'
        }
        
        # Filter and arrange available columns cleanly
        display_df = data.rename(columns=rename_dict)
        existing_cols = [col for col in rename_dict.values() if col in display_df.columns]
        
        st.dataframe(
            display_df[existing_cols], 
            use_container_width=True,
            hide_index=True
        )
    elif data is not None and data.empty:
        st.info("No active gainers found for this session at the moment.")
else:
    st.info("👈 Enter your API key in the sidebar to initiate the real-time feed.")
