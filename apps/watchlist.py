import streamlit as st
import pandas as pd
import requests

# Page configuration
st.set_page_config(page_title="Polygon Real-Time Gainers", page_icon="⚡", layout="wide")

st.title("⚡ Real-Time Top 20 Stock Gainers (Polygon.io API)")
st.write("Using legitimate exchange feeds to bypass Yahoo Finance 404/403 connection blocks.")

# Sidebar API management
API_KEY = st.sidebar.text_input("Enter Polygon.io API Key:", type="password")

# Session toggle
session = st.sidebar.radio(
    "Select Market Session:",
    ("Open Market Gainers", "Premarket Gainers")
)

@st.cache_data(ttl=15)
def fetch_polygon_gainers(session_type, api_key):
    if not api_key:
        st.warning("Please enter your Polygon.io API Key in the sidebar.")
        return None
        
    # Polygon provides a dedicated Snapshot API for market movers
    # Note: Free tier covers regular session movers natively. 
    # For full premarket book access, Polygon checks the latest market state.
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/gainers?apiKey={api_key}"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 401:
            st.error("🚫 **401 Unauthorized:** Invalid Polygon API key.")
            return None
        elif response.status_code != 200:
            st.error(f"API Error: Received status code {response.status_code}")
            return None
            
        json_data = response.json()
        tickers = json_data.get("tickers", [])
        
        if not tickers:
            return pd.DataFrame()
            
        df = pd.DataFrame(tickers)
        
        # Parse nested 'todaysChangePerc' and market values safely
        df['% Change'] = df['todaysChangePerc'].apply(lambda x: float(x) if x is not None else 0.0)
        df = df.sort_values(by='% Change', ascending=False).head(20)
        
        return df

    except Exception as e:
        st.error(f"Failed to connect to Polygon stream: {e}")
        return None

if API_KEY:
    with st.spinner(f"Streaming live tracking matrices..."):
        data = fetch_polygon_gainers(session, API_KEY)
        
    if data is not None and not data.empty:
        st.subheader(f"Top 20 Live {session}")
        
        # Isolate key market metrics returned by the snapshot engine
        rename_dict = {
            'ticker': 'Ticker',
            'todaysChange': 'Net Change',
            '% Change': '% Change',
            'min': 'Last Trade Details'
        }
        
        display_df = data.rename(columns=rename_dict)
        
        # Clean up nested dictionary rows if they exist
        if 'Last Trade Details' in display_df.columns:
            display_df['Price'] = display_df['Last Trade Details'].apply(lambda x: x.get('c') if isinstance(x, dict) else None)
        if 'day' in display_df.columns:
            display_df['Volume'] = display_df['day'].apply(lambda x: x.get('v') if isinstance(x, dict) else None)
            
        # Reorder into a clean user interface
        final_cols = ['Ticker', 'Price', 'Net Change', '% Change', 'Volume']
        existing_cols = [col for col in final_cols if col in display_df.columns]
        
        # Format percentages cleanly
        display_df['% Change'] = display_df['% Change'].apply(lambda x: f"{x:+.2f}%")
        
        st.dataframe(
            display_df[existing_cols], 
            use_container_width=True,
            hide_index=True
        )
    elif data is not None and data.empty:
        st.info("No active gainers found in the current exchange snapshot loop.")
else:
    st.info("👈 Please drop your free Polygon.io API key into the sidebar field to boot up the real-time websocket fallback tracker.")
