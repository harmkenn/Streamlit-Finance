import streamlit as st
import pandas as pd
import requests

# Page Layout configuration
st.set_page_config(page_title="Premarket Gainers Streamer", page_icon="🚀", layout="wide")

st.title("🚀 Top 20 Live Premarket & Open Market Gainers")
st.write("Fetching direct API JSON payloads from Yahoo Finance endpoints.")

# Session selector toggle
session = st.sidebar.radio(
    "Select Market Session:",
    ("Premarket Gainers", "Open Market Gainers")
)

@st.cache_data(ttl=15)  # Cache for 15 seconds to stream data effectively
def get_yahoo_movers(session_type):
    # Map the dropdown choices to the correct Yahoo Finance API predefined keys
    scr_id = "premarket_gainers" if session_type == "Premarket Gainers" else "day_gainers"
    
    # Direct Yahoo backend querying string endpoint
    url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds={scr_id}&count=20"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            st.error(f"Yahoo API rejected connection (Status Code: {response.status_code})")
            return pd.DataFrame()
            
        data = response.json()
        quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
        
        if not quotes:
            return pd.DataFrame()
            
        # Parse relevant ticker keys into a Pandas DataFrame
        df = pd.DataFrame(quotes)
        return df
        
    except Exception as e:
        st.error(f"Connection error: {e}")
        return pd.DataFrame()

# Execution stream
with st.spinner(f"Connecting to Yahoo servers for {session}..."):
    raw_df = get_yahoo_movers(session)

if not raw_df.empty:
    st.subheader(f"Top 20 Live {session}")
    
    # Define cleaner display dictionaries depending on what session is chosen
    if session == "Premarket Gainers":
        rename_map = {
            'symbol': 'Ticker',
            'shortName': 'Company Name',
            'preMarketPrice': 'Premarket Price',
            'preMarketChange': 'Net Change',
            'preMarketChangePercent': '% Change',
            'regularMarketVolume': 'Volume'
        }
    else:
        rename_map = {
            'symbol': 'Ticker',
            'shortName': 'Company Name',
            'regularMarketPrice': 'Market Price',
            'regularMarketChange': 'Net Change',
            'regularMarketChangePercent': '% Change',
            'regularMarketVolume': 'Volume'
        }
        
    # Remap and isolate existing column groups safely
    processed_df = raw_df.rename(columns=rename_map)
    cols_to_render = [col for col in rename_map.values() if col in processed_df.columns]
    display_df = processed_df[cols_to_render].copy()
    
    # Explicit conversion & clear styling for the % Change metric
    if '% Change' in display_df.columns:
        display_df['% Change'] = display_df['% Change'].apply(
            lambda val: f"{val:+.2f}%" if isinstance(val, (int, float)) else str(val)
        )
        
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No movers returned. **Note:** Premarket calculations are heavily populated by exchange order books between 4:00 AM and 9:30 AM EST.")
