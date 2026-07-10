import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Premarket Gainers Streamer", page_icon="🚀", layout="wide")

st.title("🚀 Top 20 Live Premarket & Open Market Gainers")
st.write("Using the native `yf.screen()` wrapper to handle Yahoo session authentication tokens safely.")

# Sidebar session selector
session = st.sidebar.radio(
    "Select Market Session:",
    ("Premarket Gainers", "Open Market Gainers")
)

@st.cache_data(ttl=30)
def fetch_movers_with_screen(session_type):
    try:
        # Map the dropdown selection to Yahoo's native predefined screener IDs
        scr_id = "premarket_gainers" if session_type == "Premarket Gainers" else "day_gainers"
        
        # yf.screen() manages cookies and crumbs natively to bypass 404 and 403 errors
        response = yf.screen(scr_id)
        
        # Extract quotes array from the returned dictionary payload
        quotes = response.get("quotes", [])
        
        if not quotes:
            return pd.DataFrame()
            
        return pd.DataFrame(quotes).head(20)
        
    except Exception as e:
        st.error(f"Screener Error: {e}")
        return pd.DataFrame()

with st.spinner(f"Requesting {session} data from Yahoo..."):
    raw_data = fetch_movers_with_screen(session)

if not raw_data.empty:
    st.subheader(f"Top 20 Live {session}")
    
    # Map raw response fields to human-readable headers
    rename_map = {
        'symbol': 'Ticker',
        'shortName': 'Company Name',
        'preMarketPrice': 'Premarket Price',
        'regularMarketPrice': 'Market Price',
        'preMarketChangePercent': '% Change',
        'regularMarketChangePercent': '% Change (Intraday)',
        'regularMarketVolume': 'Volume'
    }
    
    processed_df = raw_data.rename(columns=rename_map)
    
    # Keep columns that are actively returned in the payload
    cols_to_display = [col for col in rename_map.values() if col in processed_df.columns]
    
    # Simple formatting for percentage columns
    for pct_col in ['% Change', '% Change (Intraday)']:
        if pct_col in processed_df.columns:
            processed_df[pct_col] = processed_df[pct_col].apply(
                lambda val: f"{val:+.2f}%" if isinstance(val, (int, float)) else str(val)
            )
            
    st.dataframe(
        processed_df[cols_to_display],
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No active data returned. Note that premarket fields are only populated heavily before standard exchange hours (4:00 AM - 9:30 AM EST).")
