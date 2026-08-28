import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from streamlit_autorefresh import st_autorefresh

# Page Setup
st.set_page_config(page_title="Premarket Top 10 Scraper", page_icon="📊", layout="wide")

st.title("📊 Premarket Top 10 Gainers")

# Auto-refresh timer every 60 seconds (60,000 milliseconds)
count = st_autorefresh(interval=60 * 1000, key="premarket_scraper")

# Sidebar Controls
st.sidebar.header("Settings")
st.sidebar.write(f"**Auto-Refresh Interval:** 60 Seconds")
st.sidebar.write(f"**Total Refreshes:** {count}")

if st.sidebar.button("Force Manual Refresh"):
    st.rerun()

@st.cache_data(ttl=55)
def scrape_premarket_top10():
    url = "https://stockanalysis.com/markets/premarket/"
    
    # Custom headers to emulate standard browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")
        
        if not table:
            return None, "Table element not found on the target page."

        # Extract column headers
        headers_list = [th.text.strip() for th in table.find_all("th")]
        
        # Extract rows
        rows = []
        for tr in table.find_all("tr")[1:11]:  # Get top 10 items
            cells = [td.text.strip() for td in tr.find_all("td")]
            if cells:
                rows.append(cells)
                
        if not rows:
            return None, "No data rows found in the table."
            
        # Format into Pandas DataFrame
        df = pd.DataFrame(rows, columns=headers_list if headers_list else None)
        return df, None

    except Exception as e:
        return None, str(e)

# Fetch data
df, error = scrape_premarket_top10()

if error:
    st.error(f"Error scraping data: {error}")
else:
    st.success(f"Last updated successfully! Auto-refreshing in 60s...")
    st.dataframe(df, use_container_width=True, hide_index=True)
