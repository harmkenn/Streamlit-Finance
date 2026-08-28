import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from streamlit_autorefresh import st_autorefresh

# Page Setup
st.set_page_config(page_title="Stock Market Top 10 Tracker", page_icon="📈", layout="wide")

st.title("📈 Stock Analysis - Top 10 Data Tracker")

# Target Page Selection
PAGES = {
    "Premarket Movers": "https://stockanalysis.com/markets/premarket/",
    "Top Daily Gainers": "https://stockanalysis.com/markets/gainers/"
}

# Sidebar Controls
st.sidebar.header("⚙️ Settings")

selected_page_name = st.sidebar.selectbox(
    "Select Market View:",
    options=list(PAGES.keys())
)

refresh_seconds = st.sidebar.number_input(
    "Refresh Interval (seconds):",
    min_value=10,
    max_value=600,
    value=60,
    step=5
)

threshold_pct = st.sidebar.number_input(
    "Highlight Threshold (%):",
    min_value=0.0,
    max_value=1000.0,
    value=150.0,
    step=10.0
)

# Auto-refresh timer
refresh_count = st_autorefresh(
    interval=refresh_seconds * 1000, 
    key=f"stock_tracker_{selected_page_name.lower().replace(' ', '_')}"
)

st.sidebar.write(f"**Total Refreshes:** {refresh_count}")

if st.sidebar.button("Force Manual Refresh", use_container_width=True):
    st.rerun()

# Data Scraping Function
@st.cache_data(ttl=refresh_seconds - 5)
def scrape_stock_top10(url):
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

        # Extract table headers
        headers_list = [th.text.strip() for th in table.find_all("th")]
        
        # Extract top 10 rows
        rows = []
        for tr in table.find_all("tr")[1:11]:
            cells = [td.text.strip() for td in tr.find_all("td")]
            if cells:
                rows.append(cells)
                
        if not rows:
            return None, "No data rows found in the table."
            
        df = pd.DataFrame(rows, columns=headers_list if headers_list else None)
        return df, None

    except Exception as e:
        return None, str(e)

# Row Highlighting Helper
def highlight_high_gainers(row, threshold):
    """
    Highlights the entire row with a dark green background and bold white text 
    so the words remain easily readable in light and dark themes.
    """
    change_col = next((col for col in row.index if "%" in col or "Change" in col), None)
    
    if change_col and pd.notna(row[change_col]):
        try:
            clean_val = str(row[change_col]).replace("%", "").replace("+", "").replace(",", "").strip()
            val = float(clean_val)
            
            if val > threshold:
                # Dark green background (#1b5e20) with crisp white bold text
                return ["background-color: #1b5e20; color: #ffffff; font-weight: bold;"] * len(row)
        except ValueError:
            pass
            
    return [""] * len(row)

# Fetch and Render Data
target_url = PAGES[selected_page_name]
st.subheader(f"Showing Top 10: {selected_page_name}")

df, error = scrape_stock_top10(target_url)

if error:
    st.error(f"Error fetching data: {error}")
else:
    st.success(f"Updated successfully! Auto-refreshing every {refresh_seconds} seconds.")
    
    # Apply row highlighting via Pandas Styler
    styled_df = df.style.apply(highlight_high_gainers, threshold=threshold_pct, axis=1)
    
    # Display styled table
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
