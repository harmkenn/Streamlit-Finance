import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz

# Page Setup
st.set_page_config(page_title="Stock Market Top 10 Tracker", page_icon="📈", layout="wide")

st.title("📈 Stock Analysis - Top 10 Data Tracker")

# Target Page Selection
PAGES = {
    "Premarket Movers": "https://stockanalysis.com/markets/premarket/",
    "Top Daily Gainers": "https://stockanalysis.com/markets/gainers/"
}

# Determine default page based on US Eastern Time (09:30 - 16:00 ET)
def get_default_page_index():
    eastern_tz = pytz.timezone("US/Eastern")
    now_et = datetime.now(eastern_tz)
    
    # Check if weekday (0 = Monday, 4 = Friday)
    is_weekday = now_et.weekday() < 5
    
    # Check if time is between 09:30 AM and 04:00 PM Eastern
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=20, minute=0, second=0, microsecond=0)
    
    if is_weekday and (market_open <= now_et < market_close):
        return 1  # Index 1 = "Top Daily Gainers"
    else:
        return 0  # Index 0 = "Premarket Movers"

default_index = get_default_page_index()

# Sidebar Controls
st.sidebar.header("⚙️ Settings")

selected_page_name = st.sidebar.selectbox(
    "Select Market View:",
    options=list(PAGES.keys()),
    index=default_index
)

refresh_seconds = st.sidebar.number_input(
    "Refresh Interval (seconds):",
    min_value=10,
    max_value=600,
    value=30,
    step=5
)

yellow_threshold = st.sidebar.number_input(
    "Yellow Threshold (%):",
    min_value=0.0,
    max_value=1000.0,
    value=100.0,
    step=5.0
)

green_threshold = st.sidebar.number_input(
    "Green Threshold (%):",
    min_value=0.0,
    max_value=1000.0,
    value=150.0,
    step=5.0
)

if green_threshold < yellow_threshold:
    green_threshold = yellow_threshold

# Auto-refresh timer
refresh_count = st_autorefresh(
    interval=refresh_seconds * 1000, 
    key=f"stock_tracker_{selected_page_name.lower().replace(' ', '_')}"
)

st.sidebar.write(f"**Total Refreshes:** {refresh_count}")

if st.sidebar.button("Force Manual Refresh", width="stretch"):
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
            return None

        # Extract table headers
        headers_list = [th.text.strip() for th in table.find_all("th")]
        
        # Extract rows (fetch extra rows to preserve 10 items after filtering)
        rows = []
        for tr in table.find_all("tr")[1:25]:
            cells = [td.text.strip() for td in tr.find_all("td")]
            if cells:
                rows.append(cells)
                
        if not rows:
            return None
            
        df = pd.DataFrame(rows, columns=headers_list if headers_list else None)
        return df

    except Exception:
        return None

# Row Highlighting Helper

def highlight_high_gainers(row, yellow_threshold, green_threshold):
    change_col = next((col for col in row.index if "%" in col or "Change" in col), None)

    if change_col and pd.notna(row[change_col]):
        try:
            clean_val = str(row[change_col]).replace("%", "").replace("+", "").replace(",", "").strip()
            val = float(clean_val)

            if val >= green_threshold:
                return ["background-color: #1b5e20; color: #ffffff; font-weight: bold;"] * len(row)
            elif val >= yellow_threshold:
                return ["background-color: #f4d03f; color: #000000; font-weight: bold;"] * len(row)
        except ValueError:
            pass

    return [""] * len(row)


def parse_volume(val):
    try:
        value = str(val).replace(",", "").strip().upper()
        multiplier = 1
        if value.endswith("K"):
            multiplier = 1_000
            value = value[:-1]
        elif value.endswith("M"):
            multiplier = 1_000_000
            value = value[:-1]
        elif value.endswith("B"):
            multiplier = 1_000_000_000
            value = value[:-1]
        return float(value) * multiplier
    except (TypeError, ValueError):
        return 0.0

# Fetch Data
target_url = PAGES[selected_page_name]
st.subheader(f"Showing Top 10: {selected_page_name}")

df = scrape_stock_top10(target_url)

if df is not None:
    # Identify price column name
    price_col = next((col for col in df.columns if "price" in col.lower()), None)
    
    # Filter out tickers with price < $0.90
    if price_col:
        def parse_price(val):
            try:
                return float(str(val).replace("$", "").replace(",", "").strip())
            except ValueError:
                return 0.0
                
        df = df[df[price_col].apply(parse_price) >= 0.80].copy()

    volume_col = next((col for col in df.columns if "volume" in col.lower()), None)
    if volume_col:
        df = df[df[volume_col].apply(parse_volume) >= 500_000].copy()

    # Limit to top 10 remaining rows after filtering
    df = df.head(10)

    # Identify symbol column name (usually 'Symbol' or 'Ticker')
    symbol_col = next((col for col in df.columns if "symbol" in col.lower() or "ticker" in col.lower()), df.columns[0])
    
    # Generate full target URLs for the LinkColumn
    df["Symbol_URL"] = df[symbol_col].apply(lambda s: f"https://stockanalysis.com/stocks/{str(s).lower()}/")
    
    # Reorder columns so Symbol_URL is rendered in place of the original Symbol column
    cols = list(df.columns)
    cols.remove("Symbol_URL")
    sym_idx = cols.index(symbol_col)
    cols[sym_idx] = "Symbol_URL"
    df = df[cols]
    
    # Apply styling rules
    styled_df = df.style.apply(
        highlight_high_gainers,
        yellow_threshold=yellow_threshold,
        green_threshold=green_threshold,
        axis=1,
    )

    # Render table with LinkColumn configuration
    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Symbol_URL": st.column_config.LinkColumn(
                label=symbol_col,
                display_text=r"https://stockanalysis\.com/stocks/(.*?)/"
            )
        }
    )
