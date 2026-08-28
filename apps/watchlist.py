from datetime import datetime
import pandas as pd
import pytz
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Watchlist", page_icon="📈", layout="wide")


# --- Market Status Helper ---
def get_market_status():
    eastern_tz = pytz.timezone("US/Eastern")
    now_et = datetime.now(eastern_tz)
    is_weekday = now_et.weekday() < 5

    open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    if is_weekday and (open_time <= now_et < close_time):
        return "Market Open 🟢"
    elif is_weekday and now_et < open_time and now_et.hour >= 4:
        return "Pre-Market 🟡"
    elif is_weekday and now_et >= close_time and now_et.hour < 20:
        return "After-Hours 🌙"
    else:
        return "Market Closed 🔴"


# --- Fetch Data for Tickers ---
@st.cache_data(ttl=15)
def fetch_watchlist_data(tickers: list[str]) -> pd.DataFrame:
    rows = []

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info or {}

            price = info.get("last_price") or info.get("lastPrice")
            prev_close = info.get("previous_close") or info.get("previousClose")
            mcap = info.get("market_cap") or info.get("marketCap")
            volume = info.get("last_volume") or info.get("lastVolume")

            # Fallback for historical close if fast_info missing
            if price is None:
                hist = t.history(period="5d")
                if not hist.empty:
                    price = hist["Close"].iloc[-1]
                    prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else price
                    volume = hist["Volume"].iloc[-1]

            change = price - prev_close if price and prev_close else 0.0
            percent_change = (change / prev_close * 100) if prev_close else 0.0

            # Historical Changes for Performance
            hist_1m = t.history(period="1mo")
            chg_1m = (
                ((price - hist_1m["Close"].iloc[0]) / hist_1m["Close"].iloc[0] * 100)
                if not hist_1m.empty
                else 0.0
            )

            hist_ytd = t.history(period="ytd")
            chg_ytd = (
                ((price - hist_ytd["Close"].iloc[0]) / hist_ytd["Close"].iloc[0] * 100)
                if not hist_ytd.empty
                else 0.0
            )

            rows.append(
                {
                    "Symbol": ticker,
                    "Price": price,
                    "Change": change,
                    "% Change": percent_change,
                    "Volume": volume,
                    "Market Cap": mcap,
                    "1M %": chg_1m,
                    "YTD %": chg_ytd,
                    "52W High": info.get("year_high"),
                    "52W Low": info.get("year_low"),
                }
            )
        except Exception:
            rows.append(
                {
                    "Symbol": ticker,
                    "Price": None,
                    "Change": None,
                    "% Change": None,
                    "Volume": None,
                    "Market Cap": None,
                    "1M %": None,
                    "YTD %": None,
                    "52W High": None,
                    "52W Low": None,
                }
            )

    return pd.DataFrame(rows)


# --- Formatting Helpers ---
def format_mcap(val):
    if pd.isna(val) or val is None:
        return "N/A"
    if val >= 1e12:
        return f"${val / 1e12:.2f}T"
    if val >= 1e9:
        return f"${val / 1e9:.2f}B"
    if val >= 1e6:
        return f"${val / 1e6:.2f}M"
    return f"${val:,.0f}"


def format_volume(val):
    if pd.isna(val) or val is None:
        return "N/A"
    if val >= 1e6:
        return f"{val / 1e6:.2f}M"
    if val >= 1e3:
        return f"{val / 1e3:.1f}K"
    return f"{val:,.0f}"


def color_changes(val):
    if pd.isna(val) or val is None:
        return ""
    if val > 0:
        return "color: #00c853; font-weight: bold;"
    elif val < 0:
        return "color: #ff5252; font-weight: bold;"
    return ""


# --- Main Watchlist Application ---
st.title("📈 Watchlist")

# Extract tickers list from st.session_state
tickers_list = [
    t.strip().upper()
    for t in st.session_state.get("tickers", "").split(",")
    if t.strip()
]

# Sidebar Configuration
st.sidebar.header("⚙️ Controls")
st.sidebar.markdown(f"**Market Status:** `{get_market_status()}`")

# Allow single ticker focus or full-list view
selected_ticker = (
    st.sidebar.selectbox("Select Stock Ticker:", tickers_list)
    if tickers_list
    else st.sidebar.text_input("Enter ticker:").upper()
)

refresh_sec = st.sidebar.slider("Auto-Refresh Interval (s):", min_value=5, max_value=120, value=15)
st_autorefresh(interval=refresh_sec * 1000, key="watchlist_session_refresh")

if not tickers_list and not selected_ticker:
    st.info("No tickers found in session state. Please input a comma-separated list into `st.session_state['tickers']`.")
    st.stop()

# Determine final list of tickers to display in the grid
active_tickers = tickers_list if tickers_list else [selected_ticker]

# Fetch market metrics
df = fetch_watchlist_data(active_tickers)

# Stock Analysis Link Generation
df["Symbol_URL"] = df["Symbol"].apply(
    lambda s: f"https://stockanalysis.com/stocks/{str(s).replace('^', '').lower()}/"
)

# Stock Analysis Tabbed Layout
tab_overview, tab_performance, tab_fundamentals = st.tabs(
    ["📊 Overview", "🚀 Performance", "🏢 Fundamentals"]
)

with tab_overview:
    overview_df = df[["Symbol_URL", "Price", "Change", "% Change", "Volume", "Market Cap"]].copy()
    overview_df["Market Cap"] = overview_df["Market Cap"].apply(format_mcap)
    overview_df["Volume"] = overview_df["Volume"].apply(format_volume)

    styled_overview = overview_df.style.map(color_changes, subset=["Change", "% Change"]).format(
        {"Price": "${:.2f}", "Change": "{:+.2f}", "% Change": "{:+.2f}%"}
    )

    st.dataframe(
        styled_overview,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Symbol_URL": st.column_config.LinkColumn(
                label="Symbol", display_text=r"https://stockanalysis\.com/stocks/(.*?)/"
            )
        },
    )

with tab_performance:
    perf_df = df[["Symbol_URL", "Price", "% Change", "1M %", "YTD %"]].copy()

    styled_perf = perf_df.style.map(color_changes, subset=["% Change", "1M %", "YTD %"]).format(
        {"Price": "${:.2f}", "% Change": "{:+.2f}%", "1M %": "{:+.2f}%", "YTD %": "{:+.2f}%"}
    )

    st.dataframe(
        styled_perf,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Symbol_URL": st.column_config.LinkColumn(
                label="Symbol", display_text=r"https://stockanalysis\.com/stocks/(.*?)/"
            )
        },
    )

with tab_fundamentals:
    fund_df = df[["Symbol_URL", "Price", "52W High", "52W Low", "Market Cap"]].copy()
    fund_df["Market Cap"] = fund_df["Market Cap"].apply(format_mcap)

    st.dataframe(
        fund_df.style.format({"Price": "${:.2f}", "52W High": "${:.2f}", "52W Low": "${:.2f}"}),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Symbol_URL": st.column_config.LinkColumn(
                label="Symbol", display_text=r"https://stockanalysis\.com/stocks/(.*?)/"
            )
        },
    )
