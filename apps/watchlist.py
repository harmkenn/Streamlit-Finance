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
def _to_float(val):
    if pd.isna(val) or val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def format_mcap(val):
    numeric_val = _to_float(val)
    if numeric_val is None:
        return "N/A"
    if numeric_val >= 1e12:
        return f"${numeric_val / 1e12:.2f}T"
    if numeric_val >= 1e9:
        return f"${numeric_val / 1e9:.2f}B"
    if numeric_val >= 1e6:
        return f"${numeric_val / 1e6:.2f}M"
    return f"${numeric_val:,.0f}"


def format_volume(val):
    numeric_val = _to_float(val)
    if numeric_val is None:
        return "N/A"
    if numeric_val >= 1e6:
        return f"{numeric_val / 1e6:.2f}M"
    if numeric_val >= 1e3:
        return f"{numeric_val / 1e3:.1f}K"
    return f"{numeric_val:,.0f}"


def calculate_position_profit(price, avg_cost, shares_owned):
    price_val = _to_float(price)
    avg_cost_val = _to_float(avg_cost)
    shares_val = _to_float(shares_owned)
    if price_val is None or avg_cost_val is None or shares_val is None:
        return 0.0
    return (price_val - avg_cost_val) * shares_val


def format_money(val):
    numeric_val = _to_float(val)
    if numeric_val is None:
        return "N/A"
    return f"${numeric_val:,.2f}"


def format_signed_number(val):
    numeric_val = _to_float(val)
    if numeric_val is None:
        return "N/A"
    return f"{numeric_val:+.2f}"


def format_signed_percent(val):
    numeric_val = _to_float(val)
    if numeric_val is None:
        return "N/A"
    return f"{numeric_val:+.2f}%"


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

refresh_sec = st.sidebar.slider("Auto-Refresh Interval (s):", min_value=5, max_value=120, value=30)
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

overview_df = df[["Symbol_URL", "Price", "Change", "% Change"]].copy()
overview_df["Shares Owned"] = 0.0
overview_df["Avg Cost"] = 0.0
overview_df["Profit"] = 0.0

edited_overview = st.data_editor(
    overview_df,
    use_container_width=True,
    hide_index=True,
    disabled=["Symbol_URL", "Price", "Change", "% Change", "Profit"],
    column_config={
        "Symbol_URL": st.column_config.LinkColumn(
            label="Symbol", display_text=r"https://stockanalysis\.com/stocks/(.*?)/"
        ),
        "Shares Owned": st.column_config.NumberColumn(
            min_value=0.0,
            step=0.01,
            format="%.2f",
        ),
        "Avg Cost": st.column_config.NumberColumn(
            min_value=0.0,
            step=0.01,
            format="$%.2f",
        ),
        "Profit": st.column_config.NumberColumn(
            label="Profit",
            format="" + "$%.2f",
        ),
    },
)

edited_overview["Profit"] = edited_overview.apply(
    lambda row: calculate_position_profit(row["Price"], row["Avg Cost"], row["Shares Owned"]),
    axis=1,
)

styled_overview = edited_overview.style.map(color_changes, subset=["Change", "% Change", "Profit"]).format(
    {
        "Price": format_money,
        "Change": format_signed_number,
        "% Change": format_signed_percent,
        "Avg Cost": format_money,
        "Profit": format_money,
    }
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
