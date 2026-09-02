from datetime import datetime
import pandas as pd
import pytz
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

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

            change = (price - prev_close) if price and prev_close else 0.0
            percent_change = (change / prev_close * 100) if prev_close else 0.0

            # Historical Changes for Performance
            hist_1m = t.history(period="1mo")
            chg_1m = (
                ((price - hist_1m["Close"].iloc[0]) / hist_1m["Close"].iloc[0] * 100)
                if not hist_1m.empty and hist_1m["Close"].iloc[0] != 0
                else 0.0
            )

            hist_ytd = t.history(period="ytd")
            chg_ytd = (
                ((price - hist_ytd["Close"].iloc[0]) / hist_ytd["Close"].iloc[0] * 100)
                if not hist_ytd.empty and hist_ytd["Close"].iloc[0] != 0
                else 0.0
            )

            rows.append(
                {
                    "Symbol": ticker,
                    "Price": float(price) if price else None,
                    "Change": float(change),
                    "% Change": float(percent_change),
                    "Volume": volume,
                    "Market Cap": mcap,
                    "1M %": float(chg_1m),
                    "YTD %": float(chg_ytd),
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


# --- Calculation Helper ---
def calculate_position_profit(price, avg_cost, shares_owned):
    if pd.isna(price) or pd.isna(avg_cost) or pd.isna(shares_owned):
        return 0.0
    return (float(price) - float(avg_cost)) * float(shares_owned)


# --- Main Watchlist Application ---
st.title("📈 Watchlist")

# Extract tickers list from st.session_state (using shared "user_tickers" key)
raw_tickers = st.session_state.get("user_tickers", "")
tickers_list = [
    t.strip().upper()
    for t in raw_tickers.split(",")
    if t.strip()
]

# Sidebar Controls
st.sidebar.header("⚙️ Controls")
st.sidebar.markdown(f"**Market Status:** `{get_market_status()}`")

refresh_sec = st.sidebar.slider("Auto-Refresh Interval (s):", min_value=5, max_value=120, value=30)
st_autorefresh(interval=refresh_sec * 1000, key="watchlist_session_refresh")

if not tickers_list:
    st.info("⚠️ No tickers found in session state. Please input a comma-separated list into the sidebar text area.")
    st.stop()

# Fetch market metrics
df = fetch_watchlist_data(tickers_list)

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
    key="watchlist_positions_editor",
    use_container_width=True,
    hide_index=True,
    disabled=["Symbol_URL", "Symbol", "Price", "Change", "% Change", "Profit", "1M %", "YTD %"],
    column_config={
        "Symbol_URL": st.column_config.LinkColumn(
            label="Symbol Link",
            display_text=r"https://stockanalysis\.com/stocks/(.*?)/"
        ),
        "Symbol": st.column_config.TextColumn(label="Ticker"),
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "Change": st.column_config.NumberColumn(format="%+.2f"),
        "% Change": st.column_config.NumberColumn(format="%+.2f%%"),
        "Shares Owned": st.column_config.NumberColumn(
            min_value=0.0,
            step=0.01,
            format="%.2f",
        ),
        "Avg Cost": st.column_config.NumberColumn(
            min_value=0.0,
            step=0.5,
            format="$%.2f",
        ),
        "Profit": st.column_config.NumberColumn(
            label="Unrealized Profit",
            format="$%.2f",
        ),
    },
)

edited_overview["Profit"] = edited_overview.apply(
    lambda row: calculate_position_profit(row["Price"], row["Avg Cost"], row["Shares Owned"]),
    axis=1,
)

# Save edited positions back to session state
for idx, row in edited_df.iterrows():
    symbol = row["Symbol"]
    st.session_state["portfolio_positions"][symbol] = {
        "shares": float(row["Shares Owned"]),
        "cost": float(row["Avg Cost"])
    }

# Summary KPIs
total_profit = edited_df["Profit"].sum()
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Watchlist Portfolio Profit/Loss", f"${total_profit:,.2f}", delta=f"{total_profit:,.2f}")
with col2:
    st.metric("Monitored Tickers", len(tickers_list))
