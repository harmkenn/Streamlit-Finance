import importlib.util
import os
import streamlit as st
import yfinance as yf

# --- Helper functions ---
TICKERS_FILE = "tickers.txt"


def load_tickers():
    """Read default tickers from disk without modifying them."""
    if os.path.exists(TICKERS_FILE):
        with open(TICKERS_FILE, "r") as f:
            return f.read().strip()
    # default fallback
    return "TQQQ,UPRO,UDOW,BNO,^VIX,SPHY,MXL"


# --- Streamlit setup ---
st.set_page_config(layout="wide", page_title="Finance")

# Note: Dictionary keys must be unique. Fixed 'shortsell.py' duplicate key below.
sub_app_names = {
    "Intraday.py": "Intraday",
    "BuySellHold.py": "Buy Sell or Hold",
    "Compare.py": "Compare",
    "Whatif.py": "Last Year",
    "watchlist.py": "Watchlist",
    "shortsell.py": "Short Screener",
    "fishing.py": "Fishing",
}

sub_apps_folder = "apps"

selected_sub_app_name = st.sidebar.radio(
    "Select a sub-app", list(sub_app_names.values())
)
selected_sub_app = [
    k for k, v in sub_app_names.items() if v == selected_sub_app_name
][0]

# --- Instance-Unique Ticker List ---
# Initialize session state from tickers.txt once per session/user
if "user_tickers" not in st.session_state:
    st.session_state["user_tickers"] = load_tickers()

# Text area bound directly to session state
tickers_list = st.sidebar.text_area(
    "Enter comma-separated stock tickers",
    key="user_tickers",
    height=100,
)

# --- Run selected sub-app ---
if selected_sub_app:
    spec = importlib.util.spec_from_file_location(
        selected_sub_app, os.path.join(sub_apps_folder, selected_sub_app)
    )
    sub_app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub_app_module)
