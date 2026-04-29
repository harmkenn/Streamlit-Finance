import streamlit as st
import importlib.util
import os
import yfinance as yf

# --- Helper functions ---
TICKERS_FILE = "tickers.txt"

def load_tickers():
    if os.path.exists(TICKERS_FILE):
        with open(TICKERS_FILE, "r") as f:
            return f.read().strip()
    # default fallback
    return "TQQQ,UPRO,UDOW,BNO,NVTS,MXL,^VIX"

def save_tickers(tickers_str):
    with open(TICKERS_FILE, "w") as f:
        f.write(tickers_str.strip())

# --- Streamlit setup ---
st.set_page_config(layout="wide", page_title="Finance")

sub_app_names = {
    'Intraday.py': 'Intraday',
    'BuySellHold.py': 'Buy Sell or Hold',
    'Compare.py': 'Compare',
    'Whatif.py': 'Last Year',
    'shortsell.py': 'Trend',
    'watchlist.py': 'Watchlist',
}

sub_apps_folder = 'apps'
sub_apps = [f for f in os.listdir(sub_apps_folder) if f.endswith('.py')]

selected_sub_app_name = st.sidebar.radio('Select a sub-app', list(sub_app_names.values()))
selected_sub_app = [k for k, v in sub_app_names.items() if v == selected_sub_app_name][0]

# --- Editable ticker list ---
if "tickers" not in st.session_state:
    st.session_state["tickers"] = load_tickers()

tickers_list = st.sidebar.text_area(
    "Enter comma-separated stock tickers",
    value=st.session_state["tickers"],
    height=100
)

# Save edits when user changes
if tickers_list != st.session_state["tickers"]:
    st.session_state["tickers"] = tickers_list
    save_tickers(tickers_list)

# --- Run selected sub-app ---
if selected_sub_app:
    spec = importlib.util.spec_from_file_location(selected_sub_app, os.path.join(sub_apps_folder, selected_sub_app))
    sub_app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub_app_module)
