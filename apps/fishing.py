import streamlit as st
from streamlit_autorefresh import st_autorefresh
import time

st.set_page_config(
    page_title="URL Auto-Refresher",
    page_icon="🔄",
    layout="wide"
)

st.title("🔄 Auto-Refreshing Web Viewer")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Target URL input
    target_url = st.text_input(
        "Enter Target URL:",
        value="https://stockanalysis.com/markets/premarket/",
        help="Make sure the site allows iframe embedding (some sites block it via X-Frame-Options)."
    )
    
    # Refresh interval selection
    refresh_seconds = st.number_input(
        "Refresh Interval (seconds):",
        min_value=1,
        max_value=3600,
        value=10,
        step=1
    )
    
    # Toggle to start/stop refreshing
    auto_refresh_active = st.toggle("Enable Auto-Refresh", value=True)
    
    # Manual refresh button
    if st.button("Force Manual Refresh", use_container_width=True):
        st.rerun()

    # Frame height adjustment
    frame_height = st.slider("Frame Height (px):", min_value=300, max_value=1200, value=700, step=50)

# Auto-Refresh Logic
refresh_count = 0
if auto_refresh_active:
    # Convert seconds to milliseconds for streamlit-autorefresh
    refresh_count = st_autorefresh(
        interval=refresh_seconds * 1000,
        key="url_refresher_counter"
    )

# Display status metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Status", "Refreshing" if auto_refresh_active else "Paused")
with col2:
    st.metric("Interval", f"{refresh_seconds}s")
with col3:
    st.metric("Total Refreshes", refresh_count)

# HTML iFrame Renderer
if target_url:
    # Append a dynamic timestamp query parameter to bypass browser caching on refresh
    timestamp = int(time.time() * 1000)
    delimiter = "&" if "?" in target_url else "?"
    cache_busted_url = f"{target_url}{delimiter}_refresh={timestamp}"

    # Embed using HTML iframe
    st.components.v1.html(
        f"""
        <iframe 
            src="{cache_busted_url}" 
            width="100%" 
            height="{frame_height}px" 
            style="border:1px solid #e6e6e6; border-radius: 8px;"
            allowfullscreen>
        </iframe>
        """,
        height=frame_height + 20
    )
else:
    st.warning("Please enter a valid URL in the sidebar to begin.")
