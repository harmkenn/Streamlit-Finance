from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

# Set Streamlit layout configuration
st.set_page_config(page_title="Webull Market Gainers", layout="wide")

# Webull Top Gainers API Endpoint
WEBULL_GAINERS_URL = "https://quoteapi.webullfintech.com/api/quote/v2/rank/topGainers"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@st.cache_data(ttl=30)  # Refresh cache every 30 seconds
def fetch_webull_gainers(region_id: int = 6, limit: int = 50) -> pd.DataFrame | None:
    """Fetches top gainers directly from Webull's API.

    Region 6 corresponds to US Markets.
    """
    params = {
        "regionId": region_id,
        "userRegionId": region_id,
        "pageSize": limit,
    }
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    try:
        response = requests.get(WEBULL_GAINERS_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Webull API usually wraps items under a data key
        ticker_list = data if isinstance(data, list) else data.get("data", [])

        parsed_rows = []
        for item in ticker_list:
            ticker = item.get("symbol") or item.get("ticker", {}).get("symbol", "")
            name = item.get("name") or item.get("ticker", {}).get("name", "")
            price = float(item.get("close", 0) or item.get("price", 0))
            change_ratio = float(item.get("changeRatio", 0)) * 100  # Convert decimal to %
            volume = float(item.get("volume", 0))

            parsed_rows.append({
                "Ticker": ticker,
                "Name": name,
                "Price ($)": price,
                "Change (%)": change_ratio,
                "Volume": volume,
                "URL": f"https://www.webull.com/quote/us/ticker/{ticker.lower()}",
            })

        return pd.DataFrame(parsed_rows)

    except (requests.RequestException, KeyError, ValueError) as err:
        st.error(f"Failed to fetch data from Webull API: {err}")
        return None


def apply_filters(
    df: pd.DataFrame, min_price: float = 0.80, min_volume: float = 1_000, top_n: int = 10
) -> pd.DataFrame:
    """Filters data frame by minimum price and volume requirements."""
    if df.empty:
        return df

    filtered = df[(df["Price ($)"] >= min_price) & (df["Volume"] >= min_volume)]
    return filtered.head(top_n).copy()


def color_rows(row, yellow_thresh=5.0, orange_thresh=10.0, red_thresh=20.0):
    """Applies conditional background coloring based on Change (%) threshold."""
    val = row["Change (%)"]
    if val >= red_thresh:
        return ["background-color: #b22222; color: white; font-weight: bold;"] * len(row)
    elif val >= orange_thresh:
        return ["background-color: #ff7f0e; color: white; font-weight: bold;"] * len(row)
    elif val >= yellow_thresh:
        return ["background-color: #f4d03f; color: #17221b; font-weight: bold;"] * len(row)
    return [""] * len(row)


# --- Streamlit Dashboard Interface ---
st.title("📈 Webull Top Daily Gainers")

# Sidebar Controls
st.sidebar.header("Filters & Thresholds")
min_price = st.sidebar.number_input("Min Price ($)", value=0.80, step=0.10)
min_vol = st.sidebar.number_input("Min Volume", value=1000, step=1000)

st.sidebar.subheader("Highlight Thresholds (%)")
yellow_t = st.sidebar.number_input("Yellow Threshold (%)", value=5.0)
orange_t = st.sidebar.number_input("Orange Threshold (%)", value=10.0)
red_t = st.sidebar.number_input("Red Threshold (%)", value=20.0)

if st.button("Refresh Data"):
    st.cache_data.clear()

df_raw = fetch_webull_gainers()

if df_raw is not None and not df_raw.empty:
    df_filtered = apply_filters(df_raw, min_price=min_price, min_volume=min_vol, top_n=10)

    if df_filtered.empty:
        st.warning("No stocks matched the current filter criteria.")
    else:
        # Format links for Streamlit LinkColumn or markdown rendering
        df_display = df_filtered.copy()

        # Apply Pandas Styling for colors
        styled_df = df_display.style.apply(
            color_rows,
            yellow_thresh=yellow_t,
            orange_thresh=orange_t,
            red_thresh=red_t,
            axis=1,
        ).format({
            "Price ($)": "${:.2f}",
            "Change (%)": "+{:.2f}%",
            "Volume": "{:,.0f}",
        })

        # Render styled dataframe in Streamlit
        st.dataframe(
            styled_df,
            column_config={
                "URL": st.column_config.LinkColumn(
                    "Webull Link", display_text="Open on Webull"
                )
            },
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("Unable to retrieve market data.")
