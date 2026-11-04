from __future__ import annotations

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# Set Streamlit layout configuration
st.set_page_config(page_title="Webull Market Gainers", layout="wide")

# StockAnalysis Top Gainers page
GAINERS_URL = "https://stockanalysis.com/markets/gainers/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@st.cache_data(ttl=30)  # Refresh cache every 30 seconds
def fetch_market_gainers() -> pd.DataFrame | None:
    """Fetches top gainers from StockAnalysis.

    The Webull quote API used previously is no longer available.
    """
    try:
        response = requests.get(
            GAINERS_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        response.raise_for_status()
        return parse_stockanalysis_gainers(response.text)

    except (requests.RequestException, KeyError, ValueError) as err:
        st.error(f"Failed to fetch data from StockAnalysis: {err}")
        return None


def parse_stockanalysis_gainers(html: str) -> pd.DataFrame:
    """Parses the gainers table into the dashboard's market-data columns."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("Gainers table was not found in the response.")

    headers = [cell.get_text(" ", strip=True) for cell in table.find_all("th")]
    required_headers = ["Symbol", "Company Name", "% Change", "Stock Price", "Volume"]
    missing_headers = [header for header in required_headers if header not in headers]
    if missing_headers:
        raise ValueError(f"Gainers table is missing columns: {', '.join(missing_headers)}")

    parsed_rows = []
    for row in table.find_all("tr")[1:]:
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
        if not cells:
            continue

        item = dict(zip(headers, cells))
        ticker = item["Symbol"]
        parsed_rows.append({
            "Ticker": ticker,
            "Name": item["Company Name"],
            "Price ($)": float(item["Stock Price"].replace(",", "").replace("$", "")),
            "Change (%)": float(item["% Change"].replace(",", "").replace("%", "").replace("+", "")),
            "Volume": float(item["Volume"].replace(",", "")),
            "URL": f"https://stockanalysis.com/stocks/{ticker.lower()}/",
        })

    if not parsed_rows:
        raise ValueError("Gainers table contains no market data.")
    return pd.DataFrame(parsed_rows)


def apply_filters(
    df: pd.DataFrame, min_price: float = 0.80, min_volume: float = 1_000, top_n: int = 10
) -> pd.DataFrame:
    """Filters data frame by minimum price and volume requirements."""
    if df.empty:
        return df

    filtered = df[(df["Price ($)"] >= min_price) & (df["Volume"] >= min_volume)]
    return filtered.head(top_n).copy()


def color_rows(row, yellow_thresh=100.0, orange_thresh=150.0, red_thresh=200.0):
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
yellow_t = st.sidebar.number_input("Yellow Threshold (%)", value=100.0)
orange_t = st.sidebar.number_input("Orange Threshold (%)", value=150.0)
red_t = st.sidebar.number_input("Red Threshold (%)", value=200.0)

if st.button("Refresh Data"):
    st.cache_data.clear()

df_raw = fetch_market_gainers()

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
                    "StockAnalysis Link", display_text="Open on StockAnalysis"
                )
            },
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("Unable to retrieve market data.")
