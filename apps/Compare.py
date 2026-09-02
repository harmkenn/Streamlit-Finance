import streamlit as st
from yahooquery import Ticker
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.title("📈 Normalized Closing Prices")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_comparison_history(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Fetch and normalize yahooquery's multi-ticker response shape."""
    history = Ticker(list(tickers)).history(start=start, end=end)
    if not isinstance(history, pd.DataFrame) or history.empty:
        return pd.DataFrame()

    history = history.reset_index()
    if "symbol" not in history.columns or "date" not in history.columns:
        return pd.DataFrame()

    return history.pivot_table(
        index="date",
        columns="symbol",
        values="close",
        aggfunc="last",
    ).sort_index()


# --- Parse tickers from shared session state ---
raw_tickers = st.session_state.get("user_tickers", "")
ticker_list = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

if not ticker_list:
    st.warning("⚠️ No tickers found in the sidebar. Please enter comma-separated stock tickers in the sidebar.")
    st.stop()

# --- Controls layout ---
col1, col2 = st.columns([3, 1])

with col1:
    tickers = st.multiselect(
        "Select Tickers to Compare",
        options=ticker_list,
        default=ticker_list[:min(7, len(ticker_list))]
    )

with col2:
    time_frame = st.selectbox(
        "Time Horizon",
        options=["6 Months", "1 Year", "2 Years", "YTD"],
        index=1  # Defaults to 1 Year
    )

if not tickers:
    st.info("Select at least one ticker above to display the comparison chart.")
    st.stop()

# Determine start date based on selection
end_date = datetime.today()
if time_frame == "6 Months":
    start_date = end_date - timedelta(days=180)
elif time_frame == "1 Year":
    start_date = end_date - timedelta(days=365)
elif time_frame == "2 Years":
    start_date = end_date - timedelta(days=730)
else:  # YTD
    start_date = datetime(end_date.year, 1, 1)

# --- Fetch Data ---
try:
    close_prices = fetch_comparison_history(
        tuple(tickers),
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

    if not close_prices.empty:
        # Clean missing values safely (forward-fill missing dates rather than dropping entire rows)
        close_prices = close_prices.ffill().bfill()

        # Normalize each column to 100 based on its first valid closing price
        normalized = pd.DataFrame(index=close_prices.index)
        for col in close_prices.columns:
            first_valid = close_prices[col].first_valid_index()
            if first_valid is not None and close_prices.loc[first_valid, col] != 0:
                normalized[col] = (close_prices[col] / close_prices.loc[first_valid, col]) * 100

        # --- Plotting ---
        fig = go.Figure()
        for symbol in normalized.columns:
            fig.add_trace(go.Scatter(
                x=normalized.index,
                y=normalized[symbol],
                mode='lines',
                name=f"{symbol}",
                hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Normalized Value: %{y:.2f}<extra></extra>"
            ))

        fig.update_layout(
            title=f"Normalized Performance ({time_frame}): {', '.join(normalized.columns)}",
            xaxis_title="Date",
            yaxis_title="Normalized Price (Base = 100)",
            template="plotly_white",
            height=500,
            hovermode="x unified"
        )

        st.plotly_chart(fig, width="stretch")

        # --- Table View ---
        st.subheader("📄 Normalized Prices Table")
        st.dataframe(normalized.reset_index(), width="stretch")

    else:
        st.error(f"⚠️ Could not retrieve historical data for: {', '.join(tickers)}. Please verify the tickers.")

except Exception as e:
    st.error(f"Error fetching comparison data: {e}")
