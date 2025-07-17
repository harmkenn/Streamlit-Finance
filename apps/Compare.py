import streamlit as st
from yahooquery import Ticker
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.title("📈 Normalized Closing Prices - MSTY vs MAIN (Last 12 Months)")

# Define tickers
ticker_list = [t.strip().upper() for t in st.session_state.get("tickers", "").split(",") if t.strip()]
#ticker_list = ["MSTY", "MAIN"]
tickers = st.multiselect("Select Tickers to Compare",options=ticker_list,default=ticker_list[:3])


# Date range: last 12 months
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# Fetch data
ticker_data = Ticker(tickers)
history = ticker_data.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

# Ensure valid data
if isinstance(history, pd.DataFrame) and not history.empty:
    # Pivot the DataFrame to get closing prices for each symbol in columns
    history = history.reset_index()
    close_prices = history.pivot(index='date', columns='symbol', values='close')
    
    # Drop any rows with missing values (dates when either ticker is missing)
    close_prices.dropna(inplace=True)

    # Normalize prices to 100 based on first value
    normalized = close_prices / close_prices.iloc[0] * 100

    # Plotting
    fig = go.Figure()
    for symbol in tickers:
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized[symbol],
            mode='lines',
            name=f"{symbol} (Normalized)",
            line=dict(width=2)
        ))
    fig.update_layout(
        title="Normalized Closing Prices: MSTY vs MAIN",
        xaxis_title="Date",
        yaxis_title="Normalized Price (Starting at 100)",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show table
    st.subheader("📄 Normalized Prices Table")
    st.dataframe(normalized.reset_index(), use_container_width=True)
else:
    st.error("⚠️ Could not retrieve historical data for MSTY and MAIN.")
