import streamlit as st
from yahooquery import Ticker
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pytz

st.title("Intraday Stock Prices (Including Pre-market & After-hours)")

# User input for stock symbol
col1, col2, col3 = st.columns(3)
with col1:
    tickers_list = [t.strip().upper() for t in st.session_state.get("tickers", "").split(",") if t.strip()]
    ticker = st.selectbox("Select Stock Ticker", tickers_list) if tickers_list else st.text_input("Enter ticker").upper()

with col3:
    refresh_button = st.button("Refresh")

if ticker:
    try:
        tk = Ticker(ticker)

        # Fetch intraday data with 5-minute intervals (past 5 days) including pre/post market
        hist = tk.history(period='5d', interval='5m')
        
        if hist.empty:
            st.error(f"No data found for {ticker}. Please check the symbol and try again.")
        else:
            # Handle multiindex if multiple tickers, else single
            if isinstance(hist.index, pd.MultiIndex):
                data = hist.xs(ticker, level=0)
            else:
                data = hist
            
            # Yahooquery returns timestamps in UTC; convert to Eastern Time
            data = data.copy()
            data.index = pd.to_datetime(data.index).tz_localize('UTC').tz_convert('America/New_York')
            
            # Show current price (last close)
            latest_price = data['close'].iloc[-1]
            with col2:
                st.markdown(f"### Current Price: ${latest_price:.2f}")

            # Create plot with secondary y-axis for volume
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Price line (blue)
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['close'],
                mode='lines',
                name='Stock Price',
                line=dict(color='blue')
            ), secondary_y=False)

            # Volume bars (grey)
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker=dict(color='grey'),
                opacity=0.5
            ), secondary_y=True)

            # Layout updates
            fig.update_layout(
                title=f"{ticker} Intraday Prices (Including Pre-market & After-hours)",
                xaxis_title="Time",
                yaxis_title="Price",
                legend_title="Market Data",
                height=600,
                hovermode='x unified'
            )
            fig.update_yaxes(title_text="Volume", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)

            # Show raw data reversed so newest first
            st.subheader("Raw Data (most recent first)")
            st.dataframe(data[['close', 'volume']].iloc[::-1])

            if refresh_button:
                st.experimental_rerun()

    except Exception as e:
        st.error(f"Error fetching data: {e}")
else:
    st.info("Please enter or select a stock ticker symbol.")
