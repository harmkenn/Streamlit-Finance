import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

st.title("SOXL OHLC Plot (Last 5 Years)")

today = date.today()
five_years_ago = today - timedelta(days=5*365)

try:
    soxl = yf.Ticker("SOXL")
    data = soxl.history(start=five_years_ago, end=today)

    if data.empty:
        st.error("No data found for SOXL. Please check the ticker symbol.")
    else:
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                            open=data['Open'],
                                            high=data['High'],
                                            low=data['Low'],
                                            close=data['Close'])])

        fig.update_layout(title="SOXL OHLC Chart",
                          xaxis_title="Date",
                          yaxis_title="Price",
                          xaxis_rangeslider_visible=False) # Hide range slider for cleaner look

        st.plotly_chart(fig)


        # Add some basic stats below the chart
        st.subheader("Key Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", data['Close'][-1].round(2))
        with col2:
            st.metric("52 Week High", data['High'].rolling(window=252).max()[-1].round(2))
        with col3:
            st.metric("52 Week Low", data['Low'].rolling(window=252).min()[-1].round(2))
        
        # Add a table with the raw data. It is collapsed by default.
        with st.expander("Show raw data"):
            st.dataframe(data)

except Exception as e:
    st.error(f"An error occurred: {e}")