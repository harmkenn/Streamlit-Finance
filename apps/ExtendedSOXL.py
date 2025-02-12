
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

def fetch_data():
    ticker = "SOXL"
    # Get today's date and yesterday's date
    end = datetime.now()
    start = end - timedelta(days=1)
    
    # Download minute data including pre/post market
    data = yf.download(ticker, start=start, end=end, interval="1m", prepost=True)
    data.reset_index(inplace=True)
    return data

def main():
    st.title("SOXL ETF Price (Including Pre/Post Market)")
    
    data = fetch_data()
    
    if not data.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=data['Datetime'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='SOXL'
        )])
        
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display current price and daily stats
        current_price = data['Close'].iloc[-1]
        daily_high = data['High'].max()
        daily_low = data['Low'].min()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            st.metric("Daily High", f"${daily_high:.2f}")
        with col3:
            st.metric("Daily Low", f"${daily_low:.2f}")
            
        with st.expander("Show Raw Data"):
            st.dataframe(data.sort_values('Datetime', ascending=False)) 

if __name__ == "__main__":
    main()
