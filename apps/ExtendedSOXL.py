
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import datetime as dt

st.title("SOXL Extended Hours Price Tracker")

# Date inputs
c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input("Start Date", value=dt.date.today() - dt.timedelta(days=30))
with c2:
    end_date = st.date_input("End Date", value=dt.date.today())

# Fetch SOXL data with extended hours
data = yf.download("SOXL", start=start_date, end=end_date, prepost=True)

# Calculate moving averages
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# Create figure
fig = go.Figure()

# Add candlestick
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name='OHLC'
))

# Add moving averages
fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='20 Day MA', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='50 Day MA', line=dict(color='blue')))

# Update layout
fig.update_layout(
    title='SOXL Price with Extended Hours',
    yaxis_title='Price',
    xaxis_title='Date',
    xaxis_rangeslider_visible=False
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Display current prices
current_data = yf.download("SOXL", period='1d', interval='1m', prepost=True)
if not current_data.empty:
    last_price = current_data['Close'][-1]
    last_time = current_data.index[-1]
    
    st.metric("Current Price", f"${last_price:.2f}")
    st.write(f"Last Updated: {last_time}")

# Display raw data
with st.expander("Show Raw Data"):
    st.dataframe(data.sort_index(ascending=False))
