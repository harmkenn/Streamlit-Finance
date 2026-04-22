import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Seller's Market Detector", layout="wide")

st.title("📉 Seller's Market Detector")

# --- Input ---
ticker = st.text_input("Enter Stock Ticker", "NVTS")

# --- Download Data ---
data = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)

# --- Safety Check ---
if data.empty:
    st.error("No data found. Try another ticker.")
    st.stop()

# Keep only needed columns
data = data[['Close', 'Volume']]

# --- Indicators ---
data['MA10'] = data['Close'].rolling(10).mean()
data['MA20'] = data['Close'].rolling(20).mean()

# RSI
delta = data['Close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Volume average
data['VolAvg'] = data['Volume'].rolling(10).mean()

# Drop NaNs (important)
data = data.dropna()

# Ensure enough data
if len(data) < 30:
    st.warning("Not enough data yet to calculate indicators.")
    st.stop()

# Latest rows
latest = data.iloc[-1].copy()
prev = data.iloc[-2].copy()

# --- Scoring System ---
score = 0
signals = []

# 1. Trend breakdown
if float(latest['Close']) < float(latest['MA20']):
    score += 1
    signals.append("Below 20-day MA")

# 2. Momentum weakening
if float(latest['RSI']) < 50:
    score += 1
    signals.append("RSI below 50")

# 3. High-volume selling
if float(latest['Volume']) > float(latest['VolAvg']) and float(latest['Close']) < float(prev['Close']):
    score += 1
    signals.append("High-volume selling")

# 4. Failing to make new highs
recent_high = float(data['Close'].tail(10).max())
if float(latest['Close']) < recent_high * 0.95:
    score += 1
    signals.append("Failing to make new highs")

# 5. Strong red day
price_change = (float(latest['Close']) - float(prev['Close'])) / float(prev['Close'])
if price_change < -0.03:
    score += 1
    signals.append("Strong red day (>3% drop)")

# --- Output ---
st.subheader("📊 Market Regime")

if score >= 4:
    st.error("🔴 SELLER'S MARKET")
elif score >= 2:
    st.warning("🟡 NEUTRAL / TRANSITION")
else:
    st.success("🟢 BUYER'S MARKET")

st.write(f"Score: {score}/5")

# --- Signals ---
st.subheader("Triggered Signals")
if signals:
    for s in signals:
        st.write(f"- {s}")
else:
    st.write("No bearish signals triggered.")

# --- Chart ---
st.subheader("📈 Price Chart")
st.line_chart(data[['Close', 'MA10', 'MA20']])

# --- Debug Panel (optional but useful) ---
with st.expander("🔍 Debug Info"):
    st.write("Latest Row:")
    st.write(latest)
    st.write("Previous Row:")
    st.write(prev)
    st.write("Recent High (10 days):", recent_high)
