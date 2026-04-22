import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.title("📉 Seller's Market Detector")

ticker = st.text_input("Enter Stock Ticker", "NVTS")

data = yf.download(ticker, period="6mo", interval="1d")

# --- Indicators ---
data['MA10'] = data['Close'].rolling(10).mean()
data['MA20'] = data['Close'].rolling(20).mean()

# RSI
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Volume average
data['VolAvg'] = data['Volume'].rolling(10).mean()

latest = data.iloc[-1]
prev = data.iloc[-2]

score = 0
signals = []

# --- 1. Trend breakdown ---
if latest['Close'] < latest['MA20']:
    score += 1
    signals.append("Below 20-day MA")

# --- 2. Momentum weakening ---
if latest['RSI'] < 50:
    score += 1
    signals.append("RSI below 50")

# --- 3. Heavy selling volume ---
if latest['Volume'] > latest['VolAvg'] and latest['Close'] < prev['Close']:
    score += 1
    signals.append("High-volume selling")

# --- 4. Lower high (simple version) ---
recent_high = data['Close'][-10:].max()
if latest['Close'] < recent_high * 0.95:
    score += 1
    signals.append("Failing to make new highs")

# --- 5. Sharp drop ---
if (latest['Close'] - prev['Close']) / prev['Close'] < -0.03:
    score += 1
    signals.append("Strong red day")

# --- Decision ---
st.subheader("📊 Market Regime")

if score >= 4:
    st.error("🔴 SELLER'S MARKET")
elif score >= 2:
    st.warning("🟡 NEUTRAL / TRANSITION")
else:
    st.success("🟢 BUYER'S MARKET")

st.write(f"Score: {score}/5")

st.subheader("Triggered Signals")
for s in signals:
    st.write(f"- {s}")

# --- Chart ---
st.subheader("Price Chart")
st.line_chart(data[['Close', 'MA10', 'MA20']])
