import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Seller's Market Detector", layout="wide")

st.title("📉 Seller's Market Detector")

# -------------------------
# Helpers
# -------------------------
def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

# -------------------------
# Input
# -------------------------
ticker = st.text_input("Enter Stock Ticker", "NVTS")

# -------------------------
# Data Download
# -------------------------
data = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)

if data.empty:
    st.error("No data found for this ticker.")
    st.stop()

# Keep only needed columns safely
data = data.loc[:, ["Close", "Volume"]].copy()

# -------------------------
# Indicators
# -------------------------
data["MA10"] = data["Close"].rolling(10).mean()
data["MA20"] = data["Close"].rolling(20).mean()

# RSI
delta = data["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
data["RSI"] = 100 - (100 / (1 + rs))

# Volume avg
data["VolAvg"] = data["Volume"].rolling(10).mean()

# Clean NaNs
data = data.dropna()

if len(data) < 30:
    st.warning("Not enough data to compute indicators.")
    st.stop()

# -------------------------
# Latest values
# -------------------------
latest = data.iloc[-1].copy()
prev = data.iloc[-2].copy()

close = safe_float(latest["Close"])
ma20 = safe_float(latest["MA20"])
rsi = safe_float(latest["RSI"])
vol = safe_float(latest["Volume"])
volavg = safe_float(latest["VolAvg"])
prev_close = safe_float(prev["Close"])

# -------------------------
# Scoring system
# -------------------------
score = 0
signals = []

# 1. Trend break
if not np.isnan(close) and not np.isnan(ma20) and close < ma20:
    score += 1
    signals.append("Below 20-day MA")

# 2. Momentum weakening
if not np.isnan(rsi) and rsi < 50:
    score += 1
    signals.append("RSI below 50")

# 3. Volume selling pressure
if (
    not np.isnan(vol)
    and not np.isnan(volavg)
    and not np.isnan(prev_close)
):
    if vol > volavg and close < prev_close:
        score += 1
        signals.append("High-volume selling")

# 4. Failure to make new highs
recent_high = float(data["Close"].tail(10).max())
if close < recent_high * 0.95:
    score += 1
    signals.append("Failing to make new highs")

# 5. Strong red day
price_change = (close - prev_close) / prev_close if prev_close else 0
if price_change < -0.03:
    score += 1
    signals.append("Strong red day (>3%)")

# -------------------------
# Output
# -------------------------
st.subheader("📊 Market Regime")

if score >= 4:
    st.error("🔴 SELLER'S MARKET")
elif score >= 2:
    st.warning("🟡 NEUTRAL / TRANSITION")
else:
    st.success("🟢 BUYER'S MARKET")

st.write(f"Score: {score}/5")

# -------------------------
# Signals
# -------------------------
st.subheader("Triggered Signals")

if signals:
    for s in signals:
        st.write(f"• {s}")
else:
    st.write("No bearish signals detected.")

# -------------------------
# Chart
# -------------------------
st.subheader("📈 Price Chart")
st.line_chart(data[["Close", "MA10", "MA20"]])

# -------------------------
# Debug panel
# -------------------------
with st.expander("🔍 Debug Data"):
    st.write("Latest row:")
    st.write(latest)
    st.write("Previous row:")
    st.write(prev)
