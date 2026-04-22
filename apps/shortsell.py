import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Seller's Market Detector", layout="wide")

st.title("📉 Seller's Market Detector")

# -----------------------------
# Helper
# -----------------------------
def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

# -----------------------------
# Input
# -----------------------------
ticker = st.text_input("Enter Stock Ticker", "NVTS")

# -----------------------------
# Download data
# -----------------------------
data = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)

if data is None or data.empty:
    st.error("No data returned for this ticker.")
    st.stop()

# Ensure correct structure
data = data.copy()

# Keep only needed columns safely
if "Close" not in data.columns or "Volume" not in data.columns:
    st.error("Unexpected data format from yfinance.")
    st.stop()

data = data[["Close", "Volume"]].copy()

# -----------------------------
# Indicators
# -----------------------------
data["MA10"] = data["Close"].rolling(10).mean()
data["MA20"] = data["Close"].rolling(20).mean()

# RSI
delta = data["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
data["RSI"] = 100 - (100 / (1 + rs))

# Volume average
data["VolAvg"] = data["Volume"].rolling(10).mean()

# -----------------------------
# Clean data (safe version)
# -----------------------------
data = data.dropna(subset=["Close"])

if len(data) < 30:
    st.warning("Not enough data for indicators yet.")
    st.stop()

# Ensure indicator columns exist
for col in ["MA10", "MA20", "RSI", "VolAvg"]:
    if col not in data.columns:
        data[col] = np.nan

# -----------------------------
# Latest rows
# -----------------------------
latest = data.iloc[-1]
prev = data.iloc[-2]

close = safe_float(latest["Close"])
ma20 = safe_float(latest["MA20"])
rsi = safe_float(latest["RSI"])
vol = safe_float(latest["Volume"])
volavg = safe_float(latest["VolAvg"])
prev_close = safe_float(prev["Close"])

# -----------------------------
# Scoring system
# -----------------------------
score = 0
signals = []

# 1. Trend breakdown
if not np.isnan(close) and not np.isnan(ma20) and close < ma20:
    score += 1
    signals.append("Below 20-day MA")

# 2. RSI weakening
if not np.isnan(rsi) and rsi < 50:
    score += 1
    signals.append("RSI below 50")

# 3. Volume selling
if (
    not np.isnan(vol)
    and not np.isnan(volavg)
    and not np.isnan(prev_close)
):
    if vol > volavg and close < prev_close:
        score += 1
        signals.append("High-volume selling")

# 4. Failing highs
recent_high_series = data["Close"].tail(10).dropna()
recent_high = safe_float(recent_high_series.max())

if (
    not np.isnan(close)
    and not np.isnan(recent_high)
    and close < recent_high * 0.95
):
    score += 1
    signals.append("Failing to make new highs")

# 5. Strong red day
price_change = (close - prev_close) / prev_close if prev_close else 0

if price_change < -0.03:
    score += 1
    signals.append("Strong red day (>3%)")

# -----------------------------
# Output
# -----------------------------
st.subheader("📊 Market Regime")

if score >= 4:
    st.error("🔴 SELLER'S MARKET")
elif score >= 2:
    st.warning("🟡 TRANSITION / NEUTRAL")
else:
    st.success("🟢 BUYER'S MARKET")

st.write(f"Score: {score}/5")

# -----------------------------
# Signals
# -----------------------------
st.subheader("Triggered Signals")

if signals:
    for s in signals:
        st.write(f"• {s}")
else:
    st.write("No bearish signals detected.")

# -----------------------------
# Chart (SAFE VERSION - FIXES KEYERROR)
# -----------------------------
st.subheader("📈 Price Chart")

plot_cols = ["Close", "MA10", "MA20"]
plot_cols = [c for c in plot_cols if c in data.columns]

plot_df = data[plot_cols].copy()

if plot_df["Close"].dropna().empty:
    st.warning("No chart data available.")
else:
    st.line_chart(plot_df)

# -----------------------------
# Debug (optional)
# -----------------------------
with st.expander("🔍 Debug Data"):
    st.write("Latest row:")
    st.write(latest)
    st.write("Previous row:")
    st.write(prev)
    st.write("Recent high:")
    st.write(recent_high)
