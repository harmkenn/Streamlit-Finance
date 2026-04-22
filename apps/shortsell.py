import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from transformers import pipeline

st.set_page_config(page_title="Seller's Market Cockpit", layout="wide")

st.title("📉 Seller's Market Cockpit")

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

if "Close" not in data.columns or "Volume" not in data.columns:
    st.error("Unexpected data format from yfinance.")
    st.stop()

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

if close < ma20:
    score += 1
    signals.append("Below 20-day MA")

if rsi < 50:
    score += 1
    signals.append("RSI below 50")

if vol > volavg and close < prev_close:
    score += 1
    signals.append("High-volume selling")

recent_high = safe_float(data["Close"].tail(10).max())
if close < recent_high * 0.95:
    score += 1
    signals.append("Failing to make new highs")

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

st.subheader("Triggered Signals")
if signals:
    for s in signals:
        st.write(f"• {s}")
else:
    st.write("No bearish signals detected.")

# -----------------------------
# Candlestick Chart
# -----------------------------
st.subheader("📈 Price Chart")
fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Price"
)])
if "MA10" in data.columns:
    fig.add_trace(go.Scatter(x=data.index, y=data['MA10'], line=dict(color='blue'), name="MA10"))
if "MA20" in data.columns:
    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], line=dict(color='red'), name="MA20"))
st.plotly_chart(fig, use_container_width=True)

# RSI
st.subheader("📉 RSI Indicator")
if "RSI" in data.columns and not data["RSI"].dropna().empty:
    st.line_chart(data[["RSI"]])
else:
    st.warning("RSI data not available yet.")

# Volume
st.subheader("📊 Volume")
if "Volume" in data.columns and not data["Volume"].dropna().empty:
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume"))
    if "VolAvg" in data.columns and not data["VolAvg"].dropna().empty:
        vol_fig.add_trace(go.Scatter(x=data.index, y=data['VolAvg'], line=dict(color='orange'), name="10-day Avg"))
    st.plotly_chart(vol_fig, use_container_width=True)
else:
    st.warning("Volume data not available.")

# -----------------------------
# Benchmark Comparison
# -----------------------------
benchmark = yf.download("SPY", period="6mo", interval="1d", auto_adjust=True)
benchmark['Return'] = benchmark['Close'].pct_change().cumsum()
data['Return'] = data['Close'].pct_change().cumsum()

comp_fig = go.Figure()
comp_fig.add_trace(go.Scatter(x=data.index, y=data['Return'], name=ticker))
comp_fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark['Return'], name="SPY"))
st.subheader("📊 Relative Performance")
st.plotly_chart(comp_fig, use_container_width=True)

# -----------------------------
# Sentiment Analysis
# -----------------------------
st.subheader("📰 Headline Sentiment")
try:
    news = yf.Ticker(ticker).news
    if news:
        sentiment_model = pipeline("sentiment-analysis")
        headline_sentiments = []
        for item in news[:5]:
            result = sentiment_model(item['title'])[0]
            headline_sentiments.append((item['title'], result['label'], result['score']))
        for title, label, score in headline_sentiments:
            st.write(f"**{title}** → {label} ({score:.2f})")

        pos_count = sum(1 for _, label, _ in headline_sentiments if label == "POSITIVE")
        neg_count = sum(1 for _, label, _ in headline_sentiments if label == "NEGATIVE")

        if pos_count > neg_count:
            st.success(f"Overall sentiment: Bullish ({pos_count} positive vs {neg_count} negative)")
        elif neg_count > pos_count:
            st.error(f"Overall sentiment: Bearish ({neg_count} negative vs {pos_count} positive)")
        else:
            st.warning("Overall sentiment: Neutral")
    else:
        st.write("No headlines available.")
except Exception as e:
    st.write("Sentiment analysis unavailable:", e)

# -----------------------------
# Natural-Language Summary
# -----------------------------
summary = []
if score >= 4:
    summary.append(f"{ticker} is showing strong bearish momentum.")
elif score >= 2:
    summary.append(f"{ticker} is in a neutral/transition phase.")
else:
    summary.append(f"{ticker} is showing bullish signs.")

if signals:
    summary.append("Key signals: " + ", ".join(signals))

st.subheader("📝 Summary")
st.write(" ".join(summary))
