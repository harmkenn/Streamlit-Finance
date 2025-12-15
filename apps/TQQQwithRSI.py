import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.title("TQQQ Dabble Helper – Buy/Sell Zones for Today v4.4")

st.write(
    "This tool is **not financial advice**. It shows how someone *might* "
    "think about buy/sell zones for TQQQ using recent trend, RSI, and "
    "typical intraday behavior."
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Settings")

lookback_days = st.sidebar.slider("Lookback window (days)", 60, 1000, 500, 10)
rsi_period = st.sidebar.slider("RSI period", 5, 28, 14)
vol_lookback = st.sidebar.slider("Volatility lookback (days)", 10, 60, 20, 5)
trend_ma_len = st.sidebar.slider("QQQ trend MA length (days)", 20, 200, 50, 5)

# Base buy/sell ranges (in % from previous close)
buy_min_base = st.sidebar.slider("Base min buy dip (%)", 1.0, 10.0, 3.0, 0.5)
buy_max_base = st.sidebar.slider("Base max buy dip (%)", 2.0, 15.0, 6.5, 0.5)

sell_min_base = st.sidebar.slider("Base min sell pop (%)", 1.0, 10.0, 3.0, 0.5)
sell_max_base = st.sidebar.slider("Base max sell pop (%)", 2.0, 20.0, 6.5, 0.5)

st.sidebar.markdown("---")
st.sidebar.write("**Interpretation**")
st.sidebar.write(
    "- Buy zone is a *dip* below yesterday's close.\n"
    "- Sell zone is a *pop* above yesterday's close.\n"
    "- RSI and trend tweak these ranges."
)

# -----------------------------
# Helper functions
# -----------------------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_data(ticker: str, days: int) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end + timedelta(days=1), interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[df["Volume"] > 0]  # remove placeholder rows
    return df

# -----------------------------
# Load data
# -----------------------------
with st.spinner("Loading TQQQ and QQQ data..."):
    tqqq = fetch_data("TQQQ", lookback_days)
    qqq = fetch_data("QQQ", lookback_days)

if tqqq.empty or qqq.empty:
    st.error("Could not load data for TQQQ/QQQ.")
    st.stop()

tqqq = tqqq[["Open", "High", "Low", "Close", "Volume"]].astype(float)
qqq = qqq[["Close"]].astype(float)

# Compute RSI on TQQQ
tqqq["RSI"] = compute_rsi(tqqq["Close"], period=rsi_period)

# Compute daily returns and volatility proxy (TQQQ)
tqqq["Return"] = tqqq["Close"].pct_change()
tqqq["Volatility"] = tqqq["Return"].rolling(vol_lookback).std() * np.sqrt(252)

# Compute QQQ trend
qqq[f"MA{trend_ma_len}"] = qqq["Close"].rolling(trend_ma_len).mean()
qqq["TrendUp"] = qqq["Close"] > qqq[f"MA{trend_ma_len}"]

# Align latest dates
common_index = tqqq.index.intersection(qqq.index)
tqqq = tqqq.loc[common_index].copy()
qqq = qqq.loc[common_index].copy()

tqqq["QQQ_TrendUp"] = qqq["TrendUp"]

# Drop early NaNs
tqqq = tqqq.dropna()

if tqqq.empty:
    st.error("Not enough overlapping data after indicators.")
    st.stop()

# -----------------------------
# Today's context
# -----------------------------
latest = tqqq.iloc[-1]
prev = tqqq.iloc[-2]

latest_close = float(latest["Close"])
prev_close = float(prev["Close"])  # ✅ THIS IS NOW THE REFERENCE CLOSE
latest_rsi = float(latest["RSI"])
latest_vol = float(latest["Volatility"])
trend_up = bool(latest["QQQ_TrendUp"])

# ✅ Explicit reference close
ref_price = prev_close

st.subheader("Today's Context")

st.write(f"**Latest trading day in data:** {latest.name.date()}")
st.write(f"**Latest close (most recent row):** ${latest_close:,.2f}")
st.write(f"**Reference close (yesterday):** ${ref_price:,.2f}")

day_change = (latest_close / prev_close - 1) * 100
st.write(f"**Change vs yesterday:** {day_change:+.2f}%")

st.write(f"**RSI ({rsi_period}):** {latest_rsi:.1f}")
st.write(f"**QQQ trend ({trend_ma_len}-day MA):** {'Uptrend' if trend_up else 'Downtrend/Sideways'}")
st.write(f"**TQQQ annualized volatility (approx):** {latest_vol:.2%}")

# -----------------------------
# Derive buy/sell zones
# -----------------------------

buy_min = buy_min_base
buy_max = buy_max_base
sell_min = sell_min_base
sell_max = sell_max_base

# RSI adjustments
if latest_rsi < 40:
    buy_min *= 0.7
    buy_max *= 0.8
    sell_min *= 0.8
    sell_max *= 0.9
elif latest_rsi > 60:
    buy_min *= 1.1
    buy_max *= 1.2
    sell_min *= 0.9
    sell_max *= 0.95

# Trend adjustments
if trend_up:
    buy_min *= 0.9
    buy_max *= 0.9
    sell_min *= 0.9
    sell_max *= 0.95
else:
    buy_min *= 1.1
    buy_max *= 1.2
    sell_min *= 0.9
    sell_max *= 0.95

# Sanity
buy_min = max(0.5, min(buy_min, buy_max - 0.25))
sell_min = max(0.5, min(sell_min, sell_max - 0.25))

# ✅ BUY/SELL ZONES NOW USE ref_price (yesterday's close)
buy_zone_low_price = ref_price * (1 - buy_max / 100.0)
buy_zone_high_price = ref_price * (1 - buy_min / 100.0)

sell_zone_low_price = ref_price * (1 + sell_min / 100.0)
sell_zone_high_price = ref_price * (1 + sell_max / 100.0)

# -----------------------------
# Display zones
# -----------------------------
st.subheader("Suggested Buy/Sell Zones for Today")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Buy Zone (Dip Below Yesterday's Close)")
    st.write(f"**Reference:** ${ref_price:,.2f}")
    st.write(
        f"**Dip range:** {buy_min:.2f}% to {buy_max:.2f}% below ref\n"
        f"**Price zone:** ${buy_zone_low_price:,.2f} – ${buy_zone_high_price:,.2f}"
    )

with col2:
    st.markdown("### Sell Zone (Pop Above Yesterday's Close)")
    st.write(f"**Reference:** ${ref_price:,.2f}")
    st.write(
        f"**Pop range:** {sell_min:.2f}% to {sell_max:.2f}% above ref\n"
        f"**Price zone:** ${sell_zone_low_price:,.2f} – ${sell_zone_high_price:,.2f}"
    )

st.info(
    "This framework always measures from **yesterday's close**, even on weekends or early Mondays."
)

# -----------------------------
# Visuals
# -----------------------------
st.subheader("TQQQ Price and RSI")
st.line_chart(tqqq[["Close", "RSI"]])

st.subheader("QQQ Trend vs Moving Average")
st.line_chart(qqq[["Close", f"MA{trend_ma_len}"]].dropna())
