import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.title("TQQQ Dabble Helper – Buy/Sell Zones for Today v4.3")

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
    df = yf.download(ticker, period=f"{days}d", interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
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

tqqq = tqqq[["Open", "High", "Low", "Close"]].astype(float)
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
prev = tqqq.iloc[-2] if len(tqqq) >= 2 else None

latest_close = float(latest["Close"])
latest_rsi = float(latest["RSI"])
latest_vol = float(latest["Volatility"])
trend_up = bool(latest["QQQ_TrendUp"])

st.subheader("Today's Context")

st.write(f"**Latest trading day:** {latest.name.date()}")
st.write(f"**TQQQ close:** ${latest_close:,.2f}")
st.write(f"**RSI ({rsi_period}):** {latest_rsi:.1f}")
st.write(f"**QQQ trend ({trend_ma_len}-day MA):** {'Uptrend' if trend_up else 'Downtrend/Sideways'}")
st.write(f"**TQQQ annualized volatility (approx):** {latest_vol:.2%}")

if prev is not None:
    prev_close = float(prev["Close"])
    day_change = (latest_close / prev_close - 1) * 100
    st.write(f"**Change vs previous close:** {day_change:+.2f}%")
else:
    prev_close = latest_close
    st.info("Not enough data for previous close comparison; using latest close as reference.")
    day_change = 0.0

# -----------------------------
# Derive buy/sell zones
# -----------------------------

# 1. Start with base ranges (as % from previous close)
buy_min = buy_min_base
buy_max = buy_max_base
sell_min = sell_min_base
sell_max = sell_max_base

# 2. Adjust based on RSI regime
#    - Low RSI: buy smaller dips, sell smaller pops (expect oversold bounces)
#    - High RSI: buy deeper dips, sell sooner (expect mean reversion)
if latest_rsi < 40:
    # More eager to buy, less greedy on sells
    buy_min *= 0.7
    buy_max *= 0.8
    sell_min *= 0.8
    sell_max *= 0.9
elif latest_rsi > 60:
    # More cautious buys, quicker profit-taking
    buy_min *= 1.1
    buy_max *= 1.2
    sell_min *= 0.9
    sell_max *= 0.95
# RSI 40–60 → no change (neutral)

# 3. Adjust based on trend (QQQ)
if trend_up:
    # Uptrend: buy shallower dips, accept smaller profit targets
    buy_min *= 0.9
    buy_max *= 0.9
    sell_min *= 0.9
    sell_max *= 0.95
else:
    # Not in clear uptrend: be pickier on buys, keep sells closer
    buy_min *= 1.1
    buy_max *= 1.2
    sell_min *= 0.9
    sell_max *= 0.95

# 4. Sanity: enforce ordering
buy_min = max(0.5, min(buy_min, buy_max - 0.25))
sell_min = max(0.5, min(sell_min, sell_max - 0.25))

# 5. Translate % zones into price levels using LATEST close (Friday's close)
ref_price = float(prev["Close"])
  # Changed from prev_close to latest_close
buy_zone_low_price = ref_price * (1 - buy_max / 100.0)
buy_zone_high_price = ref_price * (1 - buy_min / 100.0)

sell_zone_low_price = ref_price * (1 + sell_min / 100.0)
sell_zone_high_price = ref_price * (1 + sell_max / 100.0)

# -----------------------------
# Display zones
# -----------------------------
st.subheader("Suggested Buy/Sell Zones for Today")

st.markdown("**These are *zones*, not exact levels.** They are based on:")
st.markdown(
"- Typical % moves of TQQQ from the latest close\n"
"- Whether RSI is low/neutral/high\n"
"- Whether QQQ is in an uptrend or not"
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Buy Zone (Dip Below Latest Close)")
    st.write(f"**Reference (latest close):** ${ref_price:,.2f}")  # Updated text
    st.write(
        f"**Dip range:** {buy_min:.2f}% to {buy_max:.2f}% below ref\n"
        f"**Price zone:** ${buy_zone_low_price:,.2f} – ${buy_zone_high_price:,.2f}"
    )

with col2:
    st.markdown("### Sell Zone (Pop Above Latest Close)")
    st.write(f"**Reference (latest close):** ${ref_price:,.2f}")  # Updated text
    st.write(
        f"**Pop range:** {sell_min:.2f}% to {sell_max:.2f}% above ref\n"
        f"**Price zone:** ${sell_zone_low_price:,.2f} – ${sell_zone_high_price:,.2f}"
    )

st.info(
    "This is a *framework* for thinking about TQQQ entries/exits, not a signal generator. "
    "It adjusts generic % dip/pop zones based on today's RSI and the QQQ trend."
)

# -----------------------------
# Visuals
# -----------------------------
st.subheader("TQQQ Price and RSI")

price_rsi = tqqq[["Close", "RSI"]].copy()
st.line_chart(price_rsi)

st.subheader("QQQ Trend vs Moving Average")
qqq_viz = qqq[[ "Close", f"MA{trend_ma_len}"]].dropna().copy()
st.line_chart(qqq_viz)
