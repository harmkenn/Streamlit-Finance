import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.title("UDOW Dabble Helper – Buy/Sell Zones for Today v 4.2")

st.write(
    "This tool is **not financial advice**. It shows how someone *might* "
    "think about buy/sell zones for UDOW using recent trend, RSI, and "
    "typical intraday behavior."
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Settings")

lookback_days = st.sidebar.slider("Lookback window (days)", 60, 1000, 500, 10)
rsi_period = st.sidebar.slider("RSI period", 5, 28, 14)
vol_lookback = st.sidebar.slider("Volatility lookback (days)", 10, 60, 20, 5)
trend_ma_len = st.sidebar.slider("DIA trend MA length (days)", 20, 200, 50, 5)

# Base buy/sell ranges (in % from previous close)
buy_min_base = st.sidebar.slider("Base min buy dip (%)", 0.5, 10.0, 1.0, 0.25)
buy_max_base = st.sidebar.slider("Base max buy dip (%)", 1.0, 15.0, 3.0, 0.25)

sell_min_base = st.sidebar.slider("Base min sell pop (%)", 0.5, 10.0, 1.5, 0.25)
sell_max_base = st.sidebar.slider("Base max sell pop (%)", 1.0, 20.0, 3.5, 0.25)

st.sidebar.markdown("---")
st.sidebar.write("**Interpretation**")
st.sidebar.write(
    "- Buy zone is a *dip* below latest close.\n"
    "- Sell zone is a *pop* above latest close.\n"
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
with st.spinner("Loading UDOW and DIA data..."):
    udow = fetch_data("UDOW", lookback_days)
    dia = fetch_data("DIA", lookback_days)

if udow.empty or dia.empty:
    st.error("Could not load data for UDOW/DIA.")
    st.stop()

udow = udow[["Open", "High", "Low", "Close"]].astype(float)
dia = dia[["Close"]].astype(float)

# Compute RSI on UDOW
udow["RSI"] = compute_rsi(udow["Close"], period=rsi_period)

# Compute daily returns and volatility proxy (UDOW)
udow["Return"] = udow["Close"].pct_change()
udow["Volatility"] = udow["Return"].rolling(vol_lookback).std() * np.sqrt(252)

# Compute DIA trend
dia[f"MA{trend_ma_len}"] = dia["Close"].rolling(trend_ma_len).mean()
dia["TrendUp"] = dia["Close"] > dia[f"MA{trend_ma_len}"]

# Align latest dates
common_index = udow.index.intersection(dia.index)
udow = udow.loc[common_index].copy()
dia = dia.loc[common_index].copy()

udow["DIA_TrendUp"] = dia["TrendUp"]

# Drop early NaNs
udow = udow.dropna()

if udow.empty:
    st.error("Not enough overlapping data after indicators.")
    st.stop()

# -----------------------------
# Today's context
# -----------------------------
latest = udow.iloc[-1]
prev = udow.iloc[-2] if len(udow) >= 2 else None

latest_close = float(latest["Close"])
latest_rsi = float(latest["RSI"])
latest_vol = float(latest["Volatility"])
trend_up = bool(latest["DIA_TrendUp"])

st.subheader("Today's Context")

st.write(f"**Latest trading day:** {latest.name.date()}")
st.write(f"**UDOW close:** ${latest_close:,.2f}")
st.write(f"**RSI ({rsi_period}):** {latest_rsi:.1f}")
st.write(f"**DIA trend ({trend_ma_len}-day MA):** {'Uptrend' if trend_up else 'Downtrend/Sideways'}")
st.write(f"**UDOW annualized volatility (approx):** {latest_vol:.2%}")

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
buy_min = max(0.25, min(buy_min, buy_max - 0.1))
sell_min = max(0.25, min(sell_min, sell_max - 0.1))

# Convert to price zones using LATEST close instead of previous close
ref_price = latest_close  # Changed from prev_close to latest_close
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
    "This is a *framework* for thinking about UDOW entries/exits, not a signal generator. "
    "It adjusts generic % dip/pop zones based on today's RSI and the DIA trend."
)

# -----------------------------
# Visuals
# -----------------------------
st.subheader("UDOW Price and RSI")
price_rsi = udow[["Close", "RSI"]].copy()
st.line_chart(price_rsi)

st.subheader("DIA Trend vs Moving Average")
dia_viz = dia[["Close", f"MA{trend_ma_len}"]].dropna().copy()
st.line_chart(dia_viz)
