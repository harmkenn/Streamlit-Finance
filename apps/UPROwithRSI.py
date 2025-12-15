import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.title("UPRO Dabble Helper – Buy/Sell Zones for Today v4.2")

st.write(
    "This tool is **not financial advice**. It shows how someone *might* "
    "think about buy/sell zones for UPRO using recent trend, RSI, and "
    "typical intraday behavior."
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Settings")

lookback_days = st.sidebar.slider("Lookback window (days)", 60, 1000, 500, 10)
rsi_period = st.sidebar.slider("RSI period", 5, 28, 14)
vol_lookback = st.sidebar.slider("Volatility lookback (days)", 10, 60, 20, 5)
trend_ma_len = st.sidebar.slider("SPY trend MA length (days)", 20, 200, 50, 5)

# Base buy/sell ranges (in % from previous close)
buy_min_base = st.sidebar.slider("Base min buy dip (%)", 0.5, 10.0, 1.5, 0.25)
buy_max_base = st.sidebar.slider("Base max buy dip (%)", 1.0, 15.0, 3.5, 0.25)

sell_min_base = st.sidebar.slider("Base min sell pop (%)", 0.5, 10.0, 2.0, 0.25)
sell_max_base = st.sidebar.slider("Base max sell pop (%)", 1.0, 20.0, 4.0, 0.25)

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
with st.spinner("Loading UPRO and SPY data..."):
    upro = fetch_data("UPRO", lookback_days)
    spy = fetch_data("SPY", lookback_days)

if upro.empty or spy.empty:
    st.error("Could not load data for UPRO/SPY.")
    st.stop()

upro = upro[["Open", "High", "Low", "Close"]].astype(float)
spy = spy[["Close"]].astype(float)

# Compute RSI on UPRO
upro["RSI"] = compute_rsi(upro["Close"], period=rsi_period)

# Compute daily returns and volatility proxy (UPRO)
upro["Return"] = upro["Close"].pct_change()
upro["Volatility"] = upro["Return"].rolling(vol_lookback).std() * np.sqrt(252)

# Compute SPY trend
spy[f"MA{trend_ma_len}"] = spy["Close"].rolling(trend_ma_len).mean()
spy["TrendUp"] = spy["Close"] > spy[f"MA{trend_ma_len}"]

# Align latest dates
common_index = upro.index.intersection(spy.index)
upro = upro.loc[common_index].copy()
spy = spy.loc[common_index].copy()

upro["SPY_TrendUp"] = spy["TrendUp"]

# Drop early NaNs
upro = upro.dropna()

if upro.empty:
    st.error("Not enough overlapping data after indicators.")
    st.stop()

# -----------------------------
# Today's context
# -----------------------------
latest = upro.iloc[-1]
prev = upro.iloc[-2] if len(upro) >= 2 else None

latest_close = float(latest["Close"])
latest_rsi = float(latest["RSI"])
latest_vol = float(latest["Volatility"])
trend_up = bool(latest["SPY_TrendUp"])

st.subheader("Today's Context")

st.write(f"**Latest trading day:** {latest.name.date()}")
st.write(f"**UPRO close:** ${latest_close:,.2f}")
st.write(f"**RSI ({rsi_period}):** {latest_rsi:.1f}")
st.write(f"**SPY trend ({trend_ma_len}-day MA):** {'Uptrend' if trend_up else 'Downtrend/Sideways'}")
st.write(f"**UPRO annualized volatility (approx):** {latest_vol:.2%}")

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
    "This is a *framework* for thinking about UPRO entries/exits, not a signal generator. "
    "It adjusts generic % dip/pop zones based on today's RSI and the SPY trend."
)

# -----------------------------
# Visuals
# -----------------------------
st.subheader("UPRO Price and RSI")
price_rsi = upro[["Close", "RSI"]].copy()
st.line_chart(price_rsi)

st.subheader("SPY Trend vs Moving Average")
spy_viz = spy[["Close", f"MA{trend_ma_len}"]].dropna().copy()
st.line_chart(spy_viz)
