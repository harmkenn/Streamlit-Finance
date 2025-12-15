import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.title("UDOW Dabble Helper – Buy/Sell Zones (Intraday-Derived Daily) v5.0")

st.write(
    "This tool is **not financial advice**. It shows how someone *might* "
    "think about buy/sell zones for UDOW using recent trend, RSI, and "
    "typical intraday behavior.\n\n"
    "Daily prices are reconstructed from 5-minute intraday data to avoid missing days "
    "in Yahoo's daily candles."
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Settings")

lookback_days = st.sidebar.slider("Lookback window (calendar days, intraday)", 20, 120, 60, 5)
rsi_period = st.sidebar.slider("RSI period (daily bars)", 5, 28, 14)
vol_lookback = st.sidebar.slider("Volatility lookback (daily bars)", 10, 60, 20, 5)
trend_ma_len = st.sidebar.slider("DIA trend MA length (daily bars)", 20, 200, 50, 5)

# Base buy/sell ranges (in % from previous close)
buy_min_base = st.sidebar.slider("Base min buy dip (%)", 1.0, 10.0, 3.0, 0.5)
buy_max_base = st.sidebar.slider("Base max buy dip (%)", 2.0, 15.0, 6.5, 0.5)

sell_min_base = st.sidebar.slider("Base min sell pop (%)", 1.0, 10.0, 3.0, 0.5)
sell_max_base = st.sidebar.slider("Base max sell pop (%)", 2.0, 20.0, 6.5, 0.5)

st.sidebar.markdown("---")
st.sidebar.write("**Interpretation**")
st.sidebar.write(
    "- Buy zone is a *dip* below **yesterday's close**.\n"
    "- Sell zone is a *pop* above **yesterday's close**.\n"
    "- RSI and DIA trend tweak these ranges.\n"
    "- All daily values are derived from intraday (regular-hours) data."
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

def build_daily_from_intraday(ticker: str, days: int) -> pd.DataFrame:
    yf_t = yf.Ticker(ticker)
    intraday = yf_t.history(period=f"{days}d", interval="5m", prepost=True)

    if intraday.empty:
        return pd.DataFrame()

    # Ensure timezone and restrict to regular-hours
    if intraday.index.tz is None:
        intraday = intraday.tz_localize("America/New_York")
    else:
        intraday = intraday.tz_convert("America/New_York")

    regular = intraday.between_time("09:30", "16:00")

    if regular.empty:
        return pd.DataFrame()

    # Aggregate to daily OHLC from regular-hours data
    daily = regular.groupby(regular.index.date).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    )

    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    return daily

# -----------------------------
# Load data via intraday → daily
# -----------------------------
with st.spinner("Loading intraday-derived daily data for UDOW and DIA..."):
    udow_daily = build_daily_from_intraday("UDOW", lookback_days)
    dia_daily = build_daily_from_intraday("DIA", lookback_days)

if udow_daily.empty or dia_daily.empty:
    st.error("Could not build daily data from intraday for UDOW/DIA.")
    st.stop()

# -----------------------------
# Indicators on daily data
# -----------------------------
# UDOW: RSI, returns, volatility
udow = udow_daily.copy()
udow["RSI"] = compute_rsi(udow["Close"], period=rsi_period)
udow["Return"] = udow["Close"].pct_change()
udow["Volatility"] = udow["Return"].rolling(vol_lookback).std() * np.sqrt(252)

# DIA: moving average trend
dia = dia_daily.copy()
dia[f"MA{trend_ma_len}"] = dia["Close"].rolling(trend_ma_len).mean()
dia["TrendUp"] = dia["Close"] > dia[f"MA{trend_ma_len}"]

# Merge DIA trend into UDOW without shrinking UDOW; align on date, forward-fill
udow = udow.join(dia["TrendUp"], how="left")
udow["TrendUp"] = udow["TrendUp"].ffill()

# Drop early NaNs where indicators are not ready
udow = udow.dropna(subset=["RSI", "Volatility", "TrendUp"])

if len(udow) < 3:
    st.error("Not enough daily bars after computing indicators.")
    st.stop()

# -----------------------------
# Today's context (daily from intraday)
# -----------------------------
latest = udow.iloc[-1]
prev = udow.iloc[-2]

latest_date = latest.name.date()
prev_date = prev.name.date()

latest_close = float(latest["Close"])
prev_close = float(prev["Close"])  # yesterday's close
latest_rsi = float(latest["RSI"])
latest_vol = float(latest["Volatility"])
trend_up = bool(latest["TrendUp"])

# Reference is explicitly yesterday's close
ref_price = prev_close

st.subheader("Today's Context")

st.write(f"**Latest trading day:** {latest_date}")
st.write(f"**UDOW close:** ${latest_close:,.2f}")
st.write(f"**Reference close (yesterday):** {prev_date} — ${ref_price:,.2f}")

day_change = (latest_close / ref_price - 1) * 100
st.write(f"**Change vs yesterday's close:** {day_change:+.2f}%")

st.write(f"**RSI ({rsi_period}):** {latest_rsi:.1f}")
st.write(f"**DIA trend ({trend_ma_len}-day MA):** {'Uptrend' if trend_up else 'Downtrend/Sideways'}")
st.write(f"**UDOW annualized volatility (approx):** {latest_vol:.2%}")

# -----------------------------
# Derive buy/sell zones from yesterday's close
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

# Sanity checks
buy_min = max(0.5, min(buy_min, buy_max - 0.25))
sell_min = max(0.5, min(sell_min, sell_max - 0.25))

# Zones based on yesterday's close (ref_price)
buy_zone_low_price = ref_price * (1 - buy_max / 100.0)
buy_zone_high_price = ref_price * (1 - buy_min / 100.0)

sell_zone_low_price = ref_price * (1 + sell_min / 100.0)
sell_zone_high_price = ref_price * (1 + sell_max / 100.0)

# -----------------------------
# Display zones
# -----------------------------
st.subheader("Suggested Buy/Sell Zones for Today (Anchored to Yesterday's Close)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Buy Zone (Dip Below Yesterday's Close)")
    st.write(f"**Reference (yesterday's close):** ${ref_price:,.2f}")
    st.write(
        f"**Dip range:** {buy_min:.2f}% to {buy_max:.2f}% below ref\n"
        f"**Price zone:** ${buy_zone_low_price:,.2f} – ${buy_zone_high_price:,.2f}"
    )

with col2:
    st.markdown("### Sell Zone (Pop Above Yesterday's Close)")
    st.write(f"**Reference (yesterday's close):** ${ref_price:,.2f}")
    st.write(
        f"**Pop range:** {sell_min:.2f}% to {sell_max:.2f}% above ref\n"
        f"**Price zone:** ${sell_zone_low_price:,.2f} – ${sell_zone_high_price:,.2f}"
    )

st.info(
    "Daily bars are reconstructed from 5-minute intraday data (regular-hours only), "
    "so missing days in Yahoo's daily candles are avoided. "
    "Zones always anchor to **yesterday's close** from this intraday-derived daily series."
)

# -----------------------------
# Visuals
# -----------------------------
st.subheader("UDOW Daily Close and RSI (from Intraday)")
st.line_chart(udow[["Close", "RSI"]])

st.subheader("DIA Daily Close and Trend MA (from Intraday)")
dia_viz = dia[["Close", f"MA{trend_ma_len}"]].dropna().copy()
st.line_chart(dia_viz)
