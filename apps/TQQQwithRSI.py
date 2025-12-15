import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.title("TQQQ Dabble Helper – Buy/Sell Zones (Intraday-Derived Daily) v5.0")

st.write(
    "This tool is **not financial advice**. It shows how someone *might* "
    "think about buy/sell zones for TQQQ using recent trend, RSI, and "
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
trend_ma_len = st.sidebar.slider("QQQ trend MA length (daily bars)", 20, 200, 50, 5)

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
    "- RSI and QQQ trend tweak these ranges.\n"
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
    """
    Fetch intraday 5m data and build daily OHLC from regular-hours candles.
    """
    yf_t = yf.Ticker(ticker)
    # period in days; use 5m to cover pre/post but we’ll restrict to RTH
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
with st.spinner("Loading intraday-derived daily data for TQQQ and QQQ..."):
    tqqq_daily = build_daily_from_intraday("TQQQ", lookback_days)
    qqq_daily = build_daily_from_intraday("QQQ", lookback_days)

if tqqq_daily.empty or qqq_daily.empty:
    st.error("Could not build daily data from intraday for TQQQ/QQQ.")
    st.stop()

# -----------------------------
# Indicators on daily data
# -----------------------------
# TQQQ: RSI, returns, volatility
tqqq = tqqq_daily.copy()
tqqq["RSI"] = compute_rsi(tqqq["Close"], period=rsi_period)
tqqq["Return"] = tqqq["Close"].pct_change()
tqqq["Volatility"] = tqqq["Return"].rolling(vol_lookback).std() * np.sqrt(252)

# QQQ: moving average trend
qqq = qqq_daily.copy()
qqq[f"MA{trend_ma_len}"] = qqq["Close"].rolling(trend_ma_len).mean()
qqq["TrendUp"] = qqq["Close"] > qqq[f"MA{trend_ma_len}"]

# Merge QQQ trend into TQQQ without shrinking TQQQ; align on date, forward-fill
tqqq = tqqq.join(qqq["TrendUp"], how="left")
tqqq["TrendUp"] = tqqq["TrendUp"].ffill()

# Drop early NaNs where indicators are not ready
tqqq = tqqq.dropna(subset=["RSI", "Volatility", "TrendUp"])

if len(tqqq) < 3:
    st.error("Not enough daily bars after computing indicators.")
    st.stop()

# -----------------------------
# Today's context (daily from intraday)
# -----------------------------
latest = tqqq.iloc[-1]
prev = tqqq.iloc[-2]

latest_date = latest.name.date()
prev_date = prev.name.date()

latest_close = float(latest["Close"])
prev_close = float(prev["Close"])  # yesterday's close
latest_rsi = float(latest["RSI"])
latest_vol = float(latest["Volatility"])
trend_up = bool(latest["TrendUp"])

# Reference is explicitly yesterday's close
ref_price = prev_close

st.subheader("Today's Context (Daily from Intraday)")

st.write(f"**Latest daily bar date:** {latest_date}")
st.write(f"**Latest close (most recent daily bar):** ${latest_close:,.2f}")
st.write(f"**Reference close (yesterday's daily bar):** {prev_date} — ${ref_price:,.2f}")

day_change = (latest_close / ref_price - 1) * 100
st.write(f"**Change vs yesterday's close:** {day_change:+.2f}%")

st.write(f"**RSI ({rsi_period}):** {latest_rsi:.1f}")
st.write(f"**QQQ trend ({trend_ma_len}-day MA):** {'Uptrend' if trend_up else 'Downtrend/Sideways'}")
st.write(f"**TQQQ annualized volatility (approx):** {latest_vol:.2%}")

# -----------------------------
# Derive buy/sell zones from yesterday's close
# -----------------------------
buy_min = buy_min_base
buy_max = buy_max_base
sell_min = sell_min_base
sell_max = sell_max_base

# RSI adjustments
if latest_rsi < 40:
    # more eager to buy, less greedy on sells
    buy_min *= 0.7
    buy_max *= 0.8
    sell_min *= 0.8
    sell_max *= 0.9
elif latest_rsi > 60:
    # more cautious buys, quicker profit-taking
    buy_min *= 1.1
    buy_max *= 1.2
    sell_min *= 0.9
    sell_max *= 0.95

# Trend adjustments
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
    "so missing days in Yahoo's daily candles (like your December 12th case) are avoided. "
    "Zones always anchor to **yesterday's close** from this intraday-derived daily series."
)

# -----------------------------
# Visuals
# -----------------------------
st.subheader("TQQQ Daily Close and RSI (from Intraday)")

price_rsi = tqqq[["Close", "RSI"]].copy()
st.line_chart(price_rsi)

st.subheader("QQQ Daily Close and Trend MA (from Intraday)")

qqq_viz = qqq[[ "Close", f"MA{trend_ma_len}"]].dropna().copy()
st.line_chart(qqq_viz)
