import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.title("TQQQ Trigger Strategy (MA-Adjusted + Bullish Bias) vs Buy-and-Hold")

# -----------------------------------------
# Strategy parameters with sliders
# -----------------------------------------
initial_cash = 100000
trade_amount = 10000

st.sidebar.header("Strategy Parameters")

drop_pct_5 = st.sidebar.slider("Buy Trigger Drop % (5% default)", 1, 20, 4) / 100
spike_pct_5 = st.sidebar.slider("Sell Trigger Rise % (5% default)", 1, 20, 7) / 100
drop_pct_10 = st.sidebar.slider("Buy Trigger Drop % (10% default)", 1, 30, 8) / 100
spike_pct_10 = st.sidebar.slider("Sell Trigger Rise % (10% default)", 1, 30, 14) / 100

ma_period = st.sidebar.slider("Moving Average Period", 5, 50, 7)

st.sidebar.header("Bullish Bias Parameters")
bullish_sell_multiplier = st.sidebar.slider("Bullish Sell Trigger Multiplier", 0.5, 1.5, 0.8, 0.1)
bullish_buy_multiplier = st.sidebar.slider("Bullish Buy Trigger Multiplier", 0.5, 2.0, 1.2, 0.1)

st.write(f"""
Simulating a strategy where:

- Start with **${initial_cash:,}** cash  
- Buy/Sell **${trade_amount:,}** on **{int(drop_pct_5*100)}% or {int(drop_pct_10*100)}% intraday moves**  
- Apply **MA{ma_period} filter**:  
  - *Buy price = min(trigger, MA{ma_period})*  
  - *Sell price = max(trigger, MA{ma_period})*  

**Bullish Bias Rules:**
- When price > MA20, MA50, MA200 AND RSI > 50:
  - Sell triggers are **{bullish_sell_multiplier}x** easier to hit
  - Buy triggers are **{bullish_buy_multiplier}x** harder to hit
""")

# -----------------------------------------
# Load 3 years of TQQQ daily data
# -----------------------------------------
ticker = "TQQQ"
df = yf.download(ticker, period="3y", interval="1d")

if df.empty:
    st.error("Error: No data returned.")
    st.stop()

df = df.astype(float)
df["PrevClose"] = df["Close"].shift(1)

# Calculate RSI
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Moving averages
df[f"MA{ma_period}"] = df["Close"].rolling(ma_period).mean()
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()
df["MA200"] = df["Close"].rolling(200).mean()
df["RSI"] = calculate_rsi(df["Close"])

# Ensure moving average columns exist
required_columns = ["MA20", "MA50", "MA200", "RSI"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing columns in DataFrame: {missing_columns}. Ensure sufficient data for calculations.")
    st.stop()

# Drop rows with NaN values in moving average columns
df = df.dropna(subset=required_columns)

# Align indices of Close and moving average columns
df["Close"], df["MA20"] = df["Close"].align(df["MA20"], axis=0)
df["Close"], df["MA50"] = df["Close"].align(df["MA50"], axis=0)
df["Close"], df["MA200"] = df["Close"].align(df["MA200"], axis=0)

# Bullish condition
df["Bullish"] = (
    (df["Close"] > df["MA20"]) & 
    (df["Close"] > df["MA50"]) & 
    (df["Close"] > df["MA200"]) & 
    (df["RSI"] > 50)
)

# -----------------------------------------
# Next-day trigger preview
last_close = float(df["Close"].iloc[-1])
ma_last = float(df[f"MA{ma_period}"].iloc[-1])
ma20_last = float(df["MA20"].iloc[-1])
ma50_last = float(df["MA50"].iloc[-1])
ma200_last = float(df["MA200"].iloc[-1])
rsi_last = float(df["RSI"].iloc[-1])
is_bullish = bool(df["Bullish"].iloc[-1])

# Base triggers
buy_5_price = last_close * (1 - drop_pct_5)
sell_5_price = last_close * (1 + spike_pct_5)
buy_10_price = last_close * (1 - drop_pct_10)
sell_10_price = last_close * (1 + spike_pct_10)

# Apply bullish bias
if is_bullish:
    buy_5_price = last_close * (1 - drop_pct_5 * bullish_buy_multiplier)
    sell_5_price = last_close * (1 + spike_pct_5 * bullish_sell_multiplier)
    buy_10_price = last_close * (1 - drop_pct_10 * bullish_buy_multiplier)
    sell_10_price = last_close * (1 + spike_pct_10 * bullish_sell_multiplier)

st.subheader("Market Conditions")
col1, col2 = st.columns(2)
with col1:
    st.write(f"Previous Close: **${last_close:,.2f}**")
    st.write(f"{ma_period}-Day MA: **${ma_last:,.2f}**")
    st.write(f"20-Day MA: **${ma20_last:,.2f}**")
    st.write(f"50-Day MA: **${ma50_last:,.2f}**")
with col2:
    st.write(f"200-Day MA: **${ma200_last:,.2f}**")
    st.write(f"RSI: **{rsi_last:.1f}**")
    bullish_status = "🟢 BULLISH" if is_bullish else "🔴 NEUTRAL/BEARISH"
    st.write(f"Market Bias: **{bullish_status}**")

st.subheader("Next Day Trigger Preview")

# Preview table
preview_df = pd.DataFrame({
    "Trigger Type": ["Buy 5%", "Sell 5%", "Buy 10%", "Sell 10%"],
    "Raw Trigger Price": [buy_5_price, sell_5_price, buy_10_price, sell_10_price],
    f"Adjusted for MA{ma_period}": [
        min(buy_5_price, ma_last),
        max(sell_5_price, ma_last),
        min(buy_10_price, ma_last),
        max(sell_10_price, ma_last)
    ],
    "Shares at Trigger Price": [
        trade_amount / min(buy_5_price, ma_last),
        trade_amount / max(sell_5_price, ma_last),
        trade_amount / min(buy_10_price, ma_last),
        trade_amount / max(sell_10_price, ma_last)
    ]
})

if is_bullish:
    st.info("🟢 Bullish conditions detected! Sell triggers are easier, buy triggers are harder.")

st.dataframe(preview_df.style.format({
    "Raw Trigger Price": "${:,.2f}",
    f"Adjusted for MA{ma_period}": "${:,.2f}",
    "Shares at Trigger Price": "{:,.4f}"
}))