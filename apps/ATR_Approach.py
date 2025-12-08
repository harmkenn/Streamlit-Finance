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
if "MA20" not in df.columns or "MA50" not in df.columns or "MA200" not in df.columns:
    st.error("One or more moving average columns are missing. Ensure sufficient data for calculation.")
    st.stop()

# Drop rows with NaN values in moving average columns
df = df.dropna(subset=["MA20", "MA50", "MA200", "RSI"])

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
# -----------------------------------------
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

# -----------------------------------------
# Simulate Strategy with MA adjustment and bullish bias
# -----------------------------------------
cash = float(initial_cash)
shares = 0.0
trades = []
portfolio_value_over_time = []

for idx, row in df.iterrows():
    prev_close = float(row["PrevClose"])
    day_low = float(row["Low"])
    day_high = float(row["High"])
    day_close = float(row["Close"])
    ma = float(row[f"MA{ma_period}"])
    is_bullish_day = bool(row["Bullish"])

    # Base triggers
    buy_5 = prev_close * (1 - drop_pct_5)
    sell_5 = prev_close * (1 + spike_pct_5)
    buy_10 = prev_close * (1 - drop_pct_10)
    sell_10 = prev_close * (1 + spike_pct_10)

    # Apply bullish bias
    if is_bullish_day:
        buy_5 = prev_close * (1 - drop_pct_5 * bullish_buy_multiplier)
        sell_5 = prev_close * (1 + spike_pct_5 * bullish_sell_multiplier)
        buy_10 = prev_close * (1 - drop_pct_10 * bullish_buy_multiplier)
        sell_10 = prev_close * (1 + spike_pct_10 * bullish_sell_multiplier)

    # Adjusted for MA
    buy_5_adj = min(buy_5, ma)
    sell_5_adj = max(sell_5, ma)
    buy_10_adj = min(buy_10, ma)
    sell_10_adj = max(sell_10, ma)

    trade_suffix = " (Bullish)" if is_bullish_day else ""

    # BUY 5% (only if cash available)
    if day_low <= buy_5_adj and cash >= trade_amount:
        qty = trade_amount / buy_5_adj
        cash -= trade_amount
        shares += qty
        total_value = cash + shares * buy_5_adj
        trades.append([idx, f"BUY 5% (MA-adjusted){trade_suffix}", buy_5_adj, qty, cash, shares, total_value])

    # SELL 5%
    if day_high >= sell_5_adj:
        qty = trade_amount / sell_5_adj
        if shares >= qty:
            cash += trade_amount
            shares -= qty
            total_value = cash + shares * sell_5_adj
            trades.append([idx, f"SELL 5% (MA-adjusted){trade_suffix}", sell_5_adj, qty, cash, shares, total_value])

    # BUY 10% (only if cash available)
    if day_low <= buy_10_adj and cash >= trade_amount:
        qty = trade_amount / buy_10_adj
        cash -= trade_amount
        shares += qty
        total_value = cash + shares * buy_10_adj
        trades.append([idx, f"BUY 10% (MA-adjusted){trade_suffix}", buy_10_adj, qty, cash, shares, total_value])

    # SELL 10%
    if day_high >= sell_10_adj:
        qty = trade_amount / sell_10_adj
        if shares >= qty:
            cash += trade_amount
            shares -= qty
            total_value = cash + shares * sell_10_adj
            trades.append([idx, f"SELL 10% (MA-adjusted){trade_suffix}", sell_10_adj, qty, cash, shares, total_value])

    # Record daily portfolio value
    portfolio_value_over_time.append([idx, cash + shares * day_close])

# Final strategy value
final_value = cash + shares * float(df["Close"].iloc[-1])

# -----------------------------------------
# Buy-and-hold comparison
# -----------------------------------------
initial_shares = float(initial_cash / df["Close"].iloc[0])
buy_hold_value = df["Close"] * initial_shares

# Align portfolio_df and buy_hold_value
portfolio_df = pd.DataFrame(portfolio_value_over_time, columns=["Date", "StrategyValue"])
portfolio_df.set_index("Date", inplace=True)
portfolio_df["BuyHoldValue"] = buy_hold_value.reindex(portfolio_df.index).values

# -----------------------------------------
# Display Results
# -----------------------------------------
st.subheader("Final Results")
strategy_return = (final_value - initial_cash) / initial_cash * 100
buy_hold_return = (float(buy_hold_value.iloc[-1]) - initial_cash) / initial_cash * 100

col1, col2 = st.columns(2)
with col1:
    st.metric("Strategy Final Value", f"${final_value:,.2f}", f"{strategy_return:+.1f}%")
with col2:
    st.metric("Buy-and-Hold Final Value", f"${float(buy_hold_value.iloc[-1]):,.2f}", f"{buy_hold_return:+.1f}%")

# Trade statistics
if trades:
    trades_df = pd.DataFrame(
        trades,
        columns=["Date", "Type", "Execution Price", "Shares", "CashAfter", "SharesAfter", "TotalValue"]
    )
    
    bullish_trades = trades_df[trades_df["Type"].str.contains("Bullish")]
    normal_trades = trades_df[~trades_df["Type"].str.contains("Bullish")]
    
    st.subheader("Trade Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Trades", len(trades_df))
    with col2:
        st.metric("Bullish Trades", len(bullish_trades))
    with col3:
        st.metric("Normal Trades", len(normal_trades))

    st.subheader("Trade Log")
    st.dataframe(trades_df.style.format({
        "Execution Price": "${:,.2f}",
        "Shares": "{:,.4f}",
        "CashAfter": "${:,.2f}",
        "SharesAfter": "{:,.4f}",
        "TotalValue": "${:,.2f}"
    }))

# Portfolio curves
st.subheader("Portfolio Value vs Buy-and-Hold")
st.line_chart(portfolio_df)

# Technical indicators chart
st.subheader("Technical Indicators")
chart_df = df[["Close", "MA20", "MA50", "MA200", "RSI"]].copy()

# Align RSI with Close before scaling
chart_df["RSI_Scaled"] = chart_df["RSI"].align(chart_df["Close"], axis=0)[0] * (chart_df["Close"].iloc[-1] / 100)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Price & Moving Averages")
    st.line_chart(chart_df[["Close", "MA20", "MA50", "MA200"]])
with col2:
    st.subheader("RSI")
    st.line_chart(chart_df[["RSI"]])

# Bullish periods
bullish_periods = df["Bullish"].sum()
total_periods = len(df)
st.write(f"**Bullish periods:** {bullish_periods} out of {total_periods} days ({bullish_periods/total_periods*100:.1f}%)")