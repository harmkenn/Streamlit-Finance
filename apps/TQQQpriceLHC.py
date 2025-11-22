import streamlit as st
import pandas as pd
import yfinance as yf

st.title("TQQQ ±5% & ±10% Intraday Trigger Strategy vs Buy-and-Hold")

# -----------------------------------------
# Strategy parameters
# -----------------------------------------
initial_cash = 100000
trade_amount = 10000
drop_pct_5 = 0.95   # 5% drop
spike_pct_5 = 1.05  # 5% rise
drop_pct_10 = 0.90  # 10% drop
spike_pct_10 = 1.10 # 10% rise

st.write(f"""
Simulating a strategy where:
- Start with **${initial_cash:,}** cash
- Buy/Sell **$10,000** on **5% or 10% intraday moves**
- Multiple trades allowed in the same day
""")

# -----------------------------------------
# Load daily TQQQ data (12 months)
# -----------------------------------------
ticker = "TQQQ"
df = yf.download(ticker, period="3y", interval="1d")

if df.empty:
    st.error("Error: No data returned.")
    st.stop()

df = df.astype(float)
df["PrevClose"] = df["Close"].shift(1)
df = df.dropna()

# -----------------------------------------
# Next day trigger preview (5% and 10%)
# -----------------------------------------
last_close = float(df["Close"].iloc[-1])

# 5% triggers
buy_5_price = last_close * drop_pct_5
sell_5_price = last_close * spike_pct_5
buy_5_shares = trade_amount / buy_5_price
sell_5_shares = trade_amount / sell_5_price

# 10% triggers
buy_10_price = last_close * drop_pct_10
sell_10_price = last_close * spike_pct_10
buy_10_shares = trade_amount / buy_10_price
sell_10_shares = trade_amount / sell_10_price

st.subheader("Next Day Trigger Preview")
st.write(f"- Previous Close: **${last_close:,.2f}**")

st.write("**5% Triggers**")
st.write(f"- Buy Trigger: **${buy_5_price:,.2f}** → Buy **{buy_5_shares:.4f} shares**")
st.write(f"- Sell Trigger: **${sell_5_price:,.2f}** → Sell **{sell_5_shares:.4f} shares**")

st.write("**10% Triggers**")
st.write(f"- Buy Trigger: **${buy_10_price:,.2f}** → Buy **{buy_10_shares:.4f} shares**")
st.write(f"- Sell Trigger: **${sell_10_price:,.2f}** → Sell **{sell_10_shares:.4f} shares**")

# -----------------------------------------
# Simulate Strategy (5% triggers only for now)
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

    # 5% triggers
    buy_5 = prev_close * drop_pct_5
    sell_5 = prev_close * spike_pct_5

    # BUY 5% first
    if day_low <= buy_5:
        qty = trade_amount / buy_5
        cash -= trade_amount
        shares += qty
        total_value = cash + shares * buy_5
        trades.append([idx, "BUY 5%", buy_5, qty, cash, shares, total_value])

    # SELL 5% next
    if day_high >= sell_5:
        qty = trade_amount / sell_5
        if shares >= qty:
            cash += trade_amount
            shares -= qty
            total_value = cash + shares * sell_5
            trades.append([idx, "SELL 5%", sell_5, qty, cash, shares, total_value])

    # Record portfolio value at day close
    portfolio_value_over_time.append([idx, cash + shares * day_close])

# Final portfolio value
final_value = cash + shares * float(df["Close"].iloc[-1])

# -----------------------------------------
# Buy-and-hold comparison
# -----------------------------------------
initial_shares = float(initial_cash / df["Close"].iloc[0])
buy_hold_value = df["Close"] * initial_shares

# -----------------------------------------
# Output results
# -----------------------------------------
st.subheader("Final Results")
st.write(f"**Strategy final portfolio value:** ${float(final_value):,.2f}")
st.write(f"**Buy-and-hold final value:** ${float(buy_hold_value.iloc[-1]):,.2f}")

# Trade log
trades_df = pd.DataFrame(
    trades,
    columns=["Date", "Type", "Execution Price", "Shares", "CashAfter", "SharesAfter", "TotalValue"]
)
st.subheader("Trade Log")
st.dataframe(trades_df)

# Portfolio equity curve
portfolio_df = pd.DataFrame(portfolio_value_over_time, columns=["Date", "StrategyValue"])
portfolio_df.set_index("Date", inplace=True)
portfolio_df["BuyHoldValue"] = buy_hold_value.values
st.subheader("Portfolio vs Buy-and-Hold")
st.line_chart(portfolio_df)

# Closing price chart
st.subheader("TQQQ Closing Price")
st.line_chart(df["Close"])
