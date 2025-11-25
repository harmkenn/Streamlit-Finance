import streamlit as st
import pandas as pd
import yfinance as yf

st.title("TQQQ ±5% & ±10% Trigger Strategy (MA-Adjusted) vs Buy-and-Hold")

# -----------------------------------------
# Strategy parameters
# -----------------------------------------
initial_cash = 100000
trade_amount = 10000

drop_pct_5 = 0.95     # 5% drop
spike_pct_5 = 1.05    # 5% rise
drop_pct_10 = 0.90    # 10% drop
spike_pct_10 = 1.10   # 10% rise

st.write(f"""
Simulating a strategy where:

- Start with **${initial_cash:,}** cash  
- Buy/Sell **$10,000** on **5% or 10% intraday moves**  
- Apply **MA10 filter**:  
  - *Buy price = min(trigger, MA10)*  
  - *Sell price = max(trigger, MA10)*  
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

# 10-day moving average
df["MA10"] = df["Close"].rolling(10).mean()

df = df.dropna()

# -----------------------------------------
# Next-day trigger preview
# -----------------------------------------
last_close = float(df["Close"].iloc[-1])
ma10_last = float(df["MA10"].iloc[-1])

# 5% triggers
buy_5_price = last_close * drop_pct_5
sell_5_price = last_close * spike_pct_5

# 10% triggers
buy_10_price = last_close * drop_pct_10
sell_10_price = last_close * spike_pct_10

st.subheader("Next Day Trigger Preview")
st.write(f"Previous Close: **${last_close:,.2f}**")
st.write(f"10-Day Moving Average: **${ma10_last:,.2f}**")

# 5% and 10% preview table
preview_df = pd.DataFrame({
    "Trigger Type": ["Buy 5%", "Sell 5%", "Buy 10%", "Sell 10%"],
    "Raw Trigger Price": [buy_5_price, sell_5_price, buy_10_price, sell_10_price],
    "Adjusted for MA10": [
        min(buy_5_price, ma10_last),
        max(sell_5_price, ma10_last),
        min(buy_10_price, ma10_last),
        max(sell_10_price, ma10_last)
    ],
    "Shares at Trigger Price": [
        trade_amount / min(buy_5_price, ma10_last),
        trade_amount / max(sell_5_price, ma10_last),
        trade_amount / min(buy_10_price, ma10_last),
        trade_amount / max(sell_10_price, ma10_last)
    ]
})

st.dataframe(preview_df.style.format({
    "Raw Trigger Price": "${:,.2f}",
    "Adjusted for MA10": "${:,.2f}",
    "Shares at Trigger Price": "{:,.4f}"
}))

# -----------------------------------------
# Simulate Strategy with MA10 adjustment
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
    ma10 = float(row["MA10"])

    # Raw 5% triggers
    buy_5 = prev_close * drop_pct_5
    sell_5 = prev_close * spike_pct_5

    # Apply MA10 filter:
    buy_5_adj = min(buy_5, ma10)
    sell_5_adj = max(sell_5, ma10)

    # BUY 5%
    if day_low <= buy_5_adj:
        qty = trade_amount / buy_5_adj
        cash -= trade_amount
        shares += qty
        total_value = cash + shares * buy_5_adj
        trades.append([idx, "BUY 5% (MA-adjusted)", buy_5_adj, qty, cash, shares, total_value])

    # SELL 5%
    if day_high >= sell_5_adj:
        qty = trade_amount / sell_5_adj
        if shares >= qty:
            cash += trade_amount
            shares -= qty
            total_value = cash + shares * sell_5_adj
            trades.append([idx, "SELL 5% (MA-adjusted)", sell_5_adj, qty, cash, shares, total_value])

    # Record daily portfolio value
    portfolio_value_over_time.append([idx, cash + shares * day_close])

# Final strategy value
final_value = cash + shares * float(df["Close"].iloc[-1])

# -----------------------------------------
# Buy-and-hold comparison
# -----------------------------------------
initial_shares = float(initial_cash / df["Close"].iloc[0])
buy_hold_value = df["Close"] * initial_shares

# -----------------------------------------
# Display Results
# -----------------------------------------
st.subheader("Final Results")

st.write(f"**Strategy Final Value:** ${final_value:,.2f}")
st.write(f"**Buy-and-Hold Final Value:** ${float(buy_hold_value.iloc[-1]):,.2f}")

# Trade log
trades_df = pd.DataFrame(
    trades,
    columns=["Date", "Type", "Execution Price", "Shares", "CashAfter", "SharesAfter", "TotalValue"]
)
st.subheader("Trade Log")
st.dataframe(trades_df)

# Portfolio curves
portfolio_df = pd.DataFrame(portfolio_value_over_time, columns=["Date", "StrategyValue"])
portfolio_df.set_index("Date", inplace=True)
portfolio_df["BuyHoldValue"] = buy_hold_value.values

st.subheader("Portfolio Value vs Buy-and-Hold")
st.line_chart(portfolio_df)

st.subheader("TQQQ Closing Price")
st.line_chart(df["Close"])
