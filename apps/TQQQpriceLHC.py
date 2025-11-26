import streamlit as st
import pandas as pd
import yfinance as yf

st.title("TQQQ Trigger Strategy (MA-Adjusted) vs Buy-and-Hold")

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

st.write(f"""
Simulating a strategy where:

- Start with **${initial_cash:,}** cash  
- Buy/Sell **${trade_amount:,}** on **{int(drop_pct_5*100)}% or {int(drop_pct_10*100)}% intraday moves**  
- Apply **MA{ma_period} filter**:  
  - *Buy price = min(trigger, MA{ma_period})*  
  - *Sell price = max(trigger, MA{ma_period})*  
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

# Moving average
df[f"MA{ma_period}"] = df["Close"].rolling(ma_period).mean()

df = df.dropna()

# -----------------------------------------
# Next-day trigger preview
# -----------------------------------------
last_close = float(df["Close"].iloc[-1])
ma_last = float(df[f"MA{ma_period}"].iloc[-1])

# 5% triggers
buy_5_price = last_close * (1 - drop_pct_5)
sell_5_price = last_close * (1 + spike_pct_5)

# 10% triggers
buy_10_price = last_close * (1 - drop_pct_10)
sell_10_price = last_close * (1 + spike_pct_10)

st.subheader("Next Day Trigger Preview")
st.write(f"Previous Close: **${last_close:,.2f}**")
st.write(f"{ma_period}-Day Moving Average: **${ma_last:,.2f}**")

# 5% and 10% preview table
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

st.dataframe(preview_df.style.format({
    "Raw Trigger Price": "${:,.2f}",
    f"Adjusted for MA{ma_period}": "${:,.2f}",
    "Shares at Trigger Price": "{:,.4f}"
}))

# -----------------------------------------
# Simulate Strategy with MA adjustment
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

    # Raw triggers
    buy_5 = prev_close * (1 - drop_pct_5)
    sell_5 = prev_close * (1 + spike_pct_5)
    buy_10 = prev_close * (1 - drop_pct_10)
    sell_10 = prev_close * (1 + spike_pct_10)

    # Adjusted for MA
    buy_5_adj = min(buy_5, ma)
    sell_5_adj = max(sell_5, ma)
    buy_10_adj = min(buy_10, ma)
    sell_10_adj = max(sell_10, ma)

    # BUY 5% (only if cash available)
    if day_low <= buy_5_adj and cash >= trade_amount:
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

    # BUY 10% (only if cash available)
    if day_low <= buy_10_adj and cash >= trade_amount:
        qty = trade_amount / buy_10_adj
        cash -= trade_amount
        shares += qty
        total_value = cash + shares * buy_10_adj
        trades.append([idx, "BUY 10% (MA-adjusted)", buy_10_adj, qty, cash, shares, total_value])

    # SELL 10%
    if day_high >= sell_10_adj:
        qty = trade_amount / sell_10_adj
        if shares >= qty:
            cash += trade_amount
            shares -= qty
            total_value = cash + shares * sell_10_adj
            trades.append([idx, "SELL 10% (MA-adjusted)", sell_10_adj, qty, cash, shares, total_value])

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
