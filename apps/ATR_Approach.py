import streamlit as st
import pandas as pd
import yfinance as yf

st.title("TQQQ ATR-Based Dip & Rip Strategy")

# ---------------------------------------------------------
# Strategy Parameters
# ---------------------------------------------------------
initial_cash = 100000
trade_amount = 10000

st.sidebar.header("ATR Strategy Parameters")

atr_period = st.sidebar.slider("ATR Period", 5, 30, 14)
atr_mult_buy = st.sidebar.slider("ATR Buy Multiplier", 0.5, 3.0, .5)
atr_mult_sell = st.sidebar.slider("ATR Sell Multiplier", 0.5, 3.0, 1.5)

st.write(f"""
Strategy rules:

- Start with **${initial_cash:,}**  
- Each trade amount: **${trade_amount:,}**
- Use **ATR({atr_period})**
- Buy if price drops **≥ ATR × {atr_mult_buy}** below previous close  
- Sell if price rises **≥ ATR × {atr_mult_sell}** above previous close  
- No negative cash allowed  
""")

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
ticker = "TQQQ"
df = yf.download(ticker, period="3y", interval="1d")

if df.empty:
    st.error("Error: No data returned.")
    st.stop()

df = df.astype(float)
df["PrevClose"] = df["Close"].shift(1)

# ---------------------------------------------------------
# Compute ATR
# ---------------------------------------------------------
high_low = df["High"] - df["Low"]
high_pc = (df["High"] - df["PrevClose"]).abs()
low_pc = (df["Low"] - df["PrevClose"]).abs()

tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
df["ATR"] = tr.rolling(atr_period).mean()

df = df.dropna()

# ---------------------------------------------------------
# ATR Trigger Preview (Next Day)
# ---------------------------------------------------------
last_close = float(df["Close"].iloc[-1])
last_atr = float(df["ATR"].iloc[-1])

buy_trigger = last_close - last_atr * atr_mult_buy
sell_trigger = last_close + last_atr * atr_mult_sell

st.subheader("Next-Day ATR Trigger Preview")
st.write(f"Previous Close: **${last_close:,.2f}**")
st.write(f"ATR({atr_period}): **${last_atr:,.2f}**")

preview_df = pd.DataFrame({
    "Trigger Type": ["Buy ATR", "Sell ATR"],
    "Trigger Price": [buy_trigger, sell_trigger],
    "Shares at Trigger Price": [
        trade_amount / buy_trigger,
        trade_amount / sell_trigger
    ]
})

st.dataframe(preview_df.style.format({
    "Trigger Price": "${:,.2f}",
    "Shares at Trigger Price": "{:,.4f}"
}))

# ---------------------------------------------------------
# Simulate ATR Strategy
# ---------------------------------------------------------
cash = float(initial_cash)
shares = 0.0
trades = []
portfolio_value_over_time = []

for idx, row in df.iterrows():
    prev_close = float(row["PrevClose"])
    day_low = float(row["Low"])
    day_high = float(row["High"])
    day_close = float(row["Close"])
    atr = float(row["ATR"])

    # ATR triggers
    buy_price = prev_close - atr * atr_mult_buy
    sell_price = prev_close + atr * atr_mult_sell

    # BUY ATR
    if day_low <= buy_price and cash >= trade_amount:
        qty = trade_amount / buy_price
        cash -= trade_amount
        shares += qty
        total_value = cash + shares * buy_price
        trades.append([idx, "BUY ATR", buy_price, qty, cash, shares, total_value])

    # SELL ATR
    if day_high >= sell_price:
        qty = trade_amount / sell_price
        if shares >= qty:
            cash += trade_amount
            shares -= qty
            total_value = cash + shares * sell_price
            trades.append([idx, "SELL ATR", sell_price, qty, cash, shares, total_value])

    # Record daily portfolio value
    portfolio_value_over_time.append([idx, cash + shares * day_close])

# Final value
final_value = cash + shares * float(df["Close"].iloc[-1])

# ---------------------------------------------------------
# Buy-and-Hold Comparison
# ---------------------------------------------------------
initial_shares = float(initial_cash / df["Close"].iloc[0])
buy_hold_value = df["Close"] * initial_shares

# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------
st.subheader("Final Results")
st.write(f"**ATR Strategy Final Value:** ${final_value:,.2f}")
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
