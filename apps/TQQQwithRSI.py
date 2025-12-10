import streamlit as st
import pandas as pd
import yfinance as yf

st.title("TQQQ RSI-Adjusted Trigger Strategy vs Buy-and-Hold")

# -----------------------------------------
# Strategy parameters
# -----------------------------------------
initial_cash = 100000
trade_amount = 10000

st.sidebar.header("Strategy Parameters")
ma_period = st.sidebar.slider("Moving Average Period", 5, 50, 7)

st.write(f"""
Simulating a strategy where:

- Start with **${initial_cash:,}** cash  
- Buy/Sell **${trade_amount:,}** based on RSI-adjusted thresholds  
- Apply **MA{ma_period} filter**:  
  - *Buy price = min(trigger, MA{ma_period})*  
  - *Sell price = max(trigger, MA{ma_period})*  
""")

# -----------------------------------------
# Load 5 years of TQQQ daily data
# -----------------------------------------
ticker = "TQQQ"
df = yf.download(ticker, period="5y", interval="1d")

if df.empty:
    st.error("Error: No data returned.")
    st.stop()

df = df.astype(float)
df["PrevClose"] = df["Close"].shift(1)

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    return data

df = calculate_rsi(df)
df[f"MA{ma_period}"] = df["Close"].rolling(ma_period).mean()
df = df.dropna()

# -----------------------------------------
# RSI-based dynamic thresholds
# -----------------------------------------
def get_dynamic_thresholds(rsi):
    """
    Calculate dynamic buy/sell thresholds based on RSI.
    - RSI = 75: Sell at 2.5%, Buy at 6.5%
    - RSI = 25: Sell at 6.5%, Buy at 2.5%
    - Linear scaling for RSI between 25 and 75
    """
    if rsi >= 75:
        sell_threshold = 0.025  # 2.5%
        buy_threshold = 0.065  # 6.5%
    elif rsi <= 25:
        sell_threshold = 0.065  # 6.5%
        buy_threshold = 0.025  # 2.5%
    else:
        # Linear scaling
        sell_threshold = 0.065 - (rsi - 25) * (0.065 - 0.025) / (75 - 25)
        buy_threshold = 0.025 + (rsi - 25) * (0.065 - 0.025) / (75 - 25)
    return sell_threshold, buy_threshold

# -----------------------------------------
# Simulate Strategy with RSI adjustment
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
    rsi = float(row["RSI"])

    # Get dynamic thresholds based on RSI
    sell_threshold, buy_threshold = get_dynamic_thresholds(rsi)

    # Raw triggers
    buy_price = prev_close * (1 - buy_threshold)
    sell_price = prev_close * (1 + sell_threshold)

    # Adjusted for MA
    buy_price_adj = min(buy_price, ma)
    sell_price_adj = max(sell_price, ma)

    # BUY (only if cash available)
    if day_low <= buy_price_adj and cash >= trade_amount:
        qty = trade_amount / buy_price_adj
        cash -= trade_amount
        shares += qty
        total_value = cash + shares * buy_price_adj
        trades.append([idx, "BUY (RSI-adjusted)", buy_price_adj, qty, cash, shares, total_value])

    # SELL
    if day_high >= sell_price_adj:
        qty = trade_amount / sell_price_adj
        if shares >= qty:
            cash += trade_amount
            shares -= qty
            total_value = cash + shares * sell_price_adj
            trades.append([idx, "SELL (RSI-adjusted)", sell_price_adj, qty, cash, shares, total_value])

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
