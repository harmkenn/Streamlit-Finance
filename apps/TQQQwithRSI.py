import pandas as pd
import yfinance as yf
import streamlit as st

st.title("TQQQ Trend and Volatility-Based Strategy vs Buy-and-Hold")

# -----------------------------------------
# Strategy parameters
# -----------------------------------------
initial_cash = 100000
trade_amount = 10000

st.sidebar.header("Strategy Parameters")
short_ma_period = st.sidebar.slider("Short MA Period", 5, 50, 10)
long_ma_period = st.sidebar.slider("Long MA Period", 20, 200, 50)
atr_period = st.sidebar.slider("ATR Period", 5, 50, 14)
rsi_period = st.sidebar.slider("RSI Period", 5, 50, 14)

st.write(f"""
Simulating a strategy where:

- Start with **${initial_cash:,}** cash  
- Buy/Sell **${trade_amount:,}** based on trend and volatility  
- Use **MA{short_ma_period} and MA{long_ma_period}** for trend detection  
- Use **ATR{atr_period}** for volatility adjustment  
- Use **RSI{rsi_period}** for confirmation  
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

# -----------------------------------------
# Calculate Indicators
# -----------------------------------------
# Moving Averages
df[f"MA{short_ma_period}"] = df["Close"].rolling(short_ma_period).mean()
df[f"MA{long_ma_period}"] = df["Close"].rolling(long_ma_period).mean()

# ATR (Average True Range)
df["TR"] = df[["High", "Low", "Close"]].apply(
    lambda row: max(row["High"] - row["Low"], abs(row["High"] - row["Close"]), abs(row["Low"] - row["Close"])),
    axis=1
)
df["ATR"] = df["TR"].rolling(atr_period).mean()

# RSI
def calculate_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    return data

df = calculate_rsi(df, rsi_period)
df = df.dropna()

# -----------------------------------------
# Simulate Strategy
# -----------------------------------------
cash = float(initial_cash)
shares = 0.0
trades = []
portfolio_value_over_time = []

for idx, row in df.iterrows():
    short_ma = row[f"MA{short_ma_period}"]
    long_ma = row[f"MA{long_ma_period}"]
    atr = row["ATR"]
    rsi = row["RSI"]
    day_close = row["Close"]

    # Trend detection
    bullish_crossover = short_ma > long_ma
    bearish_crossover = short_ma < long_ma

    # RSI confirmation
    overbought = rsi > 70
    oversold = rsi < 30

    # BUY signal
    if bullish_crossover and not overbought and cash >= trade_amount:
        qty = trade_amount / day_close
        cash -= trade_amount
        shares += qty
        total_value = cash + shares * day_close
        trades.append([idx, "BUY", day_close, qty, cash, shares, total_value])

    # SELL signal
    if bearish_crossover and not oversold and shares > 0:
        qty = trade_amount / day_close
        if shares >= qty:
            cash += trade_amount
            shares -= qty
            total_value = cash + shares * day_close
            trades.append([idx, "SELL", day_close, qty, cash, shares, total_value])

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
