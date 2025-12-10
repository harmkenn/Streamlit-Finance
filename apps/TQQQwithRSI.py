import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.title("TQQQ Momentum + Trend + Volatility Strategy vs Buy-and-Hold v2.5")

# --- Sidebar Controls ---
st.sidebar.header("Strategy Parameters")
period = st.sidebar.selectbox("Backtest period", ["3y", "5y", "10y"], index=1)
short_ma = st.sidebar.slider("Short MA (Trend)", 10, 50, 20)
long_ma = st.sidebar.slider("Long MA (Trend)", 50, 200, 100)
rsi_period = st.sidebar.slider("RSI Period", 5, 50, 14)
atr_period = st.sidebar.slider("ATR Period", 5, 50, 14)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.5, 5.0, 1.0, 0.5)
atr_mult_stop = st.sidebar.slider("ATR Trailing Stop (x ATR)", 1.0, 5.0, 2.0, 0.5)
initial_cash = 100000

# --- Data Loading ---
st.header("Data and Indicators")

try:
    # Download TQQQ data
    df = yf.download("TQQQ", period=period, interval="1d", auto_adjust=False)
    
    if df.empty:
        st.error("No data downloaded for the selected period. Please try again.")
        st.stop()

    # Reset index to ensure Date is a column
    df = df.reset_index()
    df = df.set_index('Date')

    # Calculate Indicators
    df[f"MA{short_ma}"] = df["Close"].rolling(window=short_ma).mean()
    df[f"MA{long_ma}"] = df["Close"].rolling(window=long_ma).mean()
    df["RSI"] = 100 - (100 / (1 + df["Close"].diff().clip(lower=0).rolling(rsi_period).mean() /
                              abs(df["Close"].diff().clip(upper=0).rolling(rsi_period).mean())))
    df["ATR"] = df["High"].sub(df["Low"]).rolling(window=atr_period).mean()

    # Drop rows with NaN values
    df = df.dropna()
    
    if df.empty:
        st.error("Not enough data to calculate indicators for the selected period. Try a longer period.")
        st.stop()

    st.success(f"Successfully loaded {len(df)} days of data for TQQQ")

except Exception as e:
    st.error(f"An error occurred during data download or processing: {e}")
    st.stop()

# --- Backtesting Engine ---
st.header("Running Backtest...")

# Initialize state variables
cash = float(initial_cash)
shares = 0.0
stop_price = None
portfolio = []
trades = []

for date, row in df.iterrows():
    price = float(row["Close"])
    atr = float(row["ATR"])
    rsi = float(row["RSI"])
    short_ma_val = float(row[f"MA{short_ma}"])
    long_ma_val = float(row[f"MA{long_ma}"])

    # --- 1. Check for Exit Signal ---
    exit_signal = False
    if shares > 0:
        if stop_price and price <= stop_price:
            exit_signal = True
        elif short_ma_val < long_ma_val:  # Exit if trend reverses
            exit_signal = True

    if exit_signal and shares > 0:
        proceeds = shares * price
        cash += proceeds
        trades.append([date, "SELL", price, shares, cash, 0.0])
        shares = 0.0
        stop_price = None

    # --- 2. Check for Entry Signal ---
    if shares == 0 and short_ma_val > long_ma_val and 30 <= rsi <= 50:
        stop_dist = atr_mult_stop * atr
        if stop_dist > 0:
            # Risk a percentage of current cash equity for the new trade
            dollar_risk = cash * (risk_pct / 100.0)
            qty = dollar_risk / stop_dist
            
            # Ensure we don't buy more than we can afford
            qty = min(qty, cash / price)

            if qty > 0 and cash >= qty * price:
                cost = qty * price
                cash -= cost
                shares += qty
                trades.append([date, "BUY", price, qty, cash, shares])
                stop_price = price - atr_mult_stop * atr

    # --- 3. Update Trailing Stop ---
    if shares > 0:
        stop_price = max(stop_price, price - atr_mult_stop * atr)

    # --- 4. Record Portfolio Value ---
    portfolio_value = cash + shares * price
    portfolio.append([date, portfolio_value])

# --- Results and Visualization ---
st.header("Backtest Results")

if not portfolio:
    st.warning("No portfolio data was recorded during this period.")
    st.stop()

# Create portfolio and buy-and-hold DataFrames
port_df = pd.DataFrame(portfolio, columns=["Date", "StrategyValue"])
port_df = port_df.set_index("Date")

# Calculate buy-and-hold performance
initial_shares_bh = initial_cash / df["Close"].iloc[0]
buy_hold_series = df["Close"] * initial_shares_bh

# Align the buy-and-hold series with the portfolio DataFrame
port_df["BuyHoldValue"] = buy_hold_series.reindex(port_df.index, method='ffill')

# Display final results
st.subheader("Final Values")
final_strategy_value = port_df['StrategyValue'].iloc[-1]
final_buyhold_value = port_df['BuyHoldValue'].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Strategy Final Value", f"${final_strategy_value:,.2f}")
col2.metric("Buy-and-Hold Final Value", f"${final_buyhold_value:,.2f}")

# Calculate performance difference
performance_diff = ((final_strategy_value - final_buyhold_value) / final_buyhold_value) * 100
col3.metric("Performance Difference", f"{performance_diff:+.2f}%")

# Display portfolio chart
st.subheader("Portfolio Value vs Buy-and-Hold")
st.line_chart(port_df)

# Display trades log
st.subheader("Trades Log")
if trades:
    trade_df = pd.DataFrame(trades, columns=["Date", "Type", "Price", "Shares", "CashAfter", "SharesAfter"])
    st.dataframe(trade_df.style.format({
        "Price": "${:.2f}", 
        "Shares": "{:.4f}", 
        "CashAfter": "${:,.2f}",
        "SharesAfter": "{:.4f}"
    }))
else:
    st.write("No trades were made during this period.")

# Display price and indicator chart
st.subheader("TQQQ Close, MA, and ATR")
viz_df = df[["Close", f"MA{short_ma}", f"MA{long_ma}", "ATR"]].copy()
st.line_chart(viz_df)
