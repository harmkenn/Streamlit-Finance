import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.title("TQQQ Trend + ATR Trailing Stop vs Buy-and-Hold v2.3")

# --- Sidebar Controls ---
st.sidebar.header("Parameters")
period = st.sidebar.selectbox("Backtest period", ["3y", "5y", "10y"], index=1)
ma_len = st.sidebar.slider("Trend MA length", 20, 200, 100)
atr_len = st.sidebar.slider("ATR length", 5, 30, 14)
atr_mult_entry = st.sidebar.slider("Min trend strength (ATR/Close max)", 0.01, 0.10, 0.05, 0.005)
atr_mult_stop = st.sidebar.slider("Trailing stop (x ATR)", 1.0, 5.0, 2.0, 0.5)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.5, 5.0, 1.0, 0.5)
initial_cash = 100000

# --- Data Loading and Indicator Calculation ---
st.header("Data and Indicators")

try:
    # Download TQQQ data
    df = yf.download("TQQQ", period=period, interval="1d")
    
    if df.empty:
        st.error("No data downloaded for the selected period. Please try again.")
        st.stop()

    # Reset index to ensure Date is a column
    df = df.reset_index()
    df = df.set_index('Date')

    # Calculate True Range (TR) components
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    
    # Calculate True Range (TR) - take the maximum of the three components
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)

    # Calculate indicators
    df['ATR'] = df['TR'].rolling(window=atr_len, min_periods=1).mean()
    df[f'MA{ma_len}'] = df['Close'].rolling(window=ma_len, min_periods=1).mean()

    # Define strategy conditions
    df['TrendUp'] = (df['Close'] > df[f'MA{ma_len}']) & (df[f'MA{ma_len}'].diff() > 0)
    df['VolOK'] = (df['ATR'] / df['Close']) < atr_mult_entry

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
peak_price_for_stop = None  # Highest price observed during a trade
portfolio = []
trades = []

for date, row in df.iterrows():
    price = float(row['Close'])
    atr = float(row['ATR'])
    trend_ok = bool(row['TrendUp']) and bool(row['VolOK'])

    # --- 1. Check for Exit Signal ---
    exit_signal = False
    if shares > 0:
        if stop_price and price <= stop_price:
            exit_signal = True
        elif not trend_ok:
            exit_signal = True

    if exit_signal and shares > 0:
        proceeds = shares * price
        cash += proceeds
        trades.append([date, "SELL", price, shares, cash, 0.0])
        shares = 0.0
        # Reset trade-specific state
        stop_price = None
        peak_price_for_stop = None

    # --- 2. Check for Entry Signal ---
    if shares == 0 and trend_ok and atr > 0:
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
                
                # Set initial state for the new trade
                peak_price_for_stop = price
                stop_price = price - atr_mult_stop * atr

    # --- 3. Update Trailing Stop for Open Positions ---
    if shares > 0 and peak_price_for_stop is not None:
        # Update the peak price observed during this trade
        peak_price_for_stop = max(peak_price_for_stop, price)
        
        # Calculate the new potential stop price based on the peak
        new_stop_level = peak_price_for_stop - (atr_mult_stop * atr)
        
        # The stop price should only ever move up (for a long position)
        if stop_price is not None:
            stop_price = max(stop_price, new_stop_level)
        else:
            stop_price = new_stop_level

    # --- 4. Record Daily Portfolio Value ---
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
    
    # Trade statistics
    buy_trades = trade_df[trade_df['Type'] == 'BUY']
    sell_trades = trade_df[trade_df['Type'] == 'SELL']
    
    st.write(f"**Total Trades:** {len(trade_df)}")
    st.write(f"**Buy Trades:** {len(buy_trades)}")
    st.write(f"**Sell Trades:** {len(sell_trades)}")
else:
    st.write("No trades were made during this period.")

# Display price and indicator chart
st.subheader("TQQQ Close, MA, and ATR")
viz_df = df[["Close", f"MA{ma_len}", "ATR"]].copy()
st.line_chart(viz_df)

# Display strategy statistics
st.subheader("Strategy Statistics")
if len(port_df) > 1:
    strategy_returns = port_df['StrategyValue'].pct_change().dropna()
    buyhold_returns = port_df['BuyHoldValue'].pct_change().dropna()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Strategy Statistics:**")
        st.write(f"Total Return: {((final_strategy_value / initial_cash) - 1) * 100:.2f}%")
        st.write(f"Volatility: {strategy_returns.std() * np.sqrt(252) * 100:.2f}%")
        st.write(f"Max Drawdown: {((port_df['StrategyValue'] / port_df['StrategyValue'].cummax()) - 1).min() * 100:.2f}%")
    
    with col2:
        st.write("**Buy-and-Hold Statistics:**")
        st.write(f"Total Return: {((final_buyhold_value / initial_cash) - 1) * 100:.2f}%")
        st.write(f"Volatility: {buyhold_returns.std() * np.sqrt(252) * 100:.2f}%")
        st.write(f"Max Drawdown: {((port_df['BuyHoldValue'] / port_df['BuyHoldValue'].cummax()) - 1).min() * 100:.2f}%")
