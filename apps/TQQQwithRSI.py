import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.title("TQQQ Trend + ATR Trailing Stop vs Buy-and-Hold v2.2")

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
    # Corrected yf.download call with 'tickers' argument
    df = yf.download(tickers="TQQQ", period=period, interval="1d")
    
    if df.empty:
        st.error("No data downloaded for the selected period. Please try again.")
        st.stop()

    # Calculate True Range (TR)
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)

    # Calculate indicators
    df['ATR'] = df['TR'].rolling(atr_len, min_periods=1).mean()
    df[f'MA{ma_len}'] = df['Close'].rolling(ma_len, min_periods=1).mean()

    # Define strategy conditions
    df['TrendUp'] = (df['Close'] > df[f'MA{ma_len}']) & (df[f'MA{ma_len}'].diff() > 0)
    df['VolOK'] = (df['ATR'] / df['Close']) < atr_mult_entry

    df = df.dropna()
    if df.empty:
        st.error("Not enough data to calculate indicators for the selected period. Try a longer period.")
        st.stop()

except Exception as e:
    st.error(f"An error occurred during data download or processing: {e}")
    st.stop()


# --- Backtesting Engine ---

# Initialize state variables
cash = float(initial_cash)
shares = 0.0
stop_price = None
peak_price_for_stop = None # Highest price observed during a trade
portfolio = []
trades = []

for date, row in df.iterrows():
    price = row['Close']
    atr = row['ATR']
    trend_ok = row['TrendUp'] and row['VolOK']

    # --- 1. Check for Exit Signal ---
    exit_signal = False
    if shares > 0:
        if stop_price and price <= stop_price:
            exit_signal = True
        elif not trend_ok:
            exit_signal = True

    if exit_signal:
        proceeds = shares * price
        cash += proceeds
        trades.append([date, "SELL", price, shares, cash, 0.0])
        shares = 0.0
        # Reset trade-specific state
        stop_price = None
        peak_price_for_stop = None

    # --- 2. Check for Entry Signal ---
    if shares == 0 and trend_ok:
        stop_dist = atr_mult_stop * atr
        if stop_dist > 0:
            # Risk a percentage of current cash equity for the new trade
            dollar_risk = cash * (risk_pct / 100.0)
            qty = dollar_risk / stop_dist
            
            # Ensure we don't buy more than we can afford
            qty = min(qty, cash / price)

            if qty > 0:
                cost = qty * price
                cash -= cost
                shares += qty
                trades.append([date, "BUY", price, qty, cash, shares])
                
                # Set initial state for the new trade
                peak_price_for_stop = price
                stop_price = price - atr_mult_stop * atr

    # --- 3. Update Trailing Stop for Open Positions ---
    if shares > 0:
        # Update the peak price observed during this trade
        peak_price_for_stop = max(peak_price_for_stop, price)
        
        # Calculate the new potential stop price based on the peak
        new_stop_level = peak_price_for_stop - (atr_mult_stop * atr)
        
        # The stop price should only ever move up (for a long position)
        stop_price = max(stop_price, new_stop_level)

    # --- 4. Record Daily Portfolio Value ---
    portfolio.append([date, cash + shares * price])


# --- Results and Visualization ---
st.header("Backtest Results")

if not portfolio:
    st.warning("No trades were executed during this period with the selected parameters.")
    st.stop()

# Create portfolio and buy-and-hold DataFrames
port_df = pd.DataFrame(portfolio, columns=["Date", "StrategyValue"]).set_index("Date")
initial_shares = initial_cash / df["Close"].iloc[0]
buy_hold_series = df["Close"] * initial_shares
port_df["BuyHoldValue"] = buy_hold_series.reindex(port_df.index)

# Display final results
st.subheader("Final Values")
col1, col2 = st.columns(2)
col1.metric("Strategy Final Value", f"${port_df['StrategyValue'].iloc[-1]:,.2f}")
col2.metric("Buy-and-Hold Final Value", f"${port_df['BuyHoldValue'].iloc[-1]:,.2f}")

st.subheader("Portfolio Value vs Buy-and-Hold")
st.line_chart(port_df)

st.subheader("Trades Log")
if trades:
    trade_df = pd.DataFrame(trades, columns=["Date", "Type", "Price", "Shares", "CashAfter", "SharesAfter"])
    st.dataframe(trade_df.style.format({"Price": "{:.2f}", "Shares": "{:.4f}", "CashAfter": "{:,.2f}"}))
else:
    st.write("No trades were made.")

st.subheader("TQQQ Close, MA, and ATR")
viz_df = df[["Close", f"MA{ma_len}", "ATR"]].copy()
st.line_chart(viz_df)
