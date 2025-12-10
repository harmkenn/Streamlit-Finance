import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.title("Enhanced TQQQ Multi-Signal Strategy vs Buy-and-Hold v2.7")

# -----------------------------------------
# Strategy parameters with sliders
# -----------------------------------------
initial_cash = 100000
trade_amount = 10000

st.sidebar.header("Strategy Parameters")

# Original trigger parameters
drop_pct_5 = st.sidebar.slider("Buy Trigger Drop % (5% default)", 1, 20, 4) / 100
spike_pct_5 = st.sidebar.slider("Sell Trigger Rise % (5% default)", 1, 20, 7) / 100
drop_pct_10 = st.sidebar.slider("Buy Trigger Drop % (10% default)", 1, 30, 8) / 100
spike_pct_10 = st.sidebar.slider("Sell Trigger Rise % (10% default)", 1, 30, 14) / 100

# Enhanced parameters
ma_period = st.sidebar.slider("Moving Average Period", 5, 50, 7)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
volume_ma_period = st.sidebar.slider("Volume MA Period", 5, 30, 10)
volatility_period = st.sidebar.slider("Volatility Period", 5, 30, 10)

# Risk management
max_position_pct = st.sidebar.slider("Max Position % of Portfolio", 10, 100, 80) / 100
enable_volume_filter = st.sidebar.checkbox("Enable Volume Filter", value=True)
enable_volatility_filter = st.sidebar.checkbox("Enable Volatility Filter", value=True)
enable_rsi_filter = st.sidebar.checkbox("Enable RSI Filter", value=True)

st.write(f"""
Enhanced strategy features:

- Start with **${initial_cash:,}** cash  
- Buy/Sell **${trade_amount:,}** on **{int(drop_pct_5*100)}% or {int(drop_pct_10*100)}% intraday moves**  
- Apply **MA{ma_period} filter** + **RSI{rsi_period}** + **Volume** + **Volatility** filters
- Maximum position: **{int(max_position_pct*100)}%** of portfolio
""")

# -----------------------------------------
# Load 5 years of TQQQ daily data for better backtesting
# -----------------------------------------
ticker = "TQQQ"
df = yf.download(ticker, period="5y", interval="1d", auto_adjust=False)

if df.empty:
    st.error("Error: No data returned.")
    st.stop()

df = df.astype(float)
df["PrevClose"] = df["Close"].shift(1)

# Calculate indicators
df[f"MA{ma_period}"] = df["Close"].rolling(ma_period).mean()

# RSI calculation
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df["RSI"] = calculate_rsi(df["Close"], rsi_period)

# Volume indicators
df[f"Volume_MA{volume_ma_period}"] = df["Volume"].rolling(volume_ma_period).mean()
df["Volume_Ratio"] = df["Volume"] / df[f"Volume_MA{volume_ma_period}"]  # Correct calculation

# Volatility indicator (ATR-like)
df["High_Low"] = df["High"] - df["Low"]
df["High_PrevClose"] = abs(df["High"] - df["PrevClose"])
df["Low_PrevClose"] = abs(df["Low"] - df["PrevClose"])
df["True_Range"] = df[["High_Low", "High_PrevClose", "Low_PrevClose"]].max(axis=1)
df["ATR"] = df["True_Range"].rolling(volatility_period).mean()
df["Volatility_Ratio"] = df["ATR"] / df["Close"]

# Market regime detection
df["MA_Slope"] = df[f"MA{ma_period}"].diff(5)  # 5-day slope
df["Trend_Strength"] = df["MA_Slope"] / df["Close"]

df = df.dropna()

# -----------------------------------------
# Enhanced signal generation
# -----------------------------------------
def get_signal_strength(row, drop_pct_5, spike_pct_5, drop_pct_10, spike_pct_10, 
                       enable_rsi_filter, enable_volume_filter, enable_volatility_filter):
    """Calculate signal strength based on multiple factors"""
    
    prev_close = row["PrevClose"]
    day_low = row["Low"]
    day_high = row["High"]
    ma = row[f"MA{ma_period}"]
    rsi = row["RSI"]
    volume_ratio = row["Volume_Ratio"]
    volatility_ratio = row["Volatility_Ratio"]
    trend_strength = row["Trend_Strength"]
    
    # Base triggers
    buy_5 = prev_close * (1 - drop_pct_5)
    sell_5 = prev_close * (1 + spike_pct_5)
    buy_10 = prev_close * (1 - drop_pct_10)
    sell_10 = prev_close * (1 + spike_pct_10)
    
    # MA adjusted triggers
    buy_5_adj = min(buy_5, ma)
    sell_5_adj = max(sell_5, ma)
    buy_10_adj = min(buy_10, ma)
    sell_10_adj = max(sell_10, ma)
    
    signals = []
    
    # Check for buy signals
    if day_low <= buy_5_adj:
        signal_strength = 1.0
        
        # RSI filter: prefer buying when RSI is oversold but not extremely oversold
        if enable_rsi_filter:
            if 25 <= rsi <= 45:
                signal_strength *= 1.2  # Boost signal
            elif rsi < 20:
                signal_strength *= 0.5  # Reduce signal (too oversold)
            elif rsi > 60:
                signal_strength *= 0.3  # Reduce signal (not oversold enough)
        
        # Volume filter: prefer high volume moves
        if enable_volume_filter and volume_ratio > 1.2:
            signal_strength *= 1.1
        
        # Volatility filter: avoid extremely volatile periods
        if enable_volatility_filter and volatility_ratio > 0.05:
            signal_strength *= 0.7
        
        # Trend filter: prefer buying in uptrends
        if trend_strength > 0:
            signal_strength *= 1.1
        
        signals.append(("BUY_5", buy_5_adj, signal_strength))
    
    if day_low <= buy_10_adj:
        signal_strength = 1.5  # Stronger signal for larger drops
        
        if enable_rsi_filter:
            if 20 <= rsi <= 40:
                signal_strength *= 1.3
            elif rsi < 15:
                signal_strength *= 0.6
            elif rsi > 55:
                signal_strength *= 0.4
        
        if enable_volume_filter and volume_ratio > 1.3:
            signal_strength *= 1.2
        
        if enable_volatility_filter and volatility_ratio > 0.06:
            signal_strength *= 0.6
        
        if trend_strength > 0:
            signal_strength *= 1.2
        
        signals.append(("BUY_10", buy_10_adj, signal_strength))
    
    # Check for sell signals
    if day_high >= sell_5_adj:
        signal_strength = 1.0
        
        if enable_rsi_filter:
            if 55 <= rsi <= 75:
                signal_strength *= 1.2
            elif rsi > 80:
                signal_strength *= 1.5  # Strong sell signal when very overbought
            elif rsi < 50:
                signal_strength *= 0.5
        
        if enable_volume_filter and volume_ratio > 1.2:
            signal_strength *= 1.1
        
        if trend_strength < 0:
            signal_strength *= 1.2  # Sell more aggressively in downtrends
        
        signals.append(("SELL_5", sell_5_adj, signal_strength))
    
    if day_high >= sell_10_adj:
        signal_strength = 1.5
        
        if enable_rsi_filter:
            if 60 <= rsi <= 80:
                signal_strength *= 1.3
            elif rsi > 85:
                signal_strength *= 1.8
            elif rsi < 55:
                signal_strength *= 0.6
        
        if enable_volume_filter and volume_ratio > 1.3:
            signal_strength *= 1.2
        
        if trend_strength < 0:
            signal_strength *= 1.3
        
        signals.append(("SELL_10", sell_10_adj, signal_strength))
    
    return signals

# -----------------------------------------
# Simulate Enhanced Strategy
# -----------------------------------------
cash = float(initial_cash)
shares = 0.0
trades = []
portfolio_value_over_time = []

for idx, row in df.iterrows():
    day_close = float(row["Close"])
    
    # Get all signals for this day
    signals = get_signal_strength(row, drop_pct_5, spike_pct_5, drop_pct_10, spike_pct_10,
                                 enable_rsi_filter, enable_volume_filter, enable_volatility_filter)
    
    # Process signals in order of strength
    signals.sort(key=lambda x: x[2], reverse=True)
    
    current_portfolio_value = cash + shares * day_close
    max_position_value = current_portfolio_value * max_position_pct
    
    for signal_type, price, strength in signals:
        if "BUY" in signal_type and cash >= trade_amount:
            # Adjust trade amount based on signal strength
            adjusted_trade_amount = min(trade_amount * strength, cash)
            
            # Respect maximum position limit
            potential_shares_value = (shares * day_close) + adjusted_trade_amount
            if potential_shares_value <= max_position_value:
                qty = adjusted_trade_amount / price
                cash -= adjusted_trade_amount
                shares += qty
                trades.append([idx, f"{signal_type} (Enhanced)", price, qty, cash, shares, 
                             cash + shares * price, f"Strength: {strength:.2f}"])
        
        elif "SELL" in signal_type and shares > 0:
            # Adjust trade amount based on signal strength
            adjusted_trade_amount = min(trade_amount * strength, shares * price)
            qty = adjusted_trade_amount / price
            
            if shares >= qty:
                cash += adjusted_trade_amount
                shares -= qty
                trades.append([idx, f"{signal_type} (Enhanced)", price, qty, cash, shares,
                             cash + shares * price, f"Strength: {strength:.2f}"])
    
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
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Strategy Final Value", f"${final_value:,.2f}")

with col2:
    st.metric("Buy-and-Hold Final Value", f"${float(buy_hold_value.iloc[-1]):,.2f}")

with col3:
    performance_diff = ((final_value - float(buy_hold_value.iloc[-1])) / float(buy_hold_value.iloc[-1])) * 100
    st.metric("Performance Difference", f"{performance_diff:+.2f}%")

# Trade log
trades_df = pd.DataFrame(
    trades,
    columns=["Date", "Type", "Execution Price", "Shares", "CashAfter", "SharesAfter", "TotalValue", "Signal Strength"]
)
st.subheader("Trade Log")
st.dataframe(trades_df)

# Portfolio curves
portfolio_df = pd.DataFrame(portfolio_value_over_time, columns=["Date", "StrategyValue"])
portfolio_df.set_index("Date", inplace=True)
portfolio_df["BuyHoldValue"] = buy_hold_value.values

st.subheader("Portfolio Value vs Buy-and-Hold")
st.line_chart(portfolio_df)

# Additional charts
st.subheader("TQQQ Price and Indicators")
chart_df = df[["Close", f"MA{ma_period}", "RSI"]].copy()
st.line_chart(chart_df)

# Strategy statistics
st.subheader("Strategy Statistics")
if len(trades_df) > 0:
    buy_trades = trades_df[trades_df['Type'].str.contains('BUY')]
    sell_trades = trades_df[trades_df['Type'].str.contains('SELL')]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Total Trades:** {len(trades_df)}")
        st.write(f"**Buy Trades:** {len(buy_trades)}")
        st.write(f"**Sell Trades:** {len(sell_trades)}")
    
    with col2:
        if len(portfolio_df) > 1:
            returns = portfolio_df['StrategyValue'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            max_dd = ((portfolio_df['StrategyValue'] / portfolio_df['StrategyValue'].cummax()) - 1).min() * 100
            
            st.write(f"**Annual Volatility:** {volatility:.2f}%")
            st.write(f"**Max Drawdown:** {max_dd:.2f}%")
            st.write(f"**Total Return:** {((final_value / initial_cash) - 1) * 100:.2f}%")
