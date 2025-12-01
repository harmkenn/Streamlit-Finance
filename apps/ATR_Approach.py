import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime

# --- Performance Metrics Functions ---

def calculate_mdd(series):
    """Calculates Maximum Drawdown (%) of a value series."""
    peak = series.expanding(min_periods=1).max()
    drawdown = (series / peak) - 1
    # Returns a Series containing the minimum value; this needs to be cast to float later.
    return drawdown.min() 

def calculate_cagr(final_value, initial_value, years):
    """Calculates Compounded Annual Growth Rate (%)."""
    if years <= 0:
        return 0.0
    # Ensure no division by zero if initial_value is zero, though unlikely with cash.
    if initial_value == 0:
        return 0.0
    return (final_value / initial_value)**(1 / years) - 1

def calculate_sharpe_ratio(returns, risk_free_rate=0.015):
    """Calculates Sharpe Ratio (Annualized)."""
    if returns.empty:
        return 0.0
    
    # Calculate daily excess returns
    # Assuming 252 trading days per year
    excess_returns = returns - (risk_free_rate / 252)
    
    # Annualize based on 252 trading days
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


# ---------------------------------------------------------
# Streamlit Configuration and Parameters
# ---------------------------------------------------------

st.set_page_config(layout="wide")
st.title("TQQQ ATR-Based Dip & Rip Strategy Backtester 📈")
ticker = "TQQQ"

# --- Sidebar for Parameters ---
st.sidebar.header("Strategy Parameters")

# Strategy inputs
initial_cash = st.sidebar.number_input("Initial Cash ($)", value=100000, step=10000)
trade_amount = st.sidebar.number_input("Trade Amount ($)", value=10000, step=1000)

st.sidebar.markdown("---")

# ATR inputs
atr_period = st.sidebar.slider("ATR Period", 5, 30, 14)
atr_mult_buy = st.sidebar.slider("ATR Buy Multiplier", 0.1, 3.0, 0.5, 0.1)
atr_mult_sell = st.sidebar.slider("ATR Sell Multiplier", 0.1, 3.0, 1.5, 0.1)

st.sidebar.markdown("---")

# Date range selection
today = datetime.date.today()
default_start_date = today - datetime.timedelta(days=3 * 365) # 3 years ago
data_start_date = st.sidebar.date_input("Start Date", default_start_date)
data_end_date = st.sidebar.date_input("End Date", today)

st.write(f"""
Strategy rules for **{ticker}** ({data_start_date} to {data_end_date}):

- **Starting Cash:** **${initial_cash:,.0f}**
- **Trade Size:** **${trade_amount:,.0f}** per trade
- **ATR Period:** **{atr_period}**
- **BUY Rule (Dip):** Execute a BUY trade if the day's $\\text{{Low}}$ drops to or below $\\text{{PrevClose}} - \\text{{ATR}} \\times \\mathbf{{{atr_mult_buy}}}$
- **SELL Rule (Rip):** Execute a SELL trade if the day's $\\text{{High}}$ rises to or above $\\text{{PrevClose}} + \\text{{ATR}} \\times \\mathbf{{{atr_mult_sell}}}$
- **Execution Price:** **Day's Closing Price** (Conservative backtest assumption)
""")

# ---------------------------------------------------------
# Load Data & Compute ATR
# ---------------------------------------------------------

try:
    df = yf.download(ticker, start=data_start_date, end=data_end_date, interval="1d")
except Exception as e:
    st.error(f"Error loading data for {ticker}: {e}")
    st.stop()

if df.empty:
    st.error(f"Error: No data returned for {ticker} between {data_start_date} and {data_end_date}.")
    st.stop()

df = df.astype(float)
df["PrevClose"] = df["Close"].shift(1)

# Compute True Range (TR)
high_low = df["High"] - df["Low"]
high_pc = (df["High"] - df["PrevClose"]).abs()
low_pc = (df["Low"] - df["PrevClose"]).abs()

tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
df["ATR"] = tr.rolling(atr_period).mean()

df = df.dropna()

# ---------------------------------------------------------
# ATR Trigger Preview (Next Day)
# ---------------------------------------------------------

st.subheader("Next-Day ATR Trigger Preview (Based on last close)")
if not df.empty:
    last_close = float(df["Close"].iloc[-1])
    last_atr = float(df["ATR"].iloc[-1])

    buy_trigger = last_close - last_atr * atr_mult_buy
    sell_trigger = last_close + last_atr * atr_mult_sell

    col1, col2 = st.columns(2)
    col1.metric("Previous Close", f"${last_close:,.2f}")
    col2.metric(f"ATR({atr_period})", f"${last_atr:,.2f}")

    preview_df = pd.DataFrame({
        "Trigger Type": ["Buy Dip Trigger", "Sell Rip Trigger"],
        "Target Price": [buy_trigger, sell_trigger],
        "Price Change from Close": [buy_trigger - last_close, sell_trigger - last_close],
    })

    st.dataframe(preview_df.style.format({
        "Target Price": "${:,.2f}",
        "Price Change from Close": "{:,.2f}",
    }), use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# Simulate ATR Strategy
# ---------------------------------------------------------

cash = float(initial_cash)
shares = 0.0
trades = []
portfolio_value_over_time = []
num_trades = 0

for idx, row in df.iterrows():
    prev_close = float(row["PrevClose"])
    day_low = float(row["Low"])
    day_high = float(row["High"])
    day_close = float(row["Close"])
    atr = float(row["ATR"])
    
    # ATR triggers based on previous close
    buy_trigger = prev_close - atr * atr_mult_buy
    sell_trigger = prev_close + atr * atr_mult_sell

    # Execution price is set to the conservative Day Close
    execution_price = day_close 

    # 1. BUY ATR (Dip)
    if day_low <= buy_trigger and cash >= trade_amount:
        # Calculate shares based on fixed trade_amount and conservative execution_price
        qty = trade_amount / execution_price
        cash -= trade_amount
        shares += qty
        total_value = cash + shares * execution_price
        trades.append([idx, "BUY ATR (Dip)", execution_price, qty, cash, shares, total_value])
        num_trades += 1

    # 2. SELL ATR (Rip)
    if day_high >= sell_trigger:
        # Calculate shares to sell based on fixed trade_amount and conservative execution_price
        qty = trade_amount / execution_price
        if shares >= qty:
            cash += trade_amount
            shares -= qty
            total_value = cash + shares * execution_price
            trades.append([idx, "SELL ATR (Rip)", execution_price, qty, cash, shares, total_value])
            num_trades += 1

    # Record daily portfolio value at the closing price
    portfolio_value_over_time.append([idx, cash + shares * day_close])

# Final values
final_close = float(df["Close"].iloc[-1])
final_value = cash + shares * final_close

# ---------------------------------------------------------
# Buy-and-Hold Comparison & Performance Calculation
# ---------------------------------------------------------

initial_shares = float(initial_cash / df["Close"].iloc[0])
buy_hold_value = df["Close"] * initial_shares

# Create unified portfolio DataFrame
portfolio_df = pd.DataFrame(portfolio_value_over_time, columns=["Date", "StrategyValue"])
portfolio_df.set_index("Date", inplace=True)
portfolio_df["BuyHoldValue"] = buy_hold_value.values

# Calculate returns for Sharpe Ratio
strategy_returns = portfolio_df["StrategyValue"].pct_change().dropna()
buyhold_returns = portfolio_df["BuyHoldValue"].pct_change().dropna()

# Time period in years
years = (df.index[-1] - df.index[0]).days / 365.25
if years < 0.1:
    st.warning("The selected date range is very short. Annualized metrics may not be meaningful.")
    years = 1.0 


# Calculate metrics and fix TypeError by casting results to float
strat_cagr = calculate_cagr(final_value, initial_cash, years)
strat_mdd = float(calculate_mdd(portfolio_df["StrategyValue"])) # FIX APPLIED HERE
strat_sharpe = calculate_sharpe_ratio(strategy_returns)

bh_final_value = float(buy_hold_value.iloc[-1])
bh_cagr = calculate_cagr(bh_final_value, initial_cash, years)
bh_mdd = float(calculate_mdd(buy_hold_value)) # FIX APPLIED HERE
bh_sharpe = calculate_sharpe_ratio(buyhold_returns)

# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------

st.subheader("Performance Summary 📊")

performance_summary = pd.DataFrame({
    "Metric": ["Final Value", "CAGR (Annualized)", "Max Drawdown", "Sharpe Ratio (Annualized)", "Total Trades"],
    "ATR Strategy": [
        f"${final_value:,.2f}", 
        f"{strat_cagr:.2%}", 
        f"{strat_mdd:.2%}", # Formatting now works
        f"{strat_sharpe:,.2f}",
        num_trades
    ],
    "Buy-and-Hold": [
        f"${bh_final_value:,.2f}", 
        f"{bh_cagr:.2%}", 
        f"{bh_mdd:.2%}", # Formatting now works
        f"{bh_sharpe:,.2f}",
        "N/A"
    ]
})

st.table(performance_summary.set_index("Metric"))

st.markdown("---")

# --- Charts ---
col_charts = st.columns(2)

with col_charts[0]:
    st.subheader("Portfolio Value vs Buy-and-Hold")
    st.line_chart(portfolio_df, use_container_width=True)

with col_charts[1]:
    st.subheader(f"{ticker} Closing Price")
    st.line_chart(df["Close"], use_container_width=True)

st.markdown("---")

# Trade log
trades_df = pd.DataFrame(
    trades,
    columns=["Date", "Type", "Execution Price", "Shares", "Cash After", "Shares After", "Total Value"]
)

st.subheader("Trade Log")
st.dataframe(trades_df.style.format({
    "Execution Price": "${:,.2f}",
    "Shares": "{:,.4f}",
    "Cash After": "${:,.2f}",
    "Shares After": "{:,.4f}",
    "Total Value": "${:,.2f}"
}), use_container_width=True)