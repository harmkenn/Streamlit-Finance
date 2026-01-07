import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
#v1.1
st.set_page_config(page_title="TQQQ Trading Analyzer", layout="wide")

st.title("🎯 TQQQ Buy/Sell Target Price Analyzer")
st.caption("⚠️ For educational purposes only. Not financial advice.")

# Sidebar parameters
st.sidebar.header("Strategy Parameters")
lookback_period = st.sidebar.slider("Analysis Period (months)", 6, 36, 12)
rsi_oversold = st.sidebar.slider("RSI Oversold (Buy Signal)", 20, 40, 30)
rsi_overbought = st.sidebar.slider("RSI Overbought (Sell Signal)", 60, 80, 70)
ma_short = st.sidebar.slider("Short Moving Average (days)", 5, 20, 10)
ma_long = st.sidebar.slider("Long Moving Average (days)", 20, 60, 30)

# Fetch TQQQ data
@st.cache_data(ttl=3600)
def fetch_data(months):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months*30)
    ticker = yf.Ticker("TQQQ")
    data = ticker.history(start=start_date, end=end_date)
    return data

# Calculate technical indicators
def calculate_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['MA_Short'] = df['Close'].rolling(window=ma_short).mean()
    df['MA_Long'] = df['Close'].rolling(window=ma_long).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

# Generate signals
def generate_signals(df, rsi_low, rsi_high):
    df['Signal'] = 0
    
    # Buy signals: RSI oversold AND price near lower Bollinger Band
    buy_condition = (df['RSI'] < rsi_low) & (df['Close'] < df['BB_Lower'] * 1.02)
    df.loc[buy_condition, 'Signal'] = 1
    
    # Sell signals: RSI overbought AND price near upper Bollinger Band
    sell_condition = (df['RSI'] > rsi_high) & (df['Close'] > df['BB_Upper'] * 0.98)
    df.loc[sell_condition, 'Signal'] = -1
    
    return df

# Backtest strategy
def backtest_strategy(df):
    position = 0
    trades = []
    entry_price = 0
    
    for i in range(len(df)):
        if df['Signal'].iloc[i] == 1 and position == 0:  # Buy signal
            position = 1
            entry_price = df['Close'].iloc[i]
            trades.append({
                'Date': df.index[i],
                'Type': 'BUY',
                'Price': entry_price,
                'Return': 0
            })
        elif df['Signal'].iloc[i] == -1 and position == 1:  # Sell signal
            position = 0
            exit_price = df['Close'].iloc[i]
            ret = ((exit_price - entry_price) / entry_price) * 100
            trades.append({
                'Date': df.index[i],
                'Type': 'SELL',
                'Price': exit_price,
                'Return': ret
            })
    
    return pd.DataFrame(trades)

# Portfolio simulation
def simulate_portfolio(df, initial_capital=100000):
    cash = initial_capital
    shares = 0
    portfolio_value = []
    
    for i in range(len(df)):
        # Calculate current portfolio value
        current_value = cash + (shares * df['Close'].iloc[i])
        portfolio_value.append({
            'Date': df.index[i],
            'Value': current_value,
            'Cash': cash,
            'Shares': shares,
            'Price': df['Close'].iloc[i]
        })
        
        # Execute trades
        if df['Signal'].iloc[i] == 1 and shares == 0:  # Buy signal
            shares = cash / df['Close'].iloc[i]
            cash = 0
        elif df['Signal'].iloc[i] == -1 and shares > 0:  # Sell signal
            cash = shares * df['Close'].iloc[i]
            shares = 0
    
    portfolio_df = pd.DataFrame(portfolio_value)
    return portfolio_df, cash, shares

# Main app
try:
    with st.spinner("Fetching TQQQ data..."):
        data = fetch_data(lookback_period)
    
    if data.empty:
        st.error("Unable to fetch data. Please try again.")
    else:
        # Calculate indicators
        data = calculate_indicators(data)
        data = generate_signals(data, rsi_oversold, rsi_overbought)
        
        # Current metrics
        current_price = data['Close'].iloc[-1]
        current_rsi = data['RSI'].iloc[-1]
        current_bb_lower = data['BB_Lower'].iloc[-1]
        current_bb_upper = data['BB_Upper'].iloc[-1]
        
        st.header("📊 Current Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            st.metric("RSI", f"{current_rsi:.1f}")
        with col3:
            st.metric("Buy Target", f"${current_bb_lower:.2f}", 
                     delta=f"{((current_bb_lower - current_price) / current_price * 100):.1f}%")
        with col4:
            st.metric("Sell Target", f"${current_bb_upper:.2f}",
                     delta=f"{((current_bb_upper - current_price) / current_price * 100):.1f}%")
        
        # Trading recommendation
        st.subheader("🎲 Current Recommendation")
        if current_rsi < rsi_oversold and current_price < current_bb_lower * 1.02:
            st.success(f"🟢 **BUY SIGNAL** - Price near lower band (${current_bb_lower:.2f}) with RSI oversold ({current_rsi:.1f})")
        elif current_rsi > rsi_overbought and current_price > current_bb_upper * 0.98:
            st.warning(f"🔴 **SELL SIGNAL** - Price near upper band (${current_bb_upper:.2f}) with RSI overbought ({current_rsi:.1f})")
        else:
            st.info(f"⚪ **HOLD** - Wait for clearer signal (RSI: {current_rsi:.1f})")
        
        # Price chart
        st.subheader("📈 Price Chart with Indicators")
        chart_data = data[['Close', 'MA_Short', 'MA_Long', 'BB_Upper', 'BB_Lower']].tail(90)
        st.line_chart(chart_data)
        
        # RSI chart
        st.subheader("📉 RSI Indicator")
        rsi_chart = data[['RSI']].tail(90)
        st.line_chart(rsi_chart)
        st.caption(f"Buy when RSI < {rsi_oversold} | Sell when RSI > {rsi_overbought}")
        
        # Backtest results
        st.subheader("🔬 Strategy Backtest")
        trades_df = backtest_strategy(data)
        
        if not trades_df.empty:
            completed_trades = trades_df[trades_df['Type'] == 'SELL']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Trades", f"{len(completed_trades)}")
            with col2:
                avg_return = completed_trades['Return'].mean()
                st.metric("Avg Return per Trade", f"{avg_return:.2f}%")
            with col3:
                win_rate = (completed_trades['Return'] > 0).sum() / len(completed_trades) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            # Trades per year
            days_analyzed = (data.index[-1] - data.index[0]).days
            trades_per_year = (len(completed_trades) / days_analyzed) * 365
            st.info(f"📅 Historical trading frequency: ~{trades_per_year:.0f} trades per year")
            
            # Show recent trades
            with st.expander("View Recent Trades"):
                st.dataframe(trades_df.tail(20).style.format({
                    'Price': '${:.2f}',
                    'Return': '{:.2f}%'
                }))
        else:
            st.warning("No completed trades found in backtest period. Try adjusting parameters.")
        
        # Portfolio simulation
        st.subheader("💰 Portfolio Simulation - 5 Year Performance")
        
        # Fetch 5 year data for simulation
        data_5yr = fetch_data(60)  # 60 months = 5 years
        data_5yr = calculate_indicators(data_5yr)
        data_5yr = generate_signals(data_5yr, rsi_oversold, rsi_overbought)
        
        initial_capital = 100000
        portfolio_df, final_cash, final_shares = simulate_portfolio(data_5yr, initial_capital)
        
        # Calculate final values
        final_price = data_5yr['Close'].iloc[-1]
        final_portfolio_value = final_cash + (final_shares * final_price)
        total_return = ((final_portfolio_value - initial_capital) / initial_capital) * 100
        years = len(data_5yr) / 252  # Trading days in a year
        annualized_return = ((final_portfolio_value / initial_capital) ** (1 / years) - 1) * 100
        
        # Buy and hold comparison
        buy_hold_shares = initial_capital / data_5yr['Close'].iloc[0]
        buy_hold_value = buy_hold_shares * final_price
        buy_hold_return = ((buy_hold_value - initial_capital) / initial_capital) * 100
        
        st.markdown(f"**Starting Capital:** ${initial_capital:,.0f} invested on {data_5yr.index[0].strftime('%Y-%m-%d')}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Value", f"${final_portfolio_value:,.0f}", 
                     delta=f"${final_portfolio_value - initial_capital:,.0f}")
        with col2:
            st.metric("Total Return", f"{total_return:.1f}%")
        with col3:
            st.metric("Annualized Return", f"{annualized_return:.1f}%")
        with col4:
            outperformance = total_return - buy_hold_return
            st.metric("vs Buy & Hold", f"{outperformance:+.1f}%",
                     delta=f"${final_portfolio_value - buy_hold_value:,.0f}")
        
        # Portfolio value chart
        st.subheader("📊 Portfolio Value Over Time")
        portfolio_chart = portfolio_df.set_index('Date')[['Value']]
        
        # Add buy and hold line for comparison
        buy_hold_line = pd.DataFrame({
            'Date': data_5yr.index,
            'Buy & Hold': buy_hold_shares * data_5yr['Close']
        }).set_index('Date')
        
        comparison_df = pd.concat([portfolio_chart, buy_hold_line], axis=1)
        st.line_chart(comparison_df)
        
        st.caption(f"🟦 Strategy Portfolio | 🟧 Buy & Hold Comparison")
        
        # Summary comparison
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"""
            **Strategy Results:**
            - Final Value: ${final_portfolio_value:,.0f}
            - Return: {total_return:.1f}%
            - Current: {f'${final_cash:,.0f} cash' if final_shares == 0 else f'{final_shares:.2f} shares @ ${final_price:.2f}'}
            """)
        with col2:
            st.info(f"""
            **Buy & Hold Results:**
            - Final Value: ${buy_hold_value:,.0f}
            - Return: {buy_hold_return:.1f}%
            - Position: {buy_hold_shares:.2f} shares held
            """)
            
except Exception as e:
    st.error(f"Error: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.info("""
**Strategy Logic:**
- **BUY** when RSI is oversold AND price near lower Bollinger Band
- **SELL** when RSI is overbought AND price near upper Bollinger Band
- Targets aim for 2-3 trades per month
""")
