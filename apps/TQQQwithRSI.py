import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

def calculate_rsi(data, window=14):
    """
    Calculates RSI (Relative Strength Index) for the given data.
    
    Parameters:
    - data: DataFrame containing 'Close' prices.
    - window: Lookback period for RSI calculation (default is 14).
    
    Returns:
    - DataFrame with an additional 'RSI' column.
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def scale_thresholds(rsi):
    """
    Scales the sell and buy thresholds based on RSI.
    
    Parameters:
    - rsi: Current RSI value.
    
    Returns:
    - sell_threshold: Scaled sell threshold.
    - buy_threshold: Scaled buy threshold.
    """
    # Define extremes
    rsi_high = 75
    rsi_low = 25
    sell_high = 0.025  # 2.5% spike
    sell_low = 0.065  # 6.5% spike
    buy_high = -0.065  # 6.5% dip
    buy_low = -0.025  # 2.5% dip
    
    # Scale thresholds linearly based on RSI
    sell_threshold = sell_low + (sell_high - sell_low) * (rsi - rsi_low) / (rsi_high - rsi_low)
    buy_threshold = buy_low + (buy_high - buy_low) * (rsi - rsi_low) / (rsi_high - rsi_low)
    
    return sell_threshold, buy_threshold

def simulate_strategy(data, initial_capital=10000):
    """
    Simulates the trading strategy and compares it to buy-and-hold.
    
    Parameters:
    - data: DataFrame containing 'Date', 'Close', 'RSI' columns.
    - initial_capital: Starting capital for the simulation.
    
    Returns:
    - DataFrame with portfolio values for both strategies.
    """
    # Initialize variables
    capital = initial_capital
    shares = 0
    buy_and_hold_value = initial_capital / data.loc[0, 'Close']  # Buy-and-hold shares
    
    # Create columns for portfolio values
    data['Strategy Portfolio Value'] = None
    data['Buy-and-Hold Portfolio Value'] = None
    
    for i in range(1, len(data)):
        # Calculate daily price change percentage
        price_change = (data.loc[i, 'Close'] - data.loc[i - 1, 'Close']) / data.loc[i - 1, 'Close']
        
        # Get RSI and scale thresholds
        rsi = data.loc[i, 'RSI']
        sell_threshold, buy_threshold = scale_thresholds(rsi)
        
        # Execute trading strategy
        if price_change >= sell_threshold and shares > 0:
            # Sell $10,000 worth of shares
            sell_amount = min(10000, shares * data.loc[i, 'Close'])
            shares -= sell_amount / data.loc[i, 'Close']
            capital += sell_amount
        elif price_change <= buy_threshold and capital >= 10000:
            # Buy $10,000 worth of shares
            buy_amount = 10000
            shares += buy_amount / data.loc[i, 'Close']
            capital -= buy_amount
        
        # Calculate portfolio value for the strategy
        strategy_portfolio_value = capital + (shares * data.loc[i, 'Close'])
        data.loc[i, 'Strategy Portfolio Value'] = strategy_portfolio_value
        
        # Calculate portfolio value for buy-and-hold
        buy_and_hold_portfolio_value = buy_and_hold_value * data.loc[i, 'Close']
        data.loc[i, 'Buy-and-Hold Portfolio Value'] = buy_and_hold_portfolio_value
    
    return data

def plot_results(data):
    """
    Plots the portfolio values for the trading strategy and buy-and-hold using Plotly.
    
    Parameters:
    - data: DataFrame containing 'Date', 'Strategy Portfolio Value', 'Buy-and-Hold Portfolio Value'.
    """
    fig = go.Figure()
    
    # Add trading strategy portfolio value
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Strategy Portfolio Value'],
        mode='lines',
        name='Trading Strategy',
        line=dict(color='blue')
    ))
    
    # Add buy-and-hold portfolio value
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Buy-and-Hold Portfolio Value'],
        mode='lines',
        name='Buy-and-Hold',
        line=dict(color='green')
    ))
    
    # Customize layout
    fig.update_layout(
        title="Portfolio Value Comparison: Trading Strategy vs Buy-and-Hold",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        legend_title="Strategy",
        template="plotly_white"
    )
    
    st.plotly_chart(fig)

# Streamlit App
def main():
    st.title("TQQQ Trading Strategy Simulator")
    
    # User inputs
    ticker = st.text_input("Enter Ticker Symbol:", value="TQQQ")
    start_date = st.date_input("Start Date:", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date:", value=pd.to_datetime("2025-12-10"))
    initial_capital = st.number_input("Initial Capital ($):", value=10000, step=1000)
    
    if st.button("Run Simulation"):
        # Fetch historical data
        st.write("Fetching data...")
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Reset index to make 'Date' a column
        data.reset_index(inplace=True)
        
        # Calculate RSI
        st.write("Calculating RSI...")
        data = calculate_rsi(data)
        
        # Drop rows with NaN RSI values (due to rolling window)
        data = data.dropna()
        
        # Simulate the strategy
        st.write("Simulating strategy...")
        result = simulate_strategy(data, initial_capital)
        
        # Plot the results
        st.write("Plotting results...")
        plot_results(result)
        
        # Display final portfolio values
        final_strategy_value = result['Strategy Portfolio Value'].iloc[-1]
        final_buy_and_hold_value = result['Buy-and-Hold Portfolio Value'].iloc[-1]
        st.write(f"Final Portfolio Value (Trading Strategy): ${final_strategy_value:.2f}")
        st.write(f"Final Portfolio Value (Buy-and-Hold): ${final_buy_and_hold_value:.2f}")

if __name__ == "__main__":
    main()
