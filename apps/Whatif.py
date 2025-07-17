import streamlit as st
from yahooquery import Ticker
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.title("📈 $100,000 Investment Growth with Reinvested Dividends")

# Parameters
tickers_list = [t.strip().upper() for t in st.session_state.get("tickers", "").split(",") if t.strip()]
ticker_symbol = st.selectbox("Select Stock Ticker", tickers_list) if tickers_list else st.text_input("Enter ticker").upper()
initial_investment = 100000

# Date range
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# Fetch data
ticker = Ticker(ticker_symbol)
history = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

# Check for data
if isinstance(history, pd.DataFrame) and not history.empty:
    # Filter and clean
    history = history.reset_index()
    df = history[history['symbol'] == ticker_symbol][['date', 'close', 'dividends']]
    df = df.sort_values('date').reset_index(drop=True)
    df['dividends'] = df['dividends'].fillna(0)

    # Initialize columns
    df['shares'] = 0.0
    df['investment_value'] = 0.0

    # Initial shares
    shares = initial_investment / df.loc[0, 'close']
    df.loc[0, 'shares'] = shares
    df.loc[0, 'investment_value'] = shares * df.loc[0, 'close']

    # Loop through days to update shares and value
    for i in range(1, len(df)):
        shares += df.loc[i, 'dividends'] * shares / df.loc[i, 'close']
        df.loc[i, 'shares'] = shares
        df.loc[i, 'investment_value'] = shares * df.loc[i, 'close']

    # Plot investment value
    fig = go.Figure()

    # Line plot for investment value
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['investment_value'],
        mode='lines',
        name='Investment Value',
        line=dict(color='green')
    ))

    # Add stars on dividend payout days
    dividend_days = df[df['dividends'] > 0]
    fig.add_trace(go.Scatter(
        x=dividend_days['date'],
        y=dividend_days['investment_value'],
        mode='markers',
        name='Dividend Payout',
        marker=dict(size=10, symbol='star', color='red'),
        hovertemplate="Dividend: %{text}<extra></extra>",
        text=dividend_days['dividends'].apply(lambda x: f"${x:.2f}")
    ))

    fig.update_layout(
        title=f"📈 Value of $100,000 Investment in {ticker_symbol} (with Reinvested Dividends)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (USD)",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show the data
    st.subheader("📄 Investment Table")
    st.dataframe(df[['date', 'close', 'dividends', 'shares', 'investment_value']].sort_values('date', ascending=False), use_container_width=True)

else:
    st.error(f"⚠️ No data found for {ticker_symbol}. Please check the ticker or try again later.")
