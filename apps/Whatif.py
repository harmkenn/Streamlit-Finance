import streamlit as st
from yahooquery import Ticker
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.title("📈 $100,000 Investment Growth with Reinvested Dividends")

# Parameters
ticker_list = [t.strip().upper() for t in st.session_state.get("tickers", "").split(",") if t.strip()]
#ticker_list = ["MSTY", "MAIN"]
tickers = st.multiselect("Select Tickers to Compare",options=ticker_list,default=ticker_list[:5])
initial_investment = 100000

# Date range
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

fig = go.Figure()

for ticker_symbol in ticker_list:
    ticker = Ticker(ticker_symbol)
    history = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    if isinstance(history, pd.DataFrame) and not history.empty:
        history = history.reset_index()
        df = history[history['symbol'] == ticker_symbol][['date', 'close', 'dividends']]
        df = df.sort_values('date').reset_index(drop=True)
        df['dividends'] = df['dividends'].fillna(0)

        # Initialize shares and investment value
        shares = initial_investment / df.loc[0, 'close']
        df['shares'] = 0.0
        df['investment_value'] = 0.0
        df.loc[0, 'shares'] = shares
        df.loc[0, 'investment_value'] = shares * df.loc[0, 'close']

        for i in range(1, len(df)):
            shares += df.loc[i, 'dividends'] * shares / df.loc[i, 'close']
            df.loc[i, 'shares'] = shares
            df.loc[i, 'investment_value'] = shares * df.loc[i, 'close']

        # Calculate percent change
        pct_change = (df['investment_value'].iloc[-1] / df['investment_value'].iloc[0] - 1) * 100

        # Line plot for investment value
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['investment_value'],
            mode='lines',
            name=f"{ticker_symbol} ({pct_change:.1f}%)",
            hovertemplate=(
                f"<b>{ticker_symbol}</b><br>"
                "Date: %{x}<br>"
                "Value: $%{y:,.2f}<br>"
                f"Total Return: {pct_change:.1f}%<extra></extra>"
            )
        ))

        # Add stars for dividends
        dividend_days = df[df['dividends'] > 0]
        fig.add_trace(go.Scatter(
            x=dividend_days['date'],
            y=dividend_days['investment_value'],
            mode='markers',
            name=f"{ticker_symbol} Dividends",
            marker=dict(size=10, symbol='star'),
            hovertemplate="Dividend: $%{text:.2f}<br>Value: $%{y:,.2f}<extra></extra>",
            text=dividend_days['dividends']
        ))

fig.update_layout(
    title="📊 $100,000 Investment Growth with Reinvested Dividends (1 Year)",
    xaxis_title="Date",
    yaxis_title="Portfolio Value (USD)",
    template="plotly_white",
    height=600
)

st.plotly_chart(fig, width='stretch')
