import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

def calculate_etf_value(ticker, initial_investment):
    """Calculates the current value of an ETF investment with reinvested dividends."""

    try:
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        stock = yf.Ticker(ticker)
        historical_data = stock.history(start=start_date, end=end_date, actions=True)
        dividends = historical_data['Dividends']
        historical_prices = historical_data['Close']

        if historical_prices.empty:
            return "No historical price data found for the given ticker.", None, None
        if dividends.empty:
            return "No dividend data found.", None, None

        # Calculate initial shares
        initial_price = historical_prices.iloc[0]
        shares = initial_investment / initial_price

        # Reinvest dividends
        dividend_dates = []
        for date, dividend in dividends.items():
            if dividend > 0:
                price_at_dividend = historical_prices.asof(date)
                if pd.isna(price_at_dividend):
                    price_at_dividend = historical_prices[historical_prices.index > date].iloc[0]

                shares += (shares * dividend) / price_at_dividend
                dividend_dates.append(date)

        # Calculate current value
        current_price = historical_prices.iloc[-1]
        current_value = shares * current_price

        return f"The current value of your investment is: ${current_value:.2f}", historical_prices, dividend_dates

    except Exception as e:
        return f"An error occurred: {e}", None, None

# Streamlit app
st.title("ETF Growth Calculator")
st.write("Enter an ETF ticker symbol and initial investment to calculate its current value.")

ticker = st.text_input("ETF Ticker Symbol (e.g., SPY, VOO, MSTY):")
initial_investment = st.number_input("Initial Investment ($):", value=10000.0)

if st.button("Calculate"):
    if ticker:
        result, historical_prices, dividend_dates = calculate_etf_value(ticker, initial_investment)
        st.write(result)

        if historical_prices is not None:
            # Create Plotly chart
            fig = go.Figure(data=go.Scatter(x=historical_prices.index, y=historical_prices.values, mode='lines'))
            fig.update_layout(title=f"{ticker} Historical Prices", xaxis_title="Date", yaxis_title="Price")

            # Add vertical lines for dividend dates
            if dividend_dates is not None:
                for date in dividend_dates:
                    fig.add_vline(x=date, line_width=1, line_dash="dash", line_color="green")

            st.plotly_chart(fig)
    else:
        st.write("Please enter an ETF ticker symbol.")