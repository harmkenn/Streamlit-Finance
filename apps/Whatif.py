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
            return "No historical price data found for the given ticker.", None, None, None
        if dividends.empty:
            return "No dividend data found.", None, None, None

        # Calculate initial shares
        initial_price = historical_prices.iloc[0]
        shares = initial_investment / initial_price

        # Reinvest dividends and calculate daily values
        dividend_dates = []
        daily_values = pd.Series(index=historical_prices.index)
        current_shares = shares
        dividend_payouts = pd.Series(index=historical_prices.index) #New series for dividend payouts

        for date, price in historical_prices.items():
            if date in dividends and dividends[date] > 0:
                dividend = dividends[date]
                current_shares += (current_shares * dividend) / price
                dividend_dates.append(date)
                dividend_payouts[date] = shares*dividend #Record the dividend payout amount

            daily_values[date] = current_shares * price

        return f"The current value of your investment is: ${daily_values.iloc[-1]:.2f}", historical_prices, dividend_dates, pd.DataFrame({'Daily Value': daily_values, 'Dividend Payout': dividend_payouts})

    except Exception as e:
        return f"An error occurred: {e}", None, None, None

# Streamlit app
st.title("ETF Growth Calculator")
st.write("Enter an ETF ticker symbol and initial investment to calculate its current value.")

ticker = st.text_input("ETF Ticker Symbol (e.g., SPY, VOO, MSTY):")
initial_investment = st.number_input("Initial Investment ($):", value=10000.0)

if st.button("Calculate"):
    if ticker:
        result, historical_prices, dividend_dates, daily_data = calculate_etf_value(ticker, initial_investment)
        st.write(result)

        if historical_prices is not None:
            # Create Plotly chart
            fig = go.Figure(data=go.Scatter(x=daily_data.index, y=daily_data['Daily Value'], mode='lines'))
            fig.update_layout(title=f"{ticker} Investment Value Over Time", xaxis_title="Date", yaxis_title="Value ($)")

            # Add vertical lines for dividend dates
            if dividend_dates is not None:
                for date in dividend_dates:
                    fig.add_vline(x=date, line_width=1, line_dash="dash", line_color="green")

            st.plotly_chart(fig)

            # Display daily values as a Pandas DataFrame
            if daily_data is not None:
                st.write("Daily Investment Values and Dividend Payouts:")
                st.dataframe(daily_data)

    else:
        st.write("Please enter an ETF ticker symbol.")