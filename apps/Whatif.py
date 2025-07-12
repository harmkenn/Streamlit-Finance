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
        dividend_payouts = pd.Series(index=historical_prices.index)

        for date, price in historical_prices.items():
            if date in dividends and dividends[date] > 0:
                dividend = dividends[date]
                current_shares += (current_shares * dividend) / price
                dividend_dates.append(date)
                dividend_payouts[date] = shares * dividend

            daily_values[date] = current_shares * price

        return f"The current value of your investment is: ${daily_values.iloc[-1]:.2f}", historical_prices, dividend_dates, pd.DataFrame({'Stock Price': historical_prices, 'Daily Value': daily_values, 'Dividend Payout': dividend_payouts})

    except Exception as e:
        return f"An error occurred: {e}", None, None, None

# Streamlit app
st.title("ETF Growth Calculator")
st.write("Enter an ETF ticker symbol and initial investment to calculate its current value.")
col1, col2 = st.columns(2)
with col1:
    # Get tickers from session state and split into a list
    tickers_list = [t.strip().upper() for t in st.session_state.get("tickers", "").split(",") if t.strip()]

    # Ticker selector
    ticker = st.selectbox("Select Stock Ticker", tickers_list) if tickers_list else ""
with col2:
    initial_investment = st.number_input("Initial Investment ($):", value=10000.0)


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

        # Display daily values as a Pandas DataFrame with currency formatting
        if daily_data is not None:
            st.write("Daily Investment Values, Dividend Payouts, and Stock Prices:")
            daily_data['Daily Value'] = daily_data['Daily Value'].apply(lambda x: "${:,.2f}".format(x))
            daily_data['Dividend Payout'] = daily_data['Dividend Payout'].apply(lambda x: "${:,.2f}".format(x))
            daily_data['Stock Price'] = daily_data['Stock Price'].apply(lambda x: "${:,.2f}".format(x))

            # Show only rows where Dividend Payout has a value (not zero or not null)
            filtered_data = daily_data[daily_data['Dividend Payout'] != "$0.00"]
            st.dataframe(filtered_data)
else:
    st.write("Please enter an ETF ticker symbol.")