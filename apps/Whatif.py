import streamlit as st
from yahooquery import Ticker
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

def calculate_etf_value(ticker, initial_investment):
    """Calculates the current value of an ETF investment with reinvested dividends using yahooquery."""

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        tq = Ticker(ticker)
        history = tq.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

        if history.empty:
            return "No historical data found for the given ticker.", None, None, None

        history = history.reset_index()
        history = history[history['symbol'] == ticker]

        if 'adjclose' not in history.columns or 'dividends' not in history.columns:
            return "Required columns missing from data.", None, None, None

        history = history[['date', 'adjclose', 'dividends']].rename(columns={
            'date': 'Date',
            'adjclose': 'Price',
            'dividends': 'Dividend'
        })
        history.set_index('Date', inplace=True)

        if history['Price'].empty:
            return "No historical price data found for the given ticker.", None, None, None

        # Calculate initial shares
        initial_price = history['Price'].iloc[0]
        shares = initial_investment / initial_price

        # Reinvest dividends
        current_shares = shares
        dividend_dates = []
        daily_values = []
        payouts = []

        for date, row in history.iterrows():
            price = row['Price']
            dividend = row['Dividend']

            if pd.notna(dividend) and dividend > 0:
                current_shares += (current_shares * dividend) / price
                dividend_dates.append(date)
                payouts.append(shares * dividend)
            else:
                payouts.append(0)

            daily_values.append(current_shares * price)

        history['Daily Value'] = daily_values
        history['Dividend Payout'] = payouts

        return f"The current value of your investment is: ${daily_values[-1]:,.2f}", history['Price'], dividend_dates, history

    except Exception as e:
        return f"An error occurred: {e}", None, None, None


# Streamlit App
st.title("ETF Growth Calculator (using Yahooquery)")
st.write("Enter an ETF ticker symbol and initial investment to calculate its current value.")

col1, col2 = st.columns(2)
with col1:
    tickers_list = [t.strip().upper() for t in st.session_state.get("tickers", "").split(",") if t.strip()]
    ticker = st.selectbox("Select Stock Ticker", tickers_list) if tickers_list else ""
with col2:
    initial_investment = st.number_input("Initial Investment ($):", value=10000.0)

if ticker:
    result, historical_prices, dividend_dates, daily_data = calculate_etf_value(ticker, initial_investment)
    st.write(result)

    if historical_prices is not None and daily_data is not None:
        # Plot investment growth
        fig = go.Figure(data=go.Scatter(x=daily_data.index, y=daily_data['Daily Value'], mode='lines'))
        fig.update_layout(title=f"{ticker} Investment Value Over Time",
                          xaxis_title="Date", yaxis_title="Value ($)")

        # Add vertical lines for dividends
        for date in dividend_dates:
            fig.add_vline(x=date, line_width=1, line_dash="dash", line_color="green")

        st.plotly_chart(fig)

        # Display table
        st.write("Daily Investment Values, Dividend Payouts, and Stock Prices:")
        formatted_data = daily_data.copy()
        formatted_data['Daily Value'] = formatted_data['Daily Value'].apply(lambda x: "${:,.2f}".format(x))
        formatted_data['Dividend Payout'] = formatted_data['Dividend Payout'].apply(lambda x: "${:,.2f}".format(x))
        formatted_data['Price'] = formatted_data['Price'].apply(lambda x: "${:,.2f}".format(x))

        filtered_data = formatted_data[formatted_data['Dividend Payout'] != "$0.00"]
        st.dataframe(filtered_data)
else:
    st.write("Please enter an ETF ticker symbol.")
