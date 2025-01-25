#Updated 20250123

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.title("Stock OHLC Plot with Moving Averages")

# Date inputs
col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("Start Date",
                               datetime.today() - timedelta(days=5 * 365))
with col2:
    end_date = st.date_input("End Date", datetime.today())
with col3:
    # Ticker input
    ticker = st.text_input("Enter Stock Ticker", "TQQQ").upper()


def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data


def calculate_mfi(data, window=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = (money_flow.where(typical_price > typical_price.shift(1),
                                      0)).rolling(window=window).sum()
    negative_flow = (money_flow.where(typical_price < typical_price.shift(1),
                                      0)).rolling(window=window).sum()
    money_flow_index = 100 - (100 / (1 + (positive_flow / negative_flow)))
    data['MFI'] = money_flow_index
    return data


def calculate_volatility(data, window=30):
    data['Daily Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=window).std() * (
        252**0.5)  # Annualized volatility
    return data


def calculate_atr(data, window=14):
    data['HL'] = data['High'] - data['Low']
    data['HC'] = abs(data['High'] - data['Close'].shift())
    data['LC'] = abs(data['Low'] - data['Close'].shift())
    data['TR'] = data[['HL', 'HC', 'LC']].max(axis=1)
    data['ATR'] = data['TR'].rolling(window=window).mean()
    return data


if ticker:
    if start_date > end_date:
        st.error("Error: Start date must be before end date.")
    else:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)

            if data.empty:
                st.error(
                    f"No data found for {ticker} within the selected date range."
                )
            else:
                # Calculate moving averages
                data['50-day MA'] = data['Close'].rolling(window=50).mean()
                data['200-day MA'] = data['Close'].rolling(window=200).mean()

                # Calculate RSI, MFI, ATR, and Volatility
                data = calculate_rsi(data)
                data = calculate_mfi(data)
                data = calculate_atr(data)
                data = calculate_volatility(data)
                data = data.drop(columns=['Dividends', 'Stock Splits', 'Capital Gains'])
                data['Open▲'] = data['Open'] - data['Close'].shift(1)
                data['Open%'] = (data['Open'] / data['Close'].shift(1) - 1)*100
                data['High▲'] = data['High'] - data['Close'].shift(1)
                data['High%'] = (data['High'] / data['Close'].shift(1) - 1)*100   
                data['Low▲'] = data['Low'] - data['Close'].shift(1)
                data['Low%'] = (data['Low'] / data['Close'].shift(1) - 1)*100
                data['Close▲'] = data['Close'] - data['Close'].shift(1)
                data['Close%'] = (data['Close'] / data['Close'].shift(1) - 1)*100

                # Get current price
                current_price = data['Close'].iloc[-1]
                # Display final RSI, MFI, ATR, and Volatility
                with col1:
                    st.write(
                        f"Final RSI (30,70) for {ticker}: {data['RSI'].iloc[-1]:.2f}"
                    )
                    st.write(
                        f"Final MFI (20,80) for {ticker}: {data['MFI'].iloc[-1]:.2f}"
                    )
                with col2:
                    st.write(
                        f"Volatility (ATR, 14-day) for {ticker}: {data['ATR'].iloc[-1]:.2f} ({(data['ATR'].iloc[-1] / current_price * 100):.2f}%)"
                    )
                    st.write(
                        f"Volatility (30-day) for {ticker}: {data['Volatility'].iloc[-1]:.2f} ({(data['Volatility'].iloc[-1] / current_price * 100):.2f}%)"
                    )
                with col3:
                    st.write(f"High: {float(data['High'].max()):.2f}")
                    st.write(f"Low: {float(data['Low'].min()):.2f}")

                fig = go.Figure()

                # Add candlestick trace
                fig.add_trace(
                    go.Candlestick(x=data.index,
                                   open=data['Open'],
                                   high=data['High'],
                                   low=data['Low'],
                                   close=data['Close'],
                                   name="OHLC"))

                # Add moving average traces
                fig.add_trace(
                    go.Scatter(x=data.index,
                               y=data['50-day MA'],
                               mode='lines',
                               name='50-day MA',
                               line=dict(color='blue')))
                fig.add_trace(
                    go.Scatter(x=data.index,
                               y=data['200-day MA'],
                               mode='lines',
                               name='200-day MA',
                               line=dict(color='red')))

                fig.update_layout(
                    title=f"{ticker} OHLC Chart ({start_date} - {end_date})",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False)

                st.plotly_chart(fig)

                st.subheader("Key Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1: 
                    st.write(f"Ave Open▲: ${float(data['Open▲'].mean()):.2f}")
                    st.write(f"Ave Open%: {float(data['Open%'].mean()):.2f}%")
                with col2:
                    st.write(f"Ave High▲: ${float(data['High▲'].mean()):.2f}")
                    st.write(f"Ave High%: {float(data['High%'].mean()):.2f}%")

                with col3:
                    st.write(f"Ave Low▲: ${float(data['Low▲'].mean()):.2f}")
                    st.write(f"Ave Low%: {float(data['Low%'].mean()):.2f}%")
                    
                with col4:
                    st.write(f"Ave Close▲: ${float(data['Close▲'].mean()):.2f}")
                    st.write(f"Ave Close%: {float(data['Close%'].mean()):.2f}%")

                with st.expander("Show raw data"):                
                    st.dataframe(data.iloc[::-1])

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please enter a stock ticker symbol.")
