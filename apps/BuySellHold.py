import streamlit as st
import pandas as pd
from yahooquery import Ticker
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("Stock OHLC Plot with Indicators, Moving Averages, and Dividends")

# Date inputs
col1, col2, col3 = st.columns(3)
with col1:
    tickers_list = [t.strip().upper() for t in st.session_state.get("tickers", "").split(",") if t.strip()]
    ticker = st.selectbox("Select Stock Ticker", tickers_list) if tickers_list else st.text_input("Enter ticker").upper()
with col2:
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=5 * 365))
with col3:
    end_date = st.date_input("End Date", datetime.today())

# --- Indicator Functions ---
def calculate_rsi(data, window=14):
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def calculate_mfi(data, window=14):
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=window).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=window).sum()
    mfi = 100 - (100 / (1 + (positive_flow / negative_flow)))
    data['MFI'] = mfi
    return data

def calculate_volatility(data, window=30):
    data['Daily Return'] = data['close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=window).std() * (252**0.5)
    return data

def calculate_atr(data, window=14):
    data['HL'] = data['high'] - data['low']
    data['HC'] = abs(data['high'] - data['close'].shift())
    data['LC'] = abs(data['low'] - data['close'].shift())
    data['TR'] = data[['HL', 'HC', 'LC']].max(axis=1)
    data['ATR'] = data['TR'].rolling(window=window).mean()
    return data

# --- Main App ---
if ticker:
    if start_date > end_date:
        st.error("Error: Start date must be before end date.")
    else:
        try:
            tk = Ticker(ticker)
            hist = tk.history(start=start_date, end=end_date, interval='1d')

            if hist.empty:
                st.error(f"No data found for {ticker} within the selected date range.")
            else:
                if isinstance(hist.index, pd.MultiIndex):
                    data = hist.xs(ticker, level=0)
                else:
                    data = hist

                data = data.reset_index()
                data = data.rename(columns={
                    'date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'dividends': 'Dividends'
                })
                data.set_index('Date', inplace=True)

                # Fill missing dividends
                data['Dividends'] = data['Dividends'].fillna(0)
                data['Yield'] = data['Dividends'] / data['Close'] * 100

                # --- Indicators ---
                calc_data = data.rename(columns=str.lower).copy()
                calc_data = calculate_rsi(calc_data)
                calc_data = calculate_mfi(calc_data)
                calc_data = calculate_atr(calc_data)
                calc_data = calculate_volatility(calc_data)

                # Merge indicator results back into main df
                data['RSI'] = calc_data['RSI']
                data['MFI'] = calc_data['MFI']
                data['ATR'] = calc_data['ATR']
                data['Volatility'] = calc_data['Volatility']

                # Moving averages
                data['20-day MA'] = data['Close'].rolling(window=20).mean()
                data['50-day MA'] = data['Close'].rolling(window=50).mean()
                data['200-day MA'] = data['Close'].rolling(window=200).mean()

                # Deltas
                data['Open▲'] = data['Open'] - data['Close'].shift(1)
                data['Open%'] = (data['Open'] / data['Close'].shift(1) - 1) * 100
                data['High▲'] = data['High'] - data['Close'].shift(1)
                data['High%'] = (data['High'] / data['Close'].shift(1) - 1) * 100
                data['Low▲'] = data['Low'] - data['Close'].shift(1)
                data['Low%'] = (data['Low'] / data['Close'].shift(1) - 1) * 100
                data['Close▲'] = data['Close'] - data['Close'].shift(1)
                data['Close%'] = (data['Close'] / data['Close'].shift(1) - 1) * 100

                current_price = data['Close'].iloc[-1]

                # --- Metrics Display ---
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Final RSI (30,70): {data['RSI'].iloc[-1]:.2f}")
                    st.write(f"Final MFI (20,80): {data['MFI'].iloc[-1]:.2f}")
                with col2:
                    st.write(f"ATR (14d): {data['ATR'].iloc[-1]:.2f} ({(data['ATR'].iloc[-1] / current_price * 100):.2f}%)")
                    st.write(f"Volatility (30d): {data['Volatility'].iloc[-1]:.2f} ({(data['Volatility'].iloc[-1] / current_price * 100):.2f}%)")
                with col3:
                    st.write(f"High: {float(data['High'].max()):.2f}")
                    st.write(f"Low: {float(data['Low'].min()):.2f}")

                # --- OHLC + MA Chart ---
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name="OHLC"
                ))
                fig.add_trace(go.Scatter(x=data.index, y=data['20-day MA'], mode='lines', name='20-day MA', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=data.index, y=data['50-day MA'], mode='lines', name='50-day MA', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=data.index, y=data['200-day MA'], mode='lines', name='200-day MA', line=dict(color='red')))

                # Dividend markers
                dividend_dates = data[data['Dividends'] > 0]
                if not dividend_dates.empty:
                    fig.add_trace(go.Scatter(
                        x=dividend_dates.index,
                        y=dividend_dates['Close'],
                        mode='markers',
                        marker=dict(symbol='star', size=10, color='green'),
                        name='Dividend Payout',
                        text=[f"Dividend: ${d:.2f}" for d in dividend_dates['Dividends']],
                        hoverinfo='text+x+y'
                    ))

                fig.update_layout(
                    title=f"{ticker} OHLC Chart with MAs and Dividends ({start_date} - {end_date})",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- RSI + MFI Subplot ---
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
                fig2.add_hline(y=70, line=dict(color="red", dash="dot"))
                fig2.add_hline(y=30, line=dict(color="green", dash="dot"))

                fig2.add_trace(go.Scatter(x=data.index, y=data['MFI'], mode='lines', name='MFI', line=dict(color='brown')))
                fig2.add_hline(y=80, line=dict(color="red", dash="dot"))
                fig2.add_hline(y=20, line=dict(color="green", dash="dot"))

                fig2.update_layout(title="RSI and MFI Indicators", xaxis_title="Date", yaxis_title="Value")
                st.plotly_chart(fig2, use_container_width=True)

                # --- Key Statistics ---
                st.subheader("Key Statistics")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.write(f"Ave Open▲: ${float(data['Open▲'].mean()):.2f}")
                    st.write(f"Ave Open%: {float(data['Open%'].mean()):.2f}%")
                with c2:
                    st.write(f"Ave High▲: ${float(data['High▲'].mean()):.2f}")
                    st.write(f"Ave High%: {float(data['High%'].mean()):.2f}%")
                with c3:
                    st.write(f"Ave Low▲: ${float(data['Low▲'].mean()):.2f}")
                    st.write(f"Ave Low%: {float(data['Low%'].mean()):.2f}%")
                with c4:
                    st.write(f"Ave Close▲: ${float(data['Close▲'].mean()):.2f}")
                    st.write(f"Ave Close%: {float(data['Close%'].mean()):.2f}%")

                # --- Option to filter dividends ---
                show_only_dividends = st.checkbox("Show only rows with dividends > 0")
                if show_only_dividends:
                    data = data[data['Dividends'] > 0]

                # --- Raw Data ---
                st.dataframe(data.iloc[::-1][[
                    'Open', 'High', 'Low', 'Close', 'Volume',
                    'Dividends', 'Yield',
                    '20-day MA', '50-day MA', '200-day MA',
                    'RSI', 'MFI', 'ATR', 'Volatility',
                    'Open▲', 'Open%', 'High▲', 'High%',
                    'Low▲', 'Low%', 'Close▲', 'Close%'
                ]])

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info("Please enter a stock ticker symbol.")
