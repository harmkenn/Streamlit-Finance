import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import datetime as dt
from numpy import floor


c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    ticker = st.text_input("Ticker", "TQQQ")
with c2:
    inc = st.number_input("Trigger", min_value=0.001, max_value=1.010, value=0.030, step=0.005, format="%.3f")
with c3:
    chunk = st.number_input("Chunk", min_value=.1, max_value=.99, value=.3, step=.05, format="%.3f")
with c4:
    start_date = st.date_input("Select start date", value=dt.date(2022, 1, 1), min_value=dt.date(2010, 1, 1), max_value=dt.date.today())  # replace with your desired start date
with c5:
    end_date = st.date_input("Select end date", value=dt.date.today(), min_value=dt.date(2010, 1, 1), max_value=dt.date.today())  # replace with your desired start date

tqqq_data = yf.download(ticker, start=start_date, end=end_date)

tqqq_data = tqqq_data.drop(['Volume', 'Adj Close'], axis=1)

# Create the OHLC graph
fig = go.Figure(data=[go.Candlestick(
    x=tqqq_data.index,
    open=tqqq_data["Open"],
    high=tqqq_data["High"],
    low=tqqq_data["Low"],
    close=tqqq_data["Close"]
)])

# Initialize the starting cash and shares
tqqq_data['Drop'] = 0.0
tqqq_data['Raise'] = 0.0
tqqq_data['Move $'] = 0.0
tqqq_data['Move shares'] = 0
tqqq_data['Cash'] = 100000.00
# Calculate the initial number of shares
start_price = tqqq_data.iloc[0]['Close']
initial_shares = int(100000 / start_price)
tqqq_data['Shares'] = initial_shares
tqqq_data['Buy/Sell'] = ''
tqqq_data['chunks'] = 0



# Iterate through each day
for i in range(1, len(tqqq_data)):
    # Initialize cash and shares for the current day
    tqqq_data.iloc[i, tqqq_data.columns.get_loc('Cash')] = tqqq_data.iloc[i-1, tqqq_data.columns.get_loc('Cash')]
    tqqq_data.iloc[i, tqqq_data.columns.get_loc('Shares')] = tqqq_data.iloc[i-1, tqqq_data.columns.get_loc('Shares')]

    # Check if the price rose by at least one Trigger
    if tqqq_data.iloc[i, tqqq_data.columns.get_loc('High')] > tqqq_data.iloc[i-1, tqqq_data.columns.get_loc('Close')] * (1+inc):
        tqqq_data.iloc[i, tqqq_data.columns.get_loc('Raise')] = (tqqq_data.iloc[i, tqqq_data.columns.get_loc('High')]-tqqq_data.iloc[i-1,
             tqqq_data.columns.get_loc('Close')])/tqqq_data.iloc[i-1, tqqq_data.columns.get_loc('Close')]
        chunks = floor(tqqq_data.iloc[i, tqqq_data.columns.get_loc('Raise')]/inc)
        tqqq_data.iloc[i, tqqq_data.columns.get_loc('chunks')] = chunks
        shares_left = tqqq_data.iloc[i-1, tqqq_data.columns.get_loc('Shares')]
        cash_to_receive = 0
        j = 1
        while j <= chunks:
            tqqq_data.iloc[i, tqqq_data.columns.get_loc('Buy/Sell')] = 'Sell'+str(j)
            shares_to_sell = int(shares_left * chunk)
            shares_left = shares_left - shares_to_sell
            tqqq_data.iloc[i, tqqq_data.columns.get_loc('Shares')] = shares_left
            cash_to_receive = cash_to_receive + shares_to_sell * (tqqq_data.iloc[i-1, tqqq_data.columns.get_loc('Close')] * (1+j*inc))
            tqqq_data.iloc[i, tqqq_data.columns.get_loc('Cash')] = tqqq_data.iloc[i-1, tqqq_data.columns.get_loc('Cash')] + cash_to_receive
            j += 1

    # Check if the price decreased by at least one Trigger
    if tqqq_data.iloc[i, tqqq_data.columns.get_loc('Low')]/tqqq_data.iloc[i-1, tqqq_data.columns.get_loc('Close')] < (1-inc):
        tqqq_data.iloc[i, tqqq_data.columns.get_loc('Drop')] = 1 - tqqq_data.iloc[i, tqqq_data.columns.get_loc('Low')]/tqqq_data.iloc[i-1, tqqq_data.columns.get_loc('Close')]
        chunks = floor(tqqq_data.iloc[i, tqqq_data.columns.get_loc('Drop')]/inc)
        tqqq_data.iloc[i, tqqq_data.columns.get_loc('chunks')] = chunks
        cash_left = tqqq_data.iloc[i-1, tqqq_data.columns.get_loc('Cash')]
        shares_to_receive = 0
        j = 1
        while j <= chunks:
            tqqq_data.iloc[i, tqqq_data.columns.get_loc('Buy/Sell')] = 'Buy'+str(j)
            cash_to_use = cash_left * chunk
            cash_left = cash_left - cash_to_use
            tqqq_data.iloc[i, tqqq_data.columns.get_loc('Cash')] = cash_left
            shares_to_receive = shares_to_receive + int(cash_to_use / (tqqq_data.iloc[i-1, tqqq_data.columns.get_loc('Close')] * (1-j*inc)))
            tqqq_data.iloc[i, tqqq_data.columns.get_loc('Shares')] = tqqq_data.iloc[i-1, tqqq_data.columns.get_loc('Shares')] + shares_to_receive
            j += 1

    tqqq_data.iloc[i, tqqq_data.columns.get_loc('Move $')] = tqqq_data.iloc[i, tqqq_data.columns.get_loc('Cash')] - tqqq_data.iloc[i-1, tqqq_data.columns.get_loc('Cash')]
    tqqq_data.iloc[i, tqqq_data.columns.get_loc('Move shares')] = tqqq_data.iloc[i, tqqq_data.columns.get_loc('Shares')] - tqqq_data.iloc[i-1, tqqq_data.columns.get_loc('Shares')]

tqqq_data['value'] = tqqq_data['Shares'] * tqqq_data['Close']
tqqq_data["total"] = tqqq_data["Cash"] + tqqq_data["value"]
tqqq_data['Close%'] = tqqq_data['Close'].pct_change()
tqqq_data['Total%'] = tqqq_data['total'].pct_change()

st.dataframe(tqqq_data.iloc[::-1], width=None, use_container_width=True)

# Add a button to download the data as a CSV
st.download_button(
    label="Download CSV",
    data=tqqq_data.to_csv(index=False),
    file_name="tqqq_data.csv",
    mime="text/csv"
)

# Add the OHLC graph to the Streamlit app

st.plotly_chart(fig, use_container_width=True)