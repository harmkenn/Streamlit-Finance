import streamlit as st
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
import plotly.express as px
from plotly.subplots import make_subplots

st.title("Intraday Stock Prices")
col1, col2 = st.columns(2)
with col1:
    stock_symbol = st.text_input("Enter stock symbol (e.g. AAPL, GOOG, MSFT):","NVDY")

if stock_symbol:
    with col2:
        refresh_button = st.button("Refresh")

    if refresh_button:
        ts = TimeSeries(key='V8RWD2L3WZMPFALK', output_format='pandas')
        data, meta_data = ts.get_intraday(symbol=stock_symbol, interval='1min', outputsize='full')

        # Reverse the order of the data to get the newest row first
        data = data.iloc[::-1]

        # Select only the Close, Volume, and Dividends columns
        data = data[["4. close", "5. volume", "7. dividend amount"]]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(px.line(data, x=data.index, y="4. close").data[0], secondary_y=False)
        fig.add_trace(px.bar(data, x=data.index, y="5. volume").data[0], secondary_y=True)
        fig.data[1].marker.color = 'red'
        fig.update_layout(title=f"{stock_symbol} Intraday Prices", xaxis_title="Time", yaxis_title="Price")
        fig.update_yaxes(title_text="Volume", secondary_y=True)

        st.plotly_chart(fig)
        st.write(data)
    else:
        ts = TimeSeries(key='YOUR_ALPHA_VANTAGE_API_KEY', output_format='pandas')
        data, meta_data = ts.get_intraday(symbol=stock_symbol, interval='1min', outputsize='full')

        # Reverse the order of the data to get the newest row first
        data = data.iloc[::-1]

        # Select only the Close, Volume, and Dividends columns
        data = data[["4. close", "5. volume", "7. dividend amount"]]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(px.line(data, x=data.index, y="4. close").data[0], secondary_y=False)
        fig.add_trace(px.bar(data, x=data.index, y="5. volume").data[0], secondary_y=True)
        fig.data[1].marker.color = 'red'
        fig.update_layout(title=f"{stock_symbol} Intraday Prices", xaxis_title="Time", yaxis_title="Price")
        fig.update_yaxes(title_text="Volume", secondary_y=True)

        st.plotly_chart(fig)
        st.write(data)