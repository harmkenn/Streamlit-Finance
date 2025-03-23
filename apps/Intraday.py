import streamlit as st
from polygon-api-client import RESTClient
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

st.title("Intraday Stock Prices")
col1, col2 = st.columns(2)
with col1:
    stock_symbol = st.text_input("Enter stock symbol (e.g. AAPL, GOOG, MSFT):","NVDY")

if stock_symbol:
    with col2:
        refresh_button = st.button("Refresh")

    if refresh_button:
        client = RESTClient(api_key='YOUR_POLYGON_API_KEY')
        data = client.get_aggs(symbol=stock_symbol, multiplier=1, timespan='minute', from_='2023-02-20', to='2023-02-20')

        # Convert data to pandas DataFrame
        df = pd.DataFrame(data.results)

        # Reverse the order of the data to get the newest row first
        df = df.iloc[::-1]

        # Select only the Close, Volume, and Dividends columns
        df = df[["c", "v", "d"]]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(px.line(df, x=df.index, y="c").data[0], secondary_y=False)
        fig.add_trace(px.bar(df, x=df.index, y="v").data[0], secondary_y=True)
        fig.data[1].marker.color = 'red'
        fig.update_layout(title=f"{stock_symbol} Intraday Prices", xaxis_title="Time", yaxis_title="Price")
        fig.update_yaxes(title_text="Volume", secondary_y=True)

        st.plotly_chart(fig)
        st.write(df)
    else:
        client = RESTClient(api_key='4UQbILKSeObAjCKlDdhDFWJ7CzVFYm4bEY')
        data = client.get_aggs(symbol=stock_symbol, multiplier=1, timespan='minute', from_='2023-02-20', to='2023-02-20')

        # Convert data to pandas DataFrame
        df = pd.DataFrame(data.results)

        # Reverse the order of the data to get the newest row first
        df = df.iloc[::-1]

        # Select only the Close, Volume, and Dividends columns
        df = df[["c", "v", "d"]]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(px.line(df, x=df.index, y="c").data[0], secondary_y=False)
        fig.add_trace(px.bar(df, x=df.index, y="v").data[0], secondary_y=True)
        fig.data[1].marker.color = 'red'
        fig.update_layout(title=f"{stock_symbol} Intraday Prices", xaxis_title="Time", yaxis_title="Price")
        fig.update_yaxes(title_text="Volume", secondary_y=True)

        st.plotly_chart(fig)
        st.write(df)