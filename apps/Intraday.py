import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# Alpha Vantage API key
api_key = "V8RWD2L3WZMPFALK"

# Function to get stock data
def get_stock_data(symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={api_key}&outputsize=full&datatype=json"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data["Time Series (1min)"]).T
    df.index = pd.to_datetime(df.index)
    return df

# Streamlit app
st.title("Intraday Stock Price and Volume")

# Get user input for stock symbol
symbol = st.text_input("Enter stock symbol (e.g. MSFT, AAPL, GOOG):")

# Get stock data
if symbol:
    df = get_stock_data(symbol)
    # Filter data to last 5 days
    #df = df.last("5d")

    # Create Plotly chart of close price
    close_df = df[["4. close"]].rename(columns={"4. close": "Close"})
    volume_df = df[["5. volume"]].rename(columns={"5. volume": "Volume"})

    # Create a figure with two subplots
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add the close price subplot
    fig.add_trace(px.line(close_df, x=close_df.index, y="Close").data[0], secondary_y=False)

    # Add the volume subplot as a secondary y-axis
    fig.add_trace(px.bar(volume_df, x=volume_df.index, y="Volume").data[0], secondary_y=True)

    # Update the layout
    fig.update_layout(title="Close Price", yaxis_title="Close Price")
    fig.update_yaxes(title_text="Volume", secondary_y=True)

    # Update the marker color for the volume subplot
    fig.data[1].marker.color = 'red'

    st.plotly_chart(fig)
    # Display data
    st.write(df)