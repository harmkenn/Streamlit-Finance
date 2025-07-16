import streamlit as st
import pandas as pd
import datetime
import requests

st.title("MSTY Daily OHLC from Investing.com")

# Calculate date range
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365)
date_str = start_date.strftime("%m/%d/%Y") + "-" + end_date.strftime("%m/%d/%Y")

def fetch_investing_evenportunhistorical(symbol, date_range):
    url = f"https://www.investing.com/etfs/{symbol.lower()}-historical-data"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    dfs = pd.read_html(res.text, attrs={"id": "curr_table"})
    df = dfs[0]

    # Clean and transform
    df = df.rename(columns={
        "Price": "Close", "Vol.": "Volume",
        "Open": "Open", "High": "High", "Low": "Low", "Change %": "Change"
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
    df[['Open','High','Low','Close']] = df[['Open','High','Low','Close']].apply(lambda col: col.str.replace(',','').astype(float))
    return df.set_index("Date")[["Open","High","Low","Close","Volume"]]

try:
    df = fetch_investing_evenportunhistorical("msty", date_str)
    st.success("Data fetched from Investing.com")
    st.dataframe(df)

    csv = df.to_csv().encode('utf-8')
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="MSTY_investing_ohlc.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"Failed to fetch data: {e}")
