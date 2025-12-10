import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.title("TQQQ Trend + ATR Trailing Stop vs Buy-and-Hold v2.0")

ticker = "TQQQ"
period = st.sidebar.selectbox("Backtest period", ["3y", "5y", "10y"], index=1)
ma_len = st.sidebar.slider("Trend MA length", 20, 200, 100)
atr_len = st.sidebar.slider("ATR length", 5, 30, 14)
atr_mult_entry = st.sidebar.slider("Min trend strength (ATR/Close max)", 0.01, 0.10, 0.05, 0.005)
atr_mult_stop = st.sidebar.slider("Trailing stop (x ATR)", 1.0, 5.0, 2.0, 0.5)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.5, 5.0, 1.0, 0.5)
initial_cash = 100000

# Data
df = yf.download(ticker, period=period, interval="1d").dropna()
df["TR"] = np.maximum(df["High"] - df["Low"],
                      np.maximum(abs(df["High"] - df["Close"].shift(1)),
                                 abs(df["Low"] - df["Close"].shift(1))))
df["ATR"] = df["TR"].rolling(atr_len).mean()
df[f"MA{ma_len}"] = df["Close"].rolling(ma_len).mean()
df["TrendUp"] = (df["Close"] > df[f"MA{ma_len}"]) & (df[f"MA{ma_len}"].diff() > 0)
df["VolOK"] = (df["ATR"] / df["Close"]) < atr_mult_entry
df = df.dropna()

# Backtest
cash = initial_cash
shares = 0.0
peak_price_for_stop = None
portfolio = []
trades = []

for i in range(len(df)):
    row = df.iloc[i]
    price = float(row["Close"])
    atr = float(row["ATR"])
    trend_ok = bool(row["TrendUp"] and row["VolOK"])

    # Update trailing stop when in position
    if shares > 0:
        peak_price_for_stop = max(peak_price_for_stop or price, price)
        stop_price = peak_price_for_stop - atr_mult_stop * atr
    else:
        stop_price = None
        peak_price_for_stop = None

    # Entry
    if shares == 0 and trend_ok:
        # position size by risk: risk_pct * equity with stop at (entry - x*ATR)
        stop_dist = atr_mult_stop * atr
        if stop_dist > 0:
            dollar_risk = cash * (risk_pct / 100.0)
            qty = min(cash / price, dollar_risk / stop_dist)
            # ensure some minimum size
            qty = max(0.0, qty)
        else:
            qty = 0.0

        cost = qty * price
        if cost > 0 and cost <= cash:
            shares += qty
            cash -= cost
            trades.append([df.index[i], "BUY", price, qty, cash, shares])

            # initialize stop anchor
            peak_price_for_stop = price
            stop_price = price - atr_mult_stop * atr

    # Exit on stop or trend break
    exit_signal = False
    if shares > 0:
        if stop_price and price <= stop_price:
            exit_signal = True
        elif not trend_ok:  # trend broken
            exit_signal = True

    if exit_signal and shares > 0:
        proceeds = shares * price
        cash += proceeds
        trades.append([df.index[i], "SELL", price, shares, cash, 0.0])
        shares = 0.0
        peak_price_for_stop = None
        stop_price = None

    portfolio.append([df.index[i], cash + shares * price])

# Results
port_df = pd.DataFrame(portfolio, columns=["Date", "StrategyValue"]).set_index("Date")
initial_shares = initial_cash / df["Close"].iloc[0]
buy_hold_series = df["Close"] * initial_shares
port_df["BuyHoldValue"] = buy_hold_series.reindex(port_df.index)

st.subheader("Final Results")
st.write(f"Strategy Final Value: ${port_df['StrategyValue'].iloc[-1]:,.2f}")
st.write(f"Buy-and-Hold Final Value: ${port_df['BuyHoldValue'].iloc[-1]:,.2f}")

trade_df = pd.DataFrame(trades, columns=["Date", "Type", "Price", "Shares", "CashAfter", "SharesAfter"])
st.subheader("Trades")
st.dataframe(trade_df)

st.subheader("Portfolio Value vs Buy-and-Hold")
st.line_chart(port_df)

st.subheader("TQQQ Close, MA, and ATR")
viz = df[[f"MA{ma_len}", "Close", "ATR"]].copy()
st.line_chart(viz)
