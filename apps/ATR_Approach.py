import streamlit as st
<<<<<<< HEAD
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# -----------------------------
# FUNCTIONS
# -----------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_stoch_rsi(series, period=14, smooth_k=3, smooth_d=3):
    rsi = compute_rsi(series, period)
    min_rsi = rsi.rolling(period).min()
    max_rsi = rsi.rolling(period).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    k = stoch_rsi.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_vwap(df):
    return (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Intraday Stock Dashboard", layout="wide")
st.title("📈 Advanced Intraday Stock Dashboard (RSI, MACD, Stochastic RSI, EMA & VWAP)")

# -----------------------------
# USER INPUTS
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    tickers_input = st.text_input("Enter stock tickers (comma-separated, e.g., AAPL, MSFT, TQQQ)")
    tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    ticker = st.selectbox("Select Stock Ticker", tickers_list) if tickers_list else None

with col2:
    rsi_period = st.number_input("RSI Period", min_value=5, max_value=50, value=14)

with col3:
    refresh_button = st.button("Refresh")

# -----------------------------
# FETCH DATA
# -----------------------------
if ticker:
    try:
        yf_ticker = yf.Ticker(ticker)
        data = yf_ticker.history(period="5d", interval="5m", prepost=True)
        if data.empty:
            st.error("No data found for this ticker.")
        else:
            data = data.tz_convert("America/New_York")

        # -----------------------------
        # INDICATORS
        # -----------------------------
        data["RSI"] = compute_rsi(data["Close"], period=rsi_period)
        data["EMA_12"] = compute_ema(data["Close"], span=12)
        data["EMA_26"] = compute_ema(data["Close"], span=26)
        data["VWAP"] = compute_vwap(data)
        data["MACD"], data["MACD_signal"], data["MACD_hist"] = compute_macd(data["Close"])
        data["Stoch_K"], data["Stoch_D"] = compute_stoch_rsi(data["Close"])

        latest_price = data["Close"].iloc[-1]
        rsi_latest = data["RSI"].iloc[-1]
        stoch_latest = data["Stoch_K"].iloc[-1]
        vwap_latest = data["VWAP"].iloc[-1]

        # -----------------------------
        # BUY/SELL SIGNALS BASED ON RSI
        # -----------------------------
        if rsi_latest <= 30:
            rsi_signal = "BUY (Oversold)"
        elif rsi_latest >= 70:
            rsi_signal = "SELL (Overbought)"
        else:
            rsi_signal = "Neutral"

        st.markdown(f"**Current Price: ${latest_price:.2f} | RSI: {rsi_latest:.1f} → {rsi_signal}**")

        # -----------------------------
        # INTRADAY BUY/SELL PRICE RECOMMENDATIONS
        # -----------------------------
        intraday_low = data["Close"].iloc[-20:].min()   # last ~100 mins (5-min bars)
        intraday_high = data["Close"].iloc[-20:].max()

        buy_price = None
        sell_price = None

        if rsi_latest <= 30 and stoch_latest <= 20 and latest_price <= vwap_latest:
            buy_price = intraday_low * 0.995
        elif rsi_latest <= 40 and latest_price <= vwap_latest:
            buy_price = intraday_low
=======
import pandas as pd
import yfinance as yf

st.title("TQQQ Trigger Strategy (MA-Adjusted) vs Buy-and-Hold")

# -----------------------------------------
# Strategy parameters with sliders
# -----------------------------------------
initial_cash = 100000
trade_amount = 10000

st.sidebar.header("Strategy Parameters")

drop_pct_5 = st.sidebar.slider("Buy Trigger Drop % (5% default)", 1, 20, 4) / 100
spike_pct_5 = st.sidebar.slider("Sell Trigger Rise % (5% default)", 1, 20, 7) / 100
drop_pct_10 = st.sidebar.slider("Buy Trigger Drop % (10% default)", 1, 30, 8) / 100
spike_pct_10 = st.sidebar.slider("Sell Trigger Rise % (10% default)", 1, 30, 14) / 100

ma_period = st.sidebar.slider("Moving Average Period", 5, 50, 7)

st.write(f"""
Simulating a strategy where:

- Start with **${initial_cash:,}** cash  
- Buy/Sell **${trade_amount:,}** on **{int(drop_pct_5*100)}% or {int(drop_pct_10*100)}% intraday moves**  
- Apply **MA{ma_period} filter**:  
  - *Buy price = min(trigger, MA{ma_period})*  
  - *Sell price = max(trigger, MA{ma_period})*  
""")

# -----------------------------------------
# Load 3 years of TQQQ daily data
# -----------------------------------------
ticker = "TQQQ"
df = yf.download(ticker, period="3y", interval="1d")

if df.empty:
    st.error("Error: No data returned.")
    st.stop()
>>>>>>> 0ad0465c (ATR 01)

        if rsi_latest >= 70 and stoch_latest >= 80 and latest_price >= vwap_latest:
            sell_price = intraday_high * 1.005
        elif rsi_latest >= 60 and latest_price >= vwap_latest:
            sell_price = intraday_high

<<<<<<< HEAD
        if buy_price:
            st.markdown(f"**💚 Recommended Buy Price:** ${buy_price:.2f}")
        if sell_price:
            st.markdown(f"**🔴 Recommended Sell Price:** ${sell_price:.2f}")
        if not buy_price and not sell_price:
            st.markdown("**⚪ No strong Buy/Sell signal currently.**")

        # -----------------------------
        # PRICE CHART WITH EMA & VWAP
        # -----------------------------
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close", line=dict(color="blue")))
        price_fig.add_trace(go.Scatter(x=data.index, y=data["EMA_12"], mode="lines", name="EMA 12", line=dict(color="orange")))
        price_fig.add_trace(go.Scatter(x=data.index, y=data["EMA_26"], mode="lines", name="EMA 26", line=dict(color="purple")))
        price_fig.add_trace(go.Scatter(x=data.index, y=data["VWAP"], mode="lines", name="VWAP", line=dict(color="green", dash="dot")))
        price_fig.update_layout(title=f"{ticker} Price + EMA + VWAP", xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(price_fig)
=======
# Moving average
df[f"MA{ma_period}"] = df["Close"].rolling(ma_period).mean()
>>>>>>> 0ad0465c (ATR 01)

        # -----------------------------
        # RSI CHART
        # -----------------------------
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI"))
        rsi_fig.add_hline(y=70, line_dash="dot")
        rsi_fig.add_hline(y=30, line_dash="dot")
        rsi_fig.update_layout(title=f"{ticker} RSI", yaxis=dict(range=[0,100]))
        st.plotly_chart(rsi_fig)

<<<<<<< HEAD
        # -----------------------------
        # STOCHASTIC RSI
        # -----------------------------
        stoch_fig = go.Figure()
        stoch_fig.add_trace(go.Scatter(x=data.index, y=data["Stoch_K"], name="Stoch %K", line=dict(color="blue")))
        stoch_fig.add_trace(go.Scatter(x=data.index, y=data["Stoch_D"], name="Stoch %D", line=dict(color="orange")))
        stoch_fig.add_hline(y=80, line_dash="dot")
        stoch_fig.add_hline(y=20, line_dash="dot")
        stoch_fig.update_layout(title=f"{ticker} Stochastic RSI", yaxis=dict(range=[0,100]))
        st.plotly_chart(stoch_fig)

        # -----------------------------
        # MACD
        # -----------------------------
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=dict(color="blue")))
        macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD_signal"], name="Signal", line=dict(color="orange")))
        macd_fig.add_bar(x=data.index, y=data["MACD_hist"], name="Histogram", marker_color="grey")
        macd_fig.update_layout(title=f"{ticker} MACD")
        st.plotly_chart(macd_fig)

        # -----------------------------
        # VOLUME CHART
        # -----------------------------
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume"))
        vol_fig.update_layout(title=f"{ticker} Volume")
        st.plotly_chart(vol_fig)

        # -----------------------------
        # RAW DATA
        # -----------------------------
        st.write(data[["Close","EMA_12","EMA_26","VWAP","RSI","Stoch_K","Stoch_D","MACD","MACD_signal","MACD_hist","Volume"]][::-1])

        # -----------------------------
        # REFRESH BUTTON
        # -----------------------------
        if refresh_button:
            st.experimental_rerun()

    except Exception as e:
        st.error(f"Error fetching data: {e}")
=======
# -----------------------------------------
# Next-day trigger preview
# -----------------------------------------
last_close = float(df["Close"].iloc[-1])
ma_last = float(df[f"MA{ma_period}"].iloc[-1])

# 5% triggers
buy_5_price = last_close * (1 - drop_pct_5)
sell_5_price = last_close * (1 + spike_pct_5)

# 10% triggers
buy_10_price = last_close * (1 - drop_pct_10)
sell_10_price = last_close * (1 + spike_pct_10)

st.subheader("Next Day Trigger Preview")
st.write(f"Previous Close: **${last_close:,.2f}**")
st.write(f"{ma_period}-Day Moving Average: **${ma_last:,.2f}**")

# 5% and 10% preview table
preview_df = pd.DataFrame({
    "Trigger Type": ["Buy 5%", "Sell 5%", "Buy 10%", "Sell 10%"],
    "Raw Trigger Price": [buy_5_price, sell_5_price, buy_10_price, sell_10_price],
    f"Adjusted for MA{ma_period}": [
        min(buy_5_price, ma_last),
        max(sell_5_price, ma_last),
        min(buy_10_price, ma_last),
        max(sell_10_price, ma_last)
    ],
    "Shares at Trigger Price": [
        trade_amount / min(buy_5_price, ma_last),
        trade_amount / max(sell_5_price, ma_last),
        trade_amount / min(buy_10_price, ma_last),
        trade_amount / max(sell_10_price, ma_last)
    ]
})

st.dataframe(preview_df.style.format({
    "Raw Trigger Price": "${:,.2f}",
    f"Adjusted for MA{ma_period}": "${:,.2f}",
    "Shares at Trigger Price": "{:,.4f}"
}))

# -----------------------------------------
# Simulate Strategy with MA adjustment
# -----------------------------------------
cash = float(initial_cash)
shares = 0.0
trades = []
portfolio_value_over_time = []

for idx, row in df.iterrows():
    prev_close = float(row["PrevClose"])
    day_low = float(row["Low"])
    day_high = float(row["High"])
    day_close = float(row["Close"])
    ma = float(row[f"MA{ma_period}"])

    # Raw triggers
    buy_5 = prev_close * (1 - drop_pct_5)
    sell_5 = prev_close * (1 + spike_pct_5)
    buy_10 = prev_close * (1 - drop_pct_10)
    sell_10 = prev_close * (1 + spike_pct_10)

    # Adjusted for MA
    buy_5_adj = min(buy_5, ma)
    sell_5_adj = max(sell_5, ma)
    buy_10_adj = min(buy_10, ma)
    sell_10_adj = max(sell_10, ma)

    # BUY 5% (only if cash available)
    if day_low <= buy_5_adj and cash >= trade_amount:
        qty = trade_amount / buy_5_adj
        cash -= trade_amount
        shares += qty
        total_value = cash + shares * buy_5_adj
        trades.append([idx, "BUY 5% (MA-adjusted)", buy_5_adj, qty, cash, shares, total_value])

    # SELL 5%
    if day_high >= sell_5_adj:
        qty = trade_amount / sell_5_adj
        if shares >= qty:
            cash += trade_amount
            shares -= qty
            total_value = cash + shares * sell_5_adj
            trades.append([idx, "SELL 5% (MA-adjusted)", sell_5_adj, qty, cash, shares, total_value])

    # BUY 10% (only if cash available)
    if day_low <= buy_10_adj and cash >= trade_amount:
        qty = trade_amount / buy_10_adj
        cash -= trade_amount
        shares += qty
        total_value = cash + shares * buy_10_adj
        trades.append([idx, "BUY 10% (MA-adjusted)", buy_10_adj, qty, cash, shares, total_value])

    # SELL 10%
    if day_high >= sell_10_adj:
        qty = trade_amount / sell_10_adj
        if shares >= qty:
            cash += trade_amount
            shares -= qty
            total_value = cash + shares * sell_10_adj
            trades.append([idx, "SELL 10% (MA-adjusted)", sell_10_adj, qty, cash, shares, total_value])

    # Record daily portfolio value
    portfolio_value_over_time.append([idx, cash + shares * day_close])

# Final strategy value
final_value = cash + shares * float(df["Close"].iloc[-1])

# -----------------------------------------
# Buy-and-hold comparison
# -----------------------------------------
initial_shares = float(initial_cash / df["Close"].iloc[0])
buy_hold_value = df["Close"] * initial_shares

# -----------------------------------------
# Display Results
# -----------------------------------------
st.subheader("Final Results")
st.write(f"**Strategy Final Value:** ${final_value:,.2f}")
st.write(f"**Buy-and-Hold Final Value:** ${float(buy_hold_value.iloc[-1]):,.2f}")

# Trade log
trades_df = pd.DataFrame(
    trades,
    columns=["Date", "Type", "Execution Price", "Shares", "CashAfter", "SharesAfter", "TotalValue"]
)
st.subheader("Trade Log")
st.dataframe(trades_df)

# Portfolio curves
portfolio_df = pd.DataFrame(portfolio_value_over_time, columns=["Date", "StrategyValue"])
portfolio_df.set_index("Date", inplace=True)
portfolio_df["BuyHoldValue"] = buy_hold_value.values

st.subheader("Portfolio Value vs Buy-and-Hold")
st.line_chart(portfolio_df)

st.subheader("TQQQ Closing Price")
st.line_chart(df["Close"])
>>>>>>> 0ad0465c (ATR 01)
