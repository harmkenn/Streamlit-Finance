import streamlit as st
import pandas as pd
import yfinance as yf

st.title("TQQQ 5-Year Strategy Comparison: Hybrid High/Low vs MA-Adjusted Drop/Spike")

initial_cash = 100000

# -----------------------------------------
# Sidebar parameters
# -----------------------------------------
st.sidebar.header("Hybrid High/Low Strategy Parameters (Strategy 1)")

buy_risk_pct = st.sidebar.slider(
    "Buy position size (% of portfolio per buy trigger)",
    min_value=1,
    max_value=25,
    value=10
) / 100.0

sell_risk_pct = st.sidebar.slider(
    "Sell position size (% of position per sell trigger)",
    min_value=1,
    max_value=50,
    value=10
) / 100.0

trend_ma_period = st.sidebar.slider(
    "Trend filter MA period (days)",
    min_value=20,
    max_value=200,
    value=50
)

st.sidebar.markdown("---")
st.sidebar.header("MA-Adjusted Drop/Spike Strategy Parameters (Strategy 2)")

trade_amount = st.sidebar.number_input(
    "Fixed trade size ($ per trade)",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=1000
)

drop_pct_5 = st.sidebar.slider("Buy Trigger Drop % (5% default)", 1, 20, 5) / 100
spike_pct_5 = st.sidebar.slider("Sell Trigger Rise % (5% default)", 1, 20, 5) / 100
drop_pct_10 = st.sidebar.slider("Buy Trigger Drop % (10% default)", 1, 30, 10) / 100
spike_pct_10 = st.sidebar.slider("Sell Trigger Rise % (10% default)", 1, 30, 10) / 100

ma_period_2 = st.sidebar.slider("MA Period (Drop/Spike Strategy)", 5, 50, 7)

st.write(f"""
Comparing two strategies on **TQQQ** over the last **5 years**:

**Strategy 1 – Hybrid High/Low with Trend Filter**

- Start with **${initial_cash:,}** cash
- Uses 6 triggers:
  - A: Buy at 7-day low  
  - B: Sell at 7-day high  
  - C: Buy at 14-day low  
  - D: Sell at 14-day high  
  - E: Buy at 21-day low  
  - F: Sell at 21-day high  
- Buys only when price is above **MA{trend_ma_period}**
- Each BUY uses **{buy_risk_pct*100:.1f}%** of portfolio (if cash allows)
- Each SELL closes **{sell_risk_pct*100:.1f}%** of current position
- No repeated BUYs until a SELL, and vice versa

**Strategy 2 – MA-Adjusted 5% / 10% Drop/Spike**

- Start with **${initial_cash:,}** cash
- Trade size: **${trade_amount:,}** per trade
- Raw triggers based on previous close:
  - Buy at **-{drop_pct_5*100:.1f}%** and **-{drop_pct_10*100:.1f}%**
  - Sell at **+{spike_pct_5*100:.1f}%** and **+{spike_pct_10*100:.1f}%**
- Triggers adjusted by **MA{ma_period_2}**:
  - Buy price = min(raw trigger, MA)
  - Sell price = max(raw trigger, MA)

Both are compared to a **buy-and-hold** benchmark (all-in on day 1).
""")

# -----------------------------------------
# Load 5 years of TQQQ data
# -----------------------------------------
ticker = "TQQQ"
df = yf.download(ticker, period="5y", interval="1d")

if df.empty:
    st.error("Error: No data returned.")
    st.stop()

df = df.astype(float)

# For Strategy 1: rolling highs/lows and trend MA
df["Low7"] = df["Low"].rolling(7).min()
df["High7"] = df["High"].rolling(7).max()
df["Low14"] = df["Low"].rolling(14).min()
df["High14"] = df["High"].rolling(14).max()
df["Low21"] = df["Low"].rolling(21).min()
df["High21"] = df["High"].rolling(21).max()
df[f"MA{trend_ma_period}"] = df["Close"].rolling(trend_ma_period).mean()

# For Strategy 2: previous close and its own MA
df["PrevClose"] = df["Close"].shift(1)
df[f"MA{ma_period_2}"] = df["Close"].rolling(ma_period_2).mean()

df = df.dropna()

# -----------------------------------------
# Strategy 1: Hybrid High/Low with Trend Filter
# -----------------------------------------
cash1 = float(initial_cash)
shares1 = 0.0
last_action1 = None  # "BUY", "SELL", or None

trades1 = []
portfolio1 = []

for idx, row in df.iterrows():
    low = float(row["Low"])
    high = float(row["High"])
    close = float(row["Close"])

    low7 = float(row["Low7"])
    high7 = float(row["High7"])
    low14 = float(row["Low14"])
    high14 = float(row["High14"])
    low21 = float(row["Low21"])
    high21 = float(row["High21"])
    trend_ma = float(row[f"MA{trend_ma_period}"])

    portfolio_value = cash1 + shares1 * close

    # BUY TRIGGERS (A, C, E) - only if above trend MA
    if last_action1 != "BUY" and close > trend_ma:
        buy_executed = False

        # A: 7-day low
        if low <= low7 and cash1 > 0:
            portfolio_value = cash1 + shares1 * close
            buy_amount = min(cash1, buy_risk_pct * portfolio_value)
            if buy_amount > 0:
                qty = buy_amount / low7
                cash1 -= buy_amount
                shares1 += qty
                buy_executed = True
                trades1.append([idx, "BUY 7-day low", low7, qty, cash1, shares1])

        # C: 14-day low
        if low <= low14 and cash1 > 0:
            portfolio_value = cash1 + shares1 * close
            buy_amount = min(cash1, buy_risk_pct * portfolio_value)
            if buy_amount > 0:
                qty = buy_amount / low14
                cash1 -= buy_amount
                shares1 += qty
                buy_executed = True
                trades1.append([idx, "BUY 14-day low", low14, qty, cash1, shares1])

        # E: 21-day low
        if low <= low21 and cash1 > 0:
            portfolio_value = cash1 + shares1 * close
            buy_amount = min(cash1, buy_risk_pct * portfolio_value)
            if buy_amount > 0:
                qty = buy_amount / low21
                cash1 -= buy_amount
                shares1 += qty
                buy_executed = True
                trades1.append([idx, "BUY 21-day low", low21, qty, cash1, shares1])

        if buy_executed:
            last_action1 = "BUY"

    # SELL TRIGGERS (B, D, F)
    if last_action1 != "SELL" and shares1 > 0:
        sell_executed = False

        # B: 7-day high
        if high >= high7 and shares1 > 0:
            position_value = shares1 * close
            sell_value = sell_risk_pct * position_value
            price = high7
            qty = min(shares1, sell_value / price)
            if qty > 0:
                cash1 += qty * price
                shares1 -= qty
                sell_executed = True
                trades1.append([idx, "SELL 7-day high", price, qty, cash1, shares1])

        # D: 14-day high
        if high >= high14 and shares1 > 0:
            position_value = shares1 * close
            sell_value = sell_risk_pct * position_value
            price = high14
            qty = min(shares1, sell_value / price)
            if qty > 0:
                cash1 += qty * price
                shares1 -= qty
                sell_executed = True
                trades1.append([idx, "SELL 14-day high", price, qty, cash1, shares1])

        # F: 21-day high
        if high >= high21 and shares1 > 0:
            position_value = shares1 * close
            sell_value = sell_risk_pct * position_value
            price = high21
            qty = min(shares1, sell_value / price)
            if qty > 0:
                cash1 += qty * price
                shares1 -= qty
                sell_executed = True
                trades1.append([idx, "SELL 21-day high", price, qty, cash1, shares1])

        if sell_executed:
            last_action1 = "SELL"

    portfolio1.append([idx, float(cash1 + shares1 * close)])

final_value1 = float(cash1 + shares1 * df["Close"].iloc[-1])

# -----------------------------------------
# Strategy 2: MA-Adjusted Drop/Spike
# -----------------------------------------
cash2 = float(initial_cash)
shares2 = 0.0
trades2 = []
portfolio2 = []

for idx, row in df.iterrows():
    prev_close = float(row["PrevClose"])
    day_low = float(row["Low"])
    day_high = float(row["High"])
    day_close = float(row["Close"])
    ma2 = float(row[f"MA{ma_period_2}"])

    # Raw triggers
    buy_5 = prev_close * (1 - drop_pct_5)
    sell_5 = prev_close * (1 + spike_pct_5)
    buy_10 = prev_close * (1 - drop_pct_10)
    sell_10 = prev_close * (1 + spike_pct_10)

    # MA-adjusted trigger levels
    buy_5_adj = min(buy_5, ma2)
    sell_5_adj = max(sell_5, ma2)
    buy_10_adj = min(buy_10, ma2)
    sell_10_adj = max(sell_10, ma2)

    # BUY 5% (if cash available)
    if day_low <= buy_5_adj and cash2 >= trade_amount:
        qty = trade_amount / buy_5_adj
        cash2 -= trade_amount
        shares2 += qty
        total_value = cash2 + shares2 * buy_5_adj
        trades2.append([idx, "BUY 5% (MA-adjusted)", buy_5_adj, qty, cash2, shares2, total_value])

    # SELL 5%
    if day_high >= sell_5_adj and shares2 > 0:
        qty = trade_amount / sell_5_adj
        if shares2 >= qty:
            cash2 += trade_amount
            shares2 -= qty
            total_value = cash2 + shares2 * sell_5_adj
            trades2.append([idx, "SELL 5% (MA-adjusted)", sell_5_adj, qty, cash2, shares2, total_value])

    # BUY 10% (if cash available)
    if day_low <= buy_10_adj and cash2 >= trade_amount:
        qty = trade_amount / buy_10_adj
        cash2 -= trade_amount
        shares2 += qty
        total_value = cash2 + shares2 * buy_10_adj
        trades2.append([idx, "BUY 10% (MA-adjusted)", buy_10_adj, qty, cash2, shares2, total_value])

    # SELL 10%
    if day_high >= sell_10_adj and shares2 > 0:
        qty = trade_amount / sell_10_adj
        if shares2 >= qty:
            cash2 += trade_amount
            shares2 -= qty
            total_value = cash2 + shares2 * sell_10_adj
            trades2.append([idx, "SELL 10% (MA-adjusted)", sell_10_adj, qty, cash2, shares2, total_value])

    portfolio2.append([idx, cash2 + shares2 * day_close])

final_value2 = cash2 + shares2 * float(df["Close"].iloc[-1])

# -----------------------------------------
# Buy-and-hold benchmark
# -----------------------------------------
first_price = float(df["Close"].iloc[0])
bh_shares = initial_cash / first_price
bh_series = df["Close"] * bh_shares
final_value_bh = float(bh_series.iloc[-1])

# -----------------------------------------
# Display results
# -----------------------------------------
st.subheader("Final Results (5-Year Backtest)")

results_df = pd.DataFrame({
    "Strategy": [
        "Hybrid High/Low with Trend (Strategy 1)",
        "MA-Adjusted Drop/Spike (Strategy 2)",
        "Buy-and-Hold"
    ],
    "Final Value ($)": [
        final_value1,
        final_value2,
        final_value_bh
    ]
})

st.dataframe(results_df.style.format({"Final Value ($)": "${:,.2f}"}))

# Trade logs
st.subheader("Trade Log – Strategy 1 (Hybrid High/Low)")
trades1_df = pd.DataFrame(
    trades1,
    columns=["Date", "Type", "Price", "Shares", "CashAfter", "SharesAfter"]
)
st.dataframe(trades1_df)

st.subheader("Trade Log – Strategy 2 (MA-Adjusted Drop/Spike)")
trades2_df = pd.DataFrame(
    trades2,
    columns=["Date", "Type", "Execution Price", "Shares", "CashAfter", "SharesAfter", "TotalValue"]
)
st.dataframe(trades2_df)

# Equity curves
portfolio1_df = pd.DataFrame(portfolio1, columns=["Date", "Strategy1"])
portfolio1_df.set_index("Date", inplace=True)

portfolio2_df = pd.DataFrame(portfolio2, columns=["Date", "Strategy2"])
portfolio2_df.set_index("Date", inplace=True)

equity_df = pd.DataFrame(index=df.index)
equity_df["Strategy1"] = portfolio1_df["Strategy1"]
equity_df["Strategy2"] = portfolio2_df["Strategy2"]
equity_df["BuyAndHold"] = bh_series

st.subheader("Portfolio Value Over Time (All Strategies)")
st.line_chart(equity_df)

st.subheader("TQQQ Closing Price")
st.line_chart(df["Close"])
