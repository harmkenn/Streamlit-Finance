import streamlit as st
import pandas as pd
import yfinance as yf

st.title("5-Year Hybrid Trigger Strategy (7/14/21-Day Highs & Lows)")
st.subheader("Your Current Portfolio")
# -----------------------------------------
# Load 5 years of TQQQ data
# -----------------------------------------
# Get tickers from session state and split into a list
col1,col2.col3 = st.columns(3)

tickers_list = [t.strip().upper() for t in st.session_state.get("tickers", "").split(",") if t.strip()]

# Ticker selector
with col1:
    ticker = st.selectbox("Select Stock Ticker", tickers_list) if tickers_list else ""

if ticker:
    df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True)
else:
    st.stop()

# -----------------------------------------
# User Portfolio Inputs
# -----------------------------------------

with col2:
    user_cash = st.number_input(
        "Enter your current cash balance ($):",
        min_value=0.0,
        value=100000.0,
        step=100.0,
        format="%.2f"
)
with col3:
    user_shares = st.number_input(
        f"Enter your current {ticker} share count:",
        min_value=0.0,
        value=2000.0,
        step=1.0,
        format="%.4f"
    )

# -----------------------------------------
# Strategy parameters
# -----------------------------------------
st.sidebar.header("Strategy Parameters")

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

if df.empty:
    st.error("No data returned from Yahoo Finance.")
    st.stop()

df = df.astype(float)

# Rolling windows for triggers
df["Low7"] = df["Low"].rolling(7).min()
df["High7"] = df["High"].rolling(7).max()
df["Low14"] = df["Low"].rolling(14).min()
df["High14"] = df["High"].rolling(14).max()
df["Low21"] = df["Low"].rolling(21).min()
df["High21"] = df["High"].rolling(21).max()

# Trend filter MA
df[f"MA{trend_ma_period}"] = df["Close"].rolling(trend_ma_period).mean()

df = df.dropna()

# -----------------------------------------
# Strategy state
# -----------------------------------------
cash = float(user_cash)
shares = float(user_shares)
last_action = None  # "BUY", "SELL", or None

trades = []
portfolio = []

# -----------------------------------------
# Helper: Display next expected triggers
# -----------------------------------------
def show_next_triggers():
    if last_action == "BUY":
        next_action = "SELL"
        next_triggers = ["B: 7-day high", "D: 14-day high", "F: 21-day high"]
    elif last_action == "SELL":
        next_action = "BUY"
        next_triggers = ["A: 7-day low", "C: 14-day low", "E: 21-day low"]
    else:
        next_action = "BUY or SELL"
        next_triggers = [
            "A: 7-day low", "C: 14-day low", "E: 21-day low",
            "B: 7-day high", "D: 14-day high", "F: 21-day high"
        ]

    latest_close = df["Close"].iloc[-1].item()
    latest_ma = df[f"MA{trend_ma_period}"].iloc[-1].item()
    trend_ok = latest_close > latest_ma

    st.subheader("Next Expected Triggers")

    # -----------------------------------------
    # Compute next BUY/SELL share quantities
    # -----------------------------------------
    st.subheader("Next Trade Size (Shares & Dollars)")

    latest = df.iloc[-1]

    close = latest["Close"].item()
    low7 = latest["Low7"].item()
    high7 = latest["High7"].item()
    low14 = latest["Low14"].item()
    high14 = latest["High14"].item()
    low21 = latest["Low21"].item()
    high21 = latest["High21"].item()
    trend_ma = latest[f"MA{trend_ma_period}"].item()


    portfolio_value = user_cash + user_shares * close
    position_value = user_shares * close

    # BUY amounts
    buy_amount = buy_risk_pct * portfolio_value if portfolio_value > 0 else 0

    # SELL amounts
    sell_amount = sell_risk_pct * position_value if position_value > 0 else 0

    if last_action == "BUY":
        st.write("**Next allowed action: SELL**")
        st.write(f"- SELL at 7-day high (${high7:.2f}): {sell_amount/high7:.4f} shares")
        st.write(f"- SELL at 14-day high (${high14:.2f}): {sell_amount/high14:.4f} shares")
        st.write(f"- SELL at 21-day high (${high21:.2f}): {sell_amount/high21:.4f} shares")

    elif last_action == "SELL":
        st.write("**Next allowed action: BUY**")
        if close > trend_ma:
            st.write(f"- BUY at 7-day low (${low7:.2f}): {buy_amount/low7:.4f} shares")
            st.write(f"- BUY at 14-day low (${low14:.2f}): {buy_amount/low14:.4f} shares")
            st.write(f"- BUY at 21-day low (${low21:.2f}): {buy_amount/low21:.4f} shares")
        else:
            st.write("❌ Trend filter blocks buys (price below MA)")

    else:
        st.write("**Next allowed action: BUY or SELL**")
        st.write("BUY triggers:")
        if close > trend_ma:
            st.write(f"- BUY at 7-day low (${low7:.2f}): {buy_amount/low7:.4f} shares")
            st.write(f"- BUY at 14-day low (${low14:.2f}): {buy_amount/low14:.4f} shares")
            st.write(f"- BUY at 21-day low (${low21:.2f}): {buy_amount/low21:.4f} shares")
        else:
            st.write("❌ Trend filter blocks buys (price below MA)")

        st.write("SELL triggers:")
        st.write(f"- SELL at 7-day high (${high7:.2f}): {sell_amount/high7:.4f} shares")
        st.write(f"- SELL at 14-day high (${high14:.2f}): {sell_amount/high14:.4f} shares")
        st.write(f"- SELL at 21-day high (${high21:.2f}): {sell_amount/high21:.4f} shares")


    st.write(f"**Next allowed action:** {next_action}")

    if next_action.startswith("BUY"):
        st.write(
            f"**Trend filter:** "
            f"{'✅ Above MA — buys allowed' if trend_ok else '❌ Below MA — buys blocked'}"
        )

show_next_triggers()

# -----------------------------------------
# Run hybrid strategy
# -----------------------------------------
for idx, row in df.iterrows():
    low = row["Low"].item()
    high = row["High"].item()
    close = row["Close"].item()

    low7 = row["Low7"].item()
    high7 = row["High7"].item()
    low14 = row["Low14"].item()
    high14 = row["High14"].item()
    low21 = row["Low21"].item()
    high21 = row["High21"].item()
    trend_ma = row[f"MA{trend_ma_period}"].item()

    portfolio_value = cash + shares * close

    # -------------------------
    # BUY TRIGGERS (A, C, E)
    # -------------------------
    if last_action != "BUY" and close > trend_ma:
        buy_executed = False

        # A: 7-day low
        if low <= low7 and cash > 0:
            portfolio_value = cash + shares * close
            buy_amount = min(cash, buy_risk_pct * portfolio_value)
            if buy_amount > 0:
                qty = buy_amount / low7
                cash -= buy_amount
                shares += qty
                buy_executed = True
                trades.append([idx, "BUY 7-day low", low7, qty, cash, shares])

        # C: 14-day low
        if low <= low14 and cash > 0:
            portfolio_value = cash + shares * close
            buy_amount = min(cash, buy_risk_pct * portfolio_value)
            if buy_amount > 0:
                qty = buy_amount / low14
                cash -= buy_amount
                shares += qty
                buy_executed = True
                trades.append([idx, "BUY 14-day low", low14, qty, cash, shares])

        # E: 21-day low
        if low <= low21 and cash > 0:
            portfolio_value = cash + shares * close
            buy_amount = min(cash, buy_risk_pct * portfolio_value)
            if buy_amount > 0:
                qty = buy_amount / low21
                cash -= buy_amount
                shares += qty
                buy_executed = True
                trades.append([idx, "BUY 21-day low", low21, qty, cash, shares])

        if buy_executed:
            last_action = "BUY"

    # -------------------------
    # SELL TRIGGERS (B, D, F)
    # -------------------------
    if last_action != "SELL" and shares > 0:
        sell_executed = False

        # B: 7-day high
        if high >= high7 and shares > 0:
            position_value = shares * close
            sell_value = sell_risk_pct * position_value
            price = high7
            qty = min(shares, sell_value / price)
            if qty > 0:
                cash += qty * price
                shares -= qty
                sell_executed = True
                trades.append([idx, "SELL 7-day high", price, qty, cash, shares])

        # D: 14-day high
        if high >= high14 and shares > 0:
            position_value = shares * close
            sell_value = sell_risk_pct * position_value
            price = high14
            qty = min(shares, sell_value / price)
            if qty > 0:
                cash += qty * price
                shares -= qty
                sell_executed = True
                trades.append([idx, "SELL 14-day high", price, qty, cash, shares])

        # F: 21-day high
        if high >= high21 and shares > 0:
            position_value = shares * close
            sell_value = sell_risk_pct * position_value
            price = high21
            qty = min(shares, sell_value / price)
            if qty > 0:
                cash += qty * price
                shares -= qty
                sell_executed = True
                trades.append([idx, "SELL 21-day high", price, qty, cash, shares])

        if sell_executed:
            last_action = "SELL"

    portfolio.append([idx, float(cash + shares * close)])

# Final strategy value
final_value = cash + shares * df["Close"].iloc[-1].item()

# -----------------------------------------
# Buy-until-cash-is-gone benchmark
# -----------------------------------------
first_price = df["Close"].iloc[0].item()
buy_hold_shares = user_cash / first_price
buy_hold_value = buy_hold_shares * df["Close"].iloc[-1].item()

# -----------------------------------------
# Display results
# -----------------------------------------
st.subheader("Final Results")
st.write(f"**Hybrid Trigger Strategy Final Value:** ${final_value:,.2f}")
st.write(f"**Buy-Until-Cash-Is-Invested Final Value:** ${buy_hold_value:,.2f}")

# Trade log
st.subheader("Trade Log")
trades_df = pd.DataFrame(
    trades,
    columns=["Date", "Trigger", "Price", "Shares", "CashAfter", "SharesAfter"]
)
st.dataframe(trades_df)

# Portfolio curve
portfolio_df = pd.DataFrame(portfolio, columns=["Date", "Value"])
portfolio_df.set_index("Date", inplace=True)

st.subheader("Portfolio Value Over Time")
st.line_chart(portfolio_df)

st.subheader(f"{ticker} Closing Price")
st.line_chart(df["Close"])
