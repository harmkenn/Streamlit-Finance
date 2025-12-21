import streamlit as st
import pandas as pd
import yfinance as yf

# ============================================================
# Page title
# ============================================================
st.title("5-Year Hybrid Trigger Strategy (7/14/21-Day Highs & Lows, ATR-MA Channel)")
st.subheader("Your Current Portfolio")

# ============================================================
# Layout and ticker selection
# ============================================================
col1, col2, col3 = st.columns(3)

# Get tickers from session state and split into a list
tickers_list = [
    t.strip().upper()
    for t in st.session_state.get("tickers", "").split(",")
    if t.strip()
]

with col1:
    ticker = st.selectbox("Select Stock Ticker", tickers_list) if tickers_list else ""

if not ticker:
    st.stop()

# ============================================================
# Sidebar strategy parameters
# ============================================================
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

atr_period = st.sidebar.slider(
    "ATR period (days)",
    min_value=5,
    max_value=50,
    value=14
)

# ============================================================
# User portfolio inputs
# ============================================================
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

# ============================================================
# Data loading & indicator computation
# ============================================================
@st.cache_data(ttl=600)
def load_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    return df


def compute_indicators(
    df: pd.DataFrame,
    ma_period: int,
    atr_period: int
) -> pd.DataFrame:
    df = df.copy().astype(float)

    # Rolling windows for triggers
    df["Low7"] = df["Low"].rolling(7).min()
    df["High7"] = df["High"].rolling(7).max()
    df["Low14"] = df["Low"].rolling(14).min()
    df["High14"] = df["High"].rolling(14).max()
    df["Low21"] = df["Low"].rolling(21).min()
    df["High21"] = df["High"].rolling(21).max()

    # Trend filter MA
    df[f"MA{ma_period}"] = df["Close"].rolling(ma_period).mean()

    # ATR (Average True Range)
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
    df["L-PC"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(atr_period).mean()

    # Volatility-adjusted MA channel
    df["UpperBand"] = df[f"MA{ma_period}"] + df["ATR"]
    df["LowerBand"] = df[f"MA{ma_period}"] - df["ATR"]

    # Clean up helper columns
    df = df.drop(columns=["H-L", "H-PC", "L-PC", "TR"])

    # Drop rows where indicators are not ready
    df = df.dropna()

    return df


df = load_data(ticker)

if df.empty:
    st.error("No data returned from Yahoo Finance.")
    st.stop()

df = compute_indicators(df, trend_ma_period, atr_period)

# ============================================================
# Strategy core
# ============================================================
def run_strategy(
    df: pd.DataFrame,
    start_cash: float,
    start_shares: float,
    buy_risk_pct: float,
    sell_risk_pct: float
):
    cash = float(start_cash)
    shares = float(start_shares)
    last_action = None  # "BUY", "SELL", or None

    trades = []
    portfolio = []

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

        upper = float(row["UpperBand"])
        lower = float(row["LowerBand"])

        portfolio_value = cash + shares * close

        # ====================================================
        # BUY TRIGGERS (A, C, E)
        # Only when price is BELOW the lower volatility band
        # ====================================================
        if last_action != "BUY" and close < lower:
            buy_executed = False

            # A: 7-day low
            if low <= low7 and cash > 0:
                buy_amount = min(cash, buy_risk_pct * portfolio_value)
                if buy_amount > 0:
                    qty = buy_amount / low7
                    cash -= buy_amount
                    shares += qty
                    buy_executed = True
                    trades.append([idx, "BUY 7-day low", low7, qty, cash, shares])

            # C: 14-day low
            if low <= low14 and cash > 0:
                buy_amount = min(cash, buy_risk_pct * portfolio_value)
                if buy_amount > 0:
                    qty = buy_amount / low14
                    cash -= buy_amount
                    shares += qty
                    buy_executed = True
                    trades.append([idx, "BUY 14-day low", low14, qty, cash, shares])

            # E: 21-day low
            if low <= low21 and cash > 0:
                buy_amount = min(cash, buy_risk_pct * portfolio_value)
                if buy_amount > 0:
                    qty = buy_amount / low21
                    cash -= buy_amount
                    shares += qty
                    buy_executed = True
                    trades.append([idx, "BUY 21-day low", low21, qty, cash, shares])

            if buy_executed:
                last_action = "BUY"

        # ====================================================
        # SELL TRIGGERS (B, D, F)
        # Only when price is ABOVE the upper volatility band
        # ====================================================
        if last_action != "SELL" and shares > 0 and close > upper:
            sell_executed = False

            # B: 7-day high
            if high >= high7 and shares > 0:
                sell_value = sell_risk_pct * (shares * close)
                price = high7
                qty = min(shares, sell_value / price)
                if qty > 0:
                    cash += qty * price
                    shares -= qty
                    sell_executed = True
                    trades.append([idx, "SELL 7-day high", price, qty, cash, shares])

            # D: 14-day high
            if high >= high14 and shares > 0:
                sell_value = sell_risk_pct * (shares * close)
                price = high14
                qty = min(shares, sell_value / price)
                if qty > 0:
                    cash += qty * price
                    shares -= qty
                    sell_executed = True
                    trades.append([idx, "SELL 14-day high", price, qty, cash, shares])

            # F: 21-day high
            if high >= high21 and shares > 0:
                sell_value = sell_risk_pct * (shares * close)
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

    final_value = float(cash + shares * df["Close"].iloc[-1].item())

    results = {
        "final_value": final_value,
        "cash": cash,
        "shares": shares,
        "trades": trades,
        "portfolio": portfolio,
    }
    return results


# ============================================================
# Helper: Next triggers display (using latest row)
# ============================================================
def show_next_triggers(df: pd.DataFrame):
    latest = df.iloc[-1]

    close = float(latest["Close"])
    low7 = float(latest["Low7"])
    high7 = float(latest["High7"])
    low14 = float(latest["Low14"])
    high14 = float(latest["High14"])
    low21 = float(latest["Low21"])
    high21 = float(latest["High21"])
    ma_val = float(latest[f"MA{trend_ma_period}"])
    upper = float(latest["UpperBand"])
    lower = float(latest["LowerBand"])

    portfolio_value = user_cash + user_shares * close
    position_value = user_shares * close

    buy_amount = buy_risk_pct * portfolio_value if portfolio_value > 0 else 0
    sell_amount = sell_risk_pct * position_value if position_value > 0 else 0

    st.subheader("Next Expected Triggers (Based on ATR-MA Channel)")

    with col1:
        st.write(f"**Latest Close:** ${close:.2f}")
        st.write(f"**MA ({trend_ma_period}):** ${ma_val:.2f}")
        st.write(f"**Lower Band:** ${lower:.2f}")
        st.write(f"**Upper Band:** ${upper:.2f}")

    with col2:
        st.write("**Potential BUY levels (if price < lower band):**")
        st.write(f"- At 7-day low (${low7:.2f}): {buy_amount / low7:.4f} shares" if buy_amount > 0 else "- No buy capital")
        st.write(f"- At 14-day low (${low14:.2f}): {buy_amount / low14:.4f} shares" if buy_amount > 0 else "")
        st.write(f"- At 21-day low (${low21:.2f}): {buy_amount / low21:.4f} shares" if buy_amount > 0 else "")

    with col3:
        st.write("**Potential SELL levels (if price > upper band):**")
        if sell_amount > 0:
            st.write(f"- At 7-day high (${high7:.2f}): {sell_amount / high7:.4f} shares")
            st.write(f"- At 14-day high (${high14:.2f}): {sell_amount / high14:.4f} shares")
            st.write(f"- At 21-day high (${high21:.2f}): {sell_amount / high21:.4f} shares")
        else:
            st.write("No shares to sell.")


show_next_triggers(df)

# ============================================================
# Run strategy
# ============================================================
results = run_strategy(
    df=df,
    start_cash=user_cash,
    start_shares=user_shares,
    buy_risk_pct=buy_risk_pct,
    sell_risk_pct=sell_risk_pct,
)

final_value = results["final_value"]
trades = results["trades"]
portfolio = results["portfolio"]

# ============================================================
# Buy-until-cash-is-gone benchmark
# ============================================================
first_price = float(df["Close"].iloc[0])
buy_hold_shares = user_cash / first_price
buy_hold_value = buy_hold_shares * float(df["Close"].iloc[-1])

# ============================================================
# Display results
# ============================================================
with col3:
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

# ============================================================
# Price chart with MA and ATR channel
# ============================================================
st.subheader(f"{ticker} Closing Price with MA & ATR Channel")

price_plot_df = df[[f"MA{trend_ma_period}", "UpperBand", "LowerBand", "Close"]].copy()
price_plot_df.columns = [
    f"MA{trend_ma_period}",
    "UpperBand",
    "LowerBand",
    "Close",
]

st.line_chart(price_plot_df)
