import streamlit as st
import pandas as pd
import yfinance as yf

# ============================================================
# Page title
# ============================================================
st.title("5-Year Hybrid Trigger Strategy (7/21/63 Highs & Lows, ATR-MA Channel)")
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
    st.warning("Please set tickers in session_state['tickers'] (comma-separated) to begin.")
    st.stop()

# ============================================================
# Sidebar strategy parameters
# ============================================================
st.sidebar.header("Strategy Parameters")

buy_risk_pct = st.sidebar.slider(
    "Base buy position size (% of portfolio per buy trigger)",
    min_value=1,
    max_value=25,
    value=10
) / 100.0

sell_risk_pct = st.sidebar.slider(
    "Base sell position size (% of position per sell trigger)",
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

st.sidebar.markdown("---")
st.sidebar.subheader("Scaled Risk Multipliers")

# All six sliders in one section (Option 2)
buy_mult_7 = st.sidebar.slider(
    "Buy 7-day trigger multiplier",
    min_value=0.1,
    max_value=3.0,
    value=1.0,
    step=0.1
)
buy_mult_21 = st.sidebar.slider(
    "Buy 21-day trigger multiplier",
    min_value=0.1,
    max_value=3.0,
    value=1.2,
    step=0.1
)
buy_mult_63 = st.sidebar.slider(
    "Buy 63-day trigger multiplier",
    min_value=0.1,
    max_value=3.0,
    value=1.4,
    step=0.1
)

sell_mult_7 = st.sidebar.slider(
    "Sell 7-day trigger multiplier",
    min_value=0.1,
    max_value=3.0,
    value=0.5,
    step=0.1
)
sell_mult_21 = st.sidebar.slider(
    "Sell 21-day trigger multiplier",
    min_value=0.1,
    max_value=3.0,
    value=1.0,
    step=0.1
)
sell_mult_63 = st.sidebar.slider(
    "Sell 63-day trigger multiplier",
    min_value=0.1,
    max_value=3.0,
    value=1.5,
    step=0.1
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

    # Rolling windows for triggers: 7, 21, 63
    df["Low7"] = df["Low"].rolling(7).min()
    df["High7"] = df["High"].rolling(7).max()

    df["Low21"] = df["Low"].rolling(21).min()
    df["High21"] = df["High"].rolling(21).max()

    df["Low63"] = df["Low"].rolling(63).min()
    df["High63"] = df["High"].rolling(63).max()

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
    base_buy_risk: float,
    base_sell_risk: float,
    buy_mult_7: float,
    buy_mult_21: float,
    buy_mult_63: float,
    sell_mult_7: float,
    sell_mult_21: float,
    sell_mult_63: float,
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
        low21 = float(row["Low21"])
        high21 = float(row["High21"])
        low63 = float(row["Low63"])
        high63 = float(row["High63"])

        upper = float(row["UpperBand"])
        lower = float(row["LowerBand"])

        portfolio_value = cash + shares * close
        position_value = shares * close

        # ====================================================
        # BUY TRIGGERS (7, 21, 63 lows)
        # Only when price is BELOW the lower volatility band
        # ====================================================
        if last_action != "BUY" and close < lower:
            buy_executed = False

            # 7-day low (tactical)
            if low <= low7 and cash > 0:
                eff_buy_risk = base_buy_risk * buy_mult_7
                buy_amount = min(cash, eff_buy_risk * portfolio_value)
                if buy_amount > 0:
                    qty = buy_amount / low7
                    cash -= buy_amount
                    shares += qty
                    buy_executed = True
                    trades.append([idx, "BUY 7-day low", low7, qty, cash, shares])

            # 21-day low (swing)
            if low <= low21 and cash > 0:
                eff_buy_risk = base_buy_risk * buy_mult_21
                buy_amount = min(cash, eff_buy_risk * portfolio_value)
                if buy_amount > 0:
                    qty = buy_amount / low21
                    cash -= buy_amount
                    shares += qty
                    buy_executed = True
                    trades.append([idx, "BUY 21-day low", low21, qty, cash, shares])

            # 63-day low (major extreme)
            if low <= low63 and cash > 0:
                eff_buy_risk = base_buy_risk * buy_mult_63
                buy_amount = min(cash, eff_buy_risk * portfolio_value)
                if buy_amount > 0:
                    qty = buy_amount / low63
                    cash -= buy_amount
                    shares += qty
                    buy_executed = True
                    trades.append([idx, "BUY 63-day low", low63, qty, cash, shares])

            if buy_executed:
                last_action = "BUY"

        # ====================================================
        # SELL TRIGGERS (7, 21, 63 highs)
        # Only when price is ABOVE the upper volatility band
        # ====================================================
        if last_action != "SELL" and shares > 0 and close > upper:
            sell_executed = False

            # 7-day high (tactical trim)
            if high >= high7 and shares > 0:
                eff_sell_risk = base_sell_risk * sell_mult_7
                sell_value = eff_sell_risk * position_value
                price = high7
                qty = min(shares, sell_value / price) if price > 0 else 0
                if qty > 0:
                    cash += qty * price
                    shares -= qty
                    sell_executed = True
                    trades.append([idx, "SELL 7-day high", price, qty, cash, shares])

            # 21-day high (swing trim)
            if high >= high21 and shares > 0:
                eff_sell_risk = base_sell_risk * sell_mult_21
                sell_value = eff_sell_risk * position_value
                price = high21
                qty = min(shares, sell_value / price) if price > 0 else 0
                if qty > 0:
                    cash += qty * price
                    shares -= qty
                    sell_executed = True
                    trades.append([idx, "SELL 21-day high", price, qty, cash, shares])

            # 63-day high (major unload)
            if high >= high63 and shares > 0:
                eff_sell_risk = base_sell_risk * sell_mult_63
                sell_value = eff_sell_risk * position_value
                price = high63
                qty = min(shares, sell_value / price) if price > 0 else 0
                if qty > 0:
                    cash += qty * price
                    shares -= qty
                    sell_executed = True
                    trades.append([idx, "SELL 63-day high", price, qty, cash, shares])

            if sell_executed:
                last_action = "SELL"

        portfolio.append([idx, float(cash + shares * close)])

    last_close = float(df["Close"].iloc[-1])
    final_value = float(cash + shares * last_close)

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
    low21 = float(latest["Low21"])
    high21 = float(latest["High21"])
    low63 = float(latest["Low63"])
    high63 = float(latest["High63"])
    ma_val = float(latest[f"MA{trend_ma_period}"])
    upper = float(latest["UpperBand"])
    lower = float(latest["LowerBand"])

    portfolio_value = user_cash + user_shares * close
    position_value = user_shares * close

    # Base risk amounts just for reference; actual trades use scaled risk.
    base_buy_amount = buy_risk_pct * portfolio_value if portfolio_value > 0 else 0
    base_sell_amount = sell_risk_pct * position_value if position_value > 0 else 0

    st.subheader("Next Expected Triggers (Based on ATR-MA Channel & 7/21/63 Structure)")

    with col1:
        st.write(f"**Latest Close:** ${close:.2f}")
        st.write(f"**MA ({trend_ma_period}):** ${ma_val:.2f}")
        st.write(f"**Lower Band:** ${lower:.2f}")
        st.write(f"**Upper Band:** ${upper:.2f}")

    with col2:
        st.write("**Potential BUY levels (if price < lower band):**")
        if base_buy_amount > 0:
            st.write(
                f"- 7-day low (${low7:.2f}), multiplier {buy_mult_7:.1f}×: "
                f"{(base_buy_amount * buy_mult_7) / low7:.4f} shares"
            )
            st.write(
                f"- 21-day low (${low21:.2f}), multiplier {buy_mult_21:.1f}×: "
                f"{(base_buy_amount * buy_mult_21) / low21:.4f} shares"
            )
            st.write(
                f"- 63-day low (${low63:.2f}), multiplier {buy_mult_63:.1f}×: "
                f"{(base_buy_amount * buy_mult_63) / low63:.4f} shares"
            )
        else:
            st.write("- No buy capital available.")

    with col3:
        st.write("**Potential SELL levels (if price > upper band):**")
        if base_sell_amount > 0 and user_shares > 0:
            st.write(
                f"- 7-day high (${high7:.2f}), multiplier {sell_mult_7:.1f}×: "
                f"{(base_sell_amount * sell_mult_7) / high7:.4f} shares"
            )
            st.write(
                f"- 21-day high (${high21:.2f}), multiplier {sell_mult_21:.1f}×: "
                f"{(base_sell_amount * sell_mult_21) / high21:.4f} shares"
            )
            st.write(
                f"- 63-day high (${high63:.2f}), multiplier {sell_mult_63:.1f}×: "
                f"{(base_sell_amount * sell_mult_63) / high63:.4f} shares"
            )
        else:
            st.write("No shares to sell or position too small.")


show_next_triggers(df)

# ============================================================
# Run strategy
# ============================================================
results = run_strategy(
    df=df,
    start_cash=user_cash,
    start_shares=user_shares,
    base_buy_risk=buy_risk_pct,
    base_sell_risk=sell_risk_pct,
    buy_mult_7=buy_mult_7,
    buy_mult_21=buy_mult_21,
    buy_mult_63=buy_mult_63,
    sell_mult_7=sell_mult_7,
    sell_mult_21=sell_mult_21,
    sell_mult_63=sell_mult_63,
)

final_value = float(results["final_value"])
trades = results["trades"]
portfolio = results["portfolio"]

# ============================================================
# Buy-until-cash-is-gone benchmark
# ============================================================
first_price = float(df["Close"].iloc[0])
last_price = float(df["Close"].iloc[-1])
buy_hold_shares = user_cash / first_price if first_price > 0 else 0.0
buy_hold_value = float(buy_hold_shares * last_price)

# ============================================================
# Display results
# ============================================================
with col3:
    st.subheader("Final Results")
    st.write(f"**Hybrid Trigger Strategy Final Value:** ${final_value:,.2f}")
    st.write(f"**Buy-Until-Cash-Is-Invested Final Value:** ${buy_hold_value:,.2f}")

# Trade log
st.subheader("Trade Log")
if trades:
    trades_df = pd.DataFrame(
        trades,
        columns=["Date", "Trigger", "Price", "Shares", "CashAfter", "SharesAfter"]
    )
    st.dataframe(trades_df)
else:
    st.write("No trades executed with the current parameters.")

# Portfolio curve
portfolio_df = pd.DataFrame(portfolio, columns=["Date", "Value"])
portfolio_df.set_index("Date", inplace=True)

st.subheader("Portfolio Value Over Time")
if not portfolio_df.empty:
    st.line_chart(portfolio_df)
else:
    st.write("No portfolio history to display.")

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

# ============================================================
# Optional: Strategy explanation expander
# ============================================================
with st.expander("Strategy Explanation (ATR-MA + 7/21/63 Hybrid Trigger System)"):
    st.markdown("""
This strategy uses a volatility-adjusted moving average channel (MA ± ATR) to define when price is
**unusually weak or strong**, and then reacts only at **7/21/63-day highs and lows**.

- **Buy conditions:** Price is below the lower band *and* hits a 7, 21, or 63-day low.
- **Sell conditions:** Price is above the upper band *and* hits a 7, 21, or 63-day high.
- **Scaling:** 7-day signals are small, 21-day are normal, 63-day are larger (via your sidebar multipliers).
- **State machine:** After a buy, the next action must be a sell; after a sell, the next must be a buy.

In short: it is a volatility-aware trend strategy that buys deep pullbacks and sells strong rallies, using
7/21/63-day structure to scale how aggressively it trades each move.
""")
