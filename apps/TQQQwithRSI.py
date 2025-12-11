import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.title("TQQQ 5-Year Trigger Optimizer (OHLC + RSI) v3.3")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Backtest Settings")

initial_cash = st.sidebar.number_input("Initial Cash", 10000, 500000, 100000, 10000)
years = st.sidebar.slider("Lookback (years)", 1, 10, 5)

# Grid of thresholds (as percentage moves from previous close)
buy_drop_min = st.sidebar.slider("Min buy drop (%)", 1.0, 20.0, 3.0, 0.5)
buy_drop_max = st.sidebar.slider("Max buy drop (%)", 2.0, 30.0, 6.5, 0.5)
buy_drop_step = st.sidebar.slider("Buy drop step (%)", 0.25, 5.0, 1.0, 0.25)

sell_rise_min = st.sidebar.slider("Min sell rise (%)", 1.0, 20.0, 3.0, 0.5)
sell_rise_max = st.sidebar.slider("Max sell rise (%)", 2.0, 30.0, 6.5, 0.5)
sell_rise_step = st.sidebar.slider("Sell rise step (%)", 0.25, 5.0, 1.0, 0.25)

use_rsi_filter = st.sidebar.checkbox("Use RSI filters", value=True)
rsi_buy_max = st.sidebar.slider("Max RSI for buys", 10, 90, 60)
rsi_sell_min = st.sidebar.slider("Min RSI for sells", 10, 90, 40)

trade_amount = st.sidebar.number_input("Trade size per signal ($)", 1000, 50000, 10000, 1000)

st.write(
    f"Backtesting on **TQQQ** with {years} years of daily data to "
    f"find the best buy/sell triggers, then applying them to today's price."
)

# -----------------------------
# Data loading
# -----------------------------
ticker = "TQQQ"
df = yf.download(ticker, period=f"{years}y", interval="1d")

if df.empty:
    st.error("No data returned for TQQQ.")
    st.stop()

# Flatten MultiIndex if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]

df = df[["Open", "High", "Low", "Close"]].astype(float)
df["PrevClose"] = df["Close"].shift(1)

# RSI calculation
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["RSI"] = compute_rsi(df["Close"])
df = df.dropna()

# -----------------------------
# Backtest function
# -----------------------------
def backtest_triggers(data, buy_drop, sell_rise, use_rsi=True,
                      rsi_buy_max=60, rsi_sell_min=40,
                      initial_cash=100000, trade_amount=10000):
    cash = float(initial_cash)
    shares = 0.0

    # convert percentage figures to decimals
    b = buy_drop / 100.0
    s = sell_rise / 100.0

    for _, row in data.iterrows():
        prev_close = float(row["PrevClose"])
        low = float(row["Low"])
        high = float(row["High"])
        close = float(row["Close"])
        prev_close = float(row["PrevClose"])

        # Skip if we don't have a valid previous close
        if np.isnan(prev_close):
            continue

        buy_trigger = prev_close * (1 - b)
        sell_trigger = prev_close * (1 + s)

        # BUY condition
        buy_cond = low <= buy_trigger
        if use_:
            buy_cond = buy_cond and ( <= _buy_max)

        if buy_cond and cash >= trade_amount:
            qty = trade_amount / buy_trigger
            cash -= trade_amount
            shares += qty

        # SELL condition
        sell_cond = high >= sell_trigger
        if use_:
            sell_cond = sell_cond and ( >= _sell_min)

        if sell_cond and shares > 0:
            # sell up to trade_amount if possible
            sell_value = trade_amount
            max_sell_value = shares * sell_trigger
            if sell_value > max_sell_value:
                sell_value = max_sell_value

            qty = sell_value / sell_trigger
            shares -= qty
            cash += sell_value

    # Final portfolio value at last close
    final_price = float(data["Close"].iloc[-1])
    final_value = cash + shares * final_price
    return final_value

# -----------------------------
# Grid search over thresholds
# -----------------------------
buy_drops = np.arange(buy_drop_min, buy_drop_max + 1e-9, buy_drop_step)
sell_rises = np.arange(sell_rise_min, sell_rise_max + 1e-9, sell_rise_step)

results = []
best_value = -np.inf
best_params = None

progress = st.progress(0.0)
total_iters = len(buy_drops) * len(sell_rises)
iter_count = 0

for b in buy_drops:
    for s in sell_rises:
        iter_count += 1
        progress.progress(iter_count / total_iters)

        final_val = backtest_triggers(
            df,
            buy_drop=b,
            sell_rise=s,
            use_=use__filter,
            _buy_max=_buy_max,
            _sell_min=_sell_min,
            initial_cash=initial_cash,
            trade_amount=trade_amount
        )
        results.append((b, s, final_val))
        if final_val > best_value:
            best_value = final_val
            best_params = (b, s)

progress.empty()

results_df = pd.DataFrame(results, columns=["BuyDropPct", "SellRisePct", "FinalValue"])
results_df = results_df.sort_values("FinalValue", ascending=False)

st.subheader("Best Historical Trigger Pair (5-Year Backtest)")
if best_params is None:
    st.write("No valid results from backtest.")
    st.stop()

best_buy, best_sell = best_params
st.write(f"**Best Buy Drop:** {best_buy:.2f}% below previous close")
st.write(f"**Best Sell Rise:** {best_sell:.2f}% above previous close")
st.write(f"**Best Final Portfolio Value:** ${best_value:,.2f}")

st.subheader("Top 15 Trigger Combinations")
st.dataframe(results_df.head(15))

# -----------------------------
# Suggest today's target prices
# -----------------------------
latest_row = df.iloc[-1]
latest_close = float(latest_row["Close"])
latest_ = float(latest_row[""])

today_buy_price = latest_close * (1 - best_buy / 100.0)
today_sell_price = latest_close * (1 + best_sell / 100.0)

st.subheader("Today's Suggested Targets (Based on Best Historical Triggers)")

st.write(f"**Latest Close:** ${latest_close:,.2f}")
st.write(f"**Latest (14):** {latest_rsi:.1f}")

st.write(f"**Suggested Buy Trigger:** "
         f"${today_buy_price:,.2f} ({best_buy:.2f}% below last close)")
st.write(f"**Suggested Sell Trigger:** "
         f"${today_sell_price:,.2f} ({best_sell:.2f}% above last close)")

st.info(
    "These levels are derived purely from 5-year historical backtest of simple triggers "
    "and are NOT financial advice."
)

# Optional: visualize price + RSI
st.subheader("TQQQ Close and RSI (Last 5 Years)")
viz = df[["Close", "RSI"]].copy()
st.line_chart(viz)
