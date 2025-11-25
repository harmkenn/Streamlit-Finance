import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

st.title("TQQQ Trigger Strategy — Parameter Heatmap Explorer")

st.write("""
This tool tests how different **down % triggers** and **up % triggers**
would have performed over the past 3 years of TQQQ data.

It simulates a simple rule:

- Start with $100,000  
- When price drops **X%** from the previous day's close → buy $10,000  
- When price rises **Y%** from previous close → sell $10,000  
- Multiple events allowed per day  
""")

# ===============================================================
# LOAD AND CLEAN TQQQ DATA
# ===============================================================
df = yf.download("TQQQ", period="3y", interval="1d")

if df.empty:
    st.error("Error: Yahoo Finance returned no data.")
    st.stop()

# FIX: flatten multi-index columns if present
df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

# FIX: ensure Close is numeric and clean
close = pd.to_numeric(df["Close"], errors="coerce").dropna()

# Align df to cleaned close
df = df.loc[close.index]
df["Close"] = close.astype(float)

# ===============================================================
# STRATEGY FUNCTION
# ===============================================================
def run_strategy(close_series, down_pct, up_pct):
    cash = 100000.0
    shares = 0.0
    trade_amount = 10000.0

    prev_close = None

    for date, price in close_series.items():
        price = float(price)

        if prev_close is None:
            prev_close = price
            continue

        buy_trigger = prev_close * (1 - down_pct)
        sell_trigger = prev_close * (1 + up_pct)

        # BUY event (if intraday low < trigger)
        if price <= buy_trigger:
            qty = trade_amount / buy_trigger
            cash -= trade_amount
            shares += qty

        # SELL event
        if price >= sell_trigger:
            qty = trade_amount / sell_trigger
            if shares >= qty:
                cash += trade_amount
                shares -= qty

        prev_close = price

    final_value = cash + shares * price
    return final_value


# ===============================================================
# USER CONTROLS
# ===============================================================
st.subheader("Parameter Sweep")

down_values = st.slider(
    "Down-trigger % range (drop from previous close)",
    1, 20, (3, 10)  # default 3% to 10%
)

up_values = st.slider(
    "Up-trigger % range (rise from previous close)",
    1, 20, (3, 10)
)

down_range = np.linspace(down_values[0]/100, down_values[1]/100, 20)
up_range = np.linspace(up_values[0]/100, up_values[1]/100, 20)

# ===============================================================
# HEATMAP COMPUTATION
# ===============================================================
st.write("Running simulations...")

results = pd.DataFrame(index=down_range, columns=up_range)

for d in down_range:
    for u in up_range:
        results.loc[d, u] = run_strategy(close, d, u)

results = results.astype(float)

# Find best combination
best_down = results.max().idxmax()
best_up = results.idxmax().max()
best_value = results.max().max()

# ===============================================================
# SHOW RESULTS
# ===============================================================
st.subheader("Heatmap — Final Portfolio Value")

fig = px.imshow(
    results,
    labels=dict(x="Up trigger %", y="Down trigger %", color="Final Value ($)"),
    x=[f"{round(v*100,2)}%" for v in results.columns],
    y=[f"{round(v*100,2)}%" for v in results.index],
    color_continuous_scale="Turbo",
)

st.plotly_chart(fig)

st.subheader("Best Parameter Pair Found")
st.write(f"""
- **Down trigger:** {best_up*100:.2f}%  
- **Up trigger:** {best_down*100:.2f}%  
- **Final Portfolio Value:** ${best_value:,.2f}
""")

# ---------------------------------------------------------------
# Show raw numeric table
# ---------------------------------------------------------------
if st.checkbox("Show raw results table"):
    st.dataframe(results.style.format("{:,.2f}"))
