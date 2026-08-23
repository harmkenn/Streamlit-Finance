import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Short Candidate Inspector", layout="wide")
st.title("📉 Single-Ticker Parabolic Short Inspector")
st.markdown("Analyze short exhaustion signals alongside a 10-day extended-hours intraday price chart.")

# --- FREE DATA ENGINE (YFINANCE) ---
def fetch_stock_data(symbol: str) -> dict:
    """Fetches intraday bars (including pre/post market), 10-day history, and short float data."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # 1. Today's 1-minute data for VWAP & Day High calculations (including pre-market)
        hist_1d = ticker.history(period="1d", interval="1m", prepost=True)
        if hist_1d.empty:
            return None

        curr_price = hist_1d['Close'].iloc[-1]
        day_high = hist_1d['High'].max()
        day_low = hist_1d['Low'].min()
        prev_close = info.get('previousClose', hist_1d['Open'].iloc[0])
        
        # Calculate Today's Anchored Intraday VWAP
        tp = (hist_1d['High'] + hist_1d['Low'] + hist_1d['Close']) / 3
        vwap = (tp * hist_1d['Volume']).sum() / hist_1d['Volume'].sum() if hist_1d['Volume'].sum() > 0 else curr_price
        
        # 2. 10-Day Intraday history (5m bars) WITH Extended Hours enabled
        hist_10d = ticker.history(period="10d", interval="5m", prepost=True)
        
        # Ratios & Metrics
        gain_24h = (curr_price - prev_close) / prev_close if prev_close > 0 else 0
        drop_from_high = (day_high - curr_price) / day_high if day_high > 0 else 0
        
        denom = (day_high - prev_close)
        rejection_pct = (day_high - curr_price) / denom if denom > 0 else 0
        
        # RVOL Proxy
        tot_vol = hist_1d['Volume'].sum()
        avg_vol = info.get('averageVolume10days', tot_vol)
        rvol = tot_vol / (avg_vol / 6.5) if avg_vol > 0 else 1.0
        
        # Float Metrics
        float_shares = info.get('floatShares', 0) or 0
        short_percent_of_float = info.get('shortPercentOfFloat', 0) or 0
        
        return {
            "symbol": symbol,
            "price": curr_price,
            "day_high": day_high,
            "day_low": day_low,
            "prev_close": prev_close,
            "gain_24h": gain_24h,
            "drop_from_high": drop_from_high,
            "rejection_pct": rejection_pct,
            "vwap": vwap,
            "rvol": rvol,
            "float_shares": float_shares,
            "short_pct_float": short_percent_of_float,
            "hist_10d": hist_10d
        }
    except Exception:
        return None

# --- BORROW RISK ESTIMATOR ---
def estimate_borrow_status(float_shares: int, short_pct_float: float, gain_24h: float) -> tuple[str, str]:
    if float_shares > 0 and float_shares < 10_000_000 and gain_24h > 0.40:
        return "🟠 Hard to Borrow Likely", "Low Float Micro-cap (<10M float) with strong intraday pump."
    elif short_pct_float > 0.20:
        return "🟠 Hard to Borrow Likely", "High Short Interest (>20% of float)."
    elif float_shares >= 50_000_000:
        return "🟢 Easy to Borrow Likely", "Large Float (>50M shares)."
    else:
        return "🟡 Check E*TRADE", "Medium Float. Verify available shares on Power E*TRADE."

# --- SHORT SCORING ENGINE ---
def calculate_short_score(data: dict) -> dict:
    boosters = []
    penalties = []
    score = 0

    if data['gain_24h'] >= 1.0:
        score += 25
        boosters.append(("+25 pts", "Massive 24h Gain (≥100%)"))
    elif data['gain_24h'] >= 0.50:
        score += 15
        boosters.append(("+15 pts", "Strong 24h Gain (≥50%)"))

    if data['rvol'] >= 10:
        score += 15
        boosters.append(("+15 pts", "Extremely High Volume Expansion (RVOL ≥10x)"))
    elif data['rvol'] >= 3:
        score += 10
        boosters.append(("+10 pts", "Elevated Volume (RVOL ≥3x)"))

    if data['drop_from_high'] >= 0.35:
        score += 20
        boosters.append(("+20 pts", "Deep Pullback from Highs (≥35%)"))
    elif data['drop_from_high'] >= 0.15:
        score += 10
        boosters.append(("+10 pts", "Moderate Pullback from Highs (≥15%)"))

    if data['rejection_pct'] >= 0.65:
        score += 20
        boosters.append(("+20 pts", "Heavy Spike Rejection (≥65% surrendered)"))

    if data['price'] < data['vwap']:
        score += 10
        boosters.append(("+10 pts", "Trading Below Intraday Anchored VWAP"))

    if data['drop_from_high'] <= 0.08:
        score -= 35
        penalties.append(("-35 pts", "Hugging Day Highs (Drop ≤8%) — Squeeze risk!"))
        
    if data['price'] > data['vwap']:
        score -= 20
        penalties.append(("-20 pts", "Holding Above VWAP — Buyers still in control."))

    final_score = max(0, min(100, score))

    if final_score >= 80 and data['price'] < data['vwap']:
        status = "🔴 TRIGGER"
        status_msg = "Multiple failure conditions aligned. Strong exhaustion signal."
    elif final_score >= 60:
        status = "🟡 ARMED"
        status_msg = "Rejection in progress. Monitor closely for VWAP loss."
    else:
        status = "⚪ CANDIDATE"
        status_msg = "Weak exhaustion or strong continuation momentum. Avoid shorting."

    return {
        "final_score": final_score,
        "status": status,
        "status_msg": status_msg,
        "boosters": boosters,
        "penalties": penalties
    }

# --- USER INPUT ---
col_input, col_btn = st.columns([3, 1])
with col_input:
    ticker_input = st.text_input("Enter Stock Ticker:", "SDOT").strip().upper()
with col_btn:
    st.write(" ")
    analyze_click = st.button("Analyze Stock", use_container_width=True)

if ticker_input or analyze_click:
    with st.spinner(f"Loading 10-day market & extended-hours data for {ticker_input}..."):
        data = fetch_stock_data(ticker_input)

    if not data:
        st.error(f"Could not retrieve data for **{ticker_input}**. Verify symbol or check if market is open.")
    else:
        eval_res = calculate_short_score(data)
        borrow_status, borrow_reason = estimate_borrow_status(
            data['float_shares'], data['short_pct_float'], data['gain_24h']
        )
        
        st.divider()

        # METRIC CARDS
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Short Score", f"{eval_res['final_score']}/100")
        m2.metric("Signal State", eval_res['status'])
        m3.metric("Current Price", f"${data['price']:.2f}", f"{data['gain_24h']*100:+.1f}%")
        m4.metric("Borrow Outlook", borrow_status)

        st.info(f"**State Summary:** {eval_res['status_msg']}")

        # --- 10-DAY EXTENDED HOURS PLOTLY CHART ---
        st.subheader("📈 10-Day Intraday Chart (Includes Pre-Market & After-Hours)")
        
        hist_df = data['hist_10d']
        fig = go.Figure()

        # Intraday Price Line
        fig.add_trace(go.Scatter(
            x=hist_df.index,
            y=hist_df['Close'],
            mode='lines',
            name='Price (5m Extended)',
            line=dict(color='#00B4D8', width=1.5)
        ))

        # Baseline: Previous Close
        fig.add_hline(
            y=data['prev_close'], 
            line_dash="dash", 
            line_color="gray", 
            annotation_text=f"Prev Close (${data['prev_close']:.2f})"
        )

        # Baseline: Today's VWAP
        fig.add_hline(
            y=data['vwap'], 
            line_dash="dot", 
            line_color="orange", 
            annotation_text=f"Today's VWAP (${data['vwap']:.2f})"
        )

        fig.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="Date/Time",
            yaxis_title="Stock Price ($)",
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified",
            # Hides weekend gaps so the line chart connects smoothly across trading sessions
            xaxis=dict(
                type="category",
                nticks=10
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # TECHNICAL BREAKDOWN
        st.subheader("📊 Intraday Metric Breakdown")
        d1, d2, d3, d4 = st.columns(4)
        d1.write(f"**Day High (Inc. Pre):** ${data['day_high']:.2f}")
        d1.write(f"**Previous Close:** ${data['prev_close']:.2f}")

        d2.write(f"**Drop From High:** {data['drop_from_high']*100:.1f}%")
        d2.write(f"**Spike Rejection:** {data['rejection_pct']*100:.1f}%")

        vwap_relation = "BELOW" if data['price'] < data['vwap'] else "ABOVE"
        d3.write(f"**Today VWAP:** ${data['vwap']:.2f} ({vwap_relation})")
        d3.write(f"**RVOL Proxy:** {data['rvol']:.1f}x")

        float_str = f"{data['float_shares']/1e6:.1f}M" if data['float_shares'] else "N/A"
        d4.write(f"**Estimated Float:** {float_str}")
        d4.write(f"**Borrow Note:** {borrow_reason}")

        st.divider()

        # SCORE BREAKDOWN
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🟢 Positive Exhaustion Points")
            if eval_res['boosters']:
                for pts, desc in eval_res['boosters']:
                    st.write(f"• **{pts}**: {desc}")
            else:
                st.write("No exhaustion boosters triggered.")

        with c2:
            st.markdown("### 🔴 Warning Penalties (Continuation Risk)")
            if eval_res['penalties']:
                for pts, desc in eval_res['penalties']:
                    st.write(f"• **{pts}**: {desc}")
            else:
                st.write("No continuation penalties active.")