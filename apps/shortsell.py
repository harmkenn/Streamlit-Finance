import re
from datetime import datetime, time
from zoneinfo import ZoneInfo

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

PREMARKET_URL = "https://stockanalysis.com/markets/premarket/"
TOP_GAINERS_URL = "https://stockanalysis.com/markets/gainers/"


# ---------------------------------------------------------
# MARKET SOURCE RESOLVER (supports manual toggle override)
# ---------------------------------------------------------
def resolve_market_source_url(
    now: datetime | None = None,
    force_premarket: bool | None = None
) -> str:

    # Manual override from UI toggle
    if force_premarket is not None:
        return PREMARKET_URL if force_premarket else TOP_GAINERS_URL

    # Automatic time-based logic
    current_time = now or datetime.now(ZoneInfo("America/New_York"))
    is_weekday = current_time.weekday() < 5
    market_open = time(9, 30)
    premarket_start = time(4, 0)

    if is_weekday and premarket_start <= current_time.timetz() < market_open:
        return PREMARKET_URL
    return TOP_GAINERS_URL


# ---------------------------------------------------------
# TOP GAINERS SCRAPER + PRICE FILTER
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def get_top_gainers(limit: int = 10, force_premarket: bool | None = None) -> list[str]:
    target_url = resolve_market_source_url(force_premarket=force_premarket)

    try:
        response = requests.get(
            target_url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        page_text = response.text

        matches = re.findall(
            r"\|\s*(?:\d+)\s*\|\s*([A-Z]{1,6})\s*\|",
            page_text,
            flags=re.IGNORECASE,
        )

        symbols = []
        seen = set()
        for symbol in matches:
            symbol = symbol.strip().upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)

        # --- PRICE FILTER: remove stocks under $0.90 ---
        filtered = []
        for sym in symbols:
            try:
                price = yf.Ticker(sym).history(period="1d")["Close"].iloc[-1]
                if price >= 0.90:
                    filtered.append(sym)
            except Exception:
                continue

        if len(filtered) >= limit:
            return filtered[:limit]
        return filtered

    except Exception:
        pass

    # Fallback list (you may also filter these if desired)
    return ["RFAI", "HOWL", "USDE", "KNRX", "SDOT", "EXYN", "LSTA", "AMCI", "NCTY", "CANG"]


# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Single-Day Parabolic Short Inspector", layout="wide")
st.title("📉 Single-Ticker Parabolic Short Inspector")
st.markdown("Optimized for **Single-Day Micro-Cap Spikes (100%+ Gains)** shorted **ABOVE VWAP** returning to baseline.")


# ---------------------------------------------------------
# STOCK DATA FETCHER
# ---------------------------------------------------------
def fetch_stock_data(symbol: str) -> dict:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        hist_1d = ticker.history(period="5d", interval="1m", prepost=True)
        if hist_1d.empty:
            return None

        latest_date = hist_1d.index.max().date()
        session_df = hist_1d[hist_1d.index.date == latest_date]
        if session_df.empty:
            session_df = hist_1d

        curr_price = session_df['Close'].iloc[-1]
        day_high = session_df['High'].max()
        day_low = session_df['Low'].min()

        prev_close = info.get('previousClose', None)
        if not prev_close or np.isnan(prev_close):
            prev_close = session_df['Open'].iloc[0]

        tp = (session_df['High'] + session_df['Low'] + session_df['Close']) / 3
        vwap = (tp * session_df['Volume']).sum() / session_df['Volume'].sum() if session_df['Volume'].sum() > 0 else curr_price

        hist_10d = ticker.history(period="10d", interval="5m", prepost=True)

        gain_24h = (curr_price - prev_close) / prev_close if prev_close > 0 else 0
        drop_from_high = (day_high - curr_price) / day_high if day_high > 0 else 0

        denom = (day_high - prev_close)
        rejection_pct = (day_high - curr_price) / denom if denom > 0 else 0

        tot_vol = session_df['Volume'].sum()
        avg_vol = info.get('averageVolume10days', tot_vol)
        rvol = tot_vol / (avg_vol / 6.5) if avg_vol > 0 else 1.0

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
            "hist_10d": hist_10d,
            "trade_date": latest_date
        }
    except Exception:
        return None


# ---------------------------------------------------------
# BORROW RISK ESTIMATOR
# ---------------------------------------------------------
def estimate_borrow_status(float_shares: int, short_pct_float: float, gain_24h: float) -> tuple[str, str]:
    if float_shares > 0 and float_shares < 10_000_000 and gain_24h > 0.40:
        return "🟠 Hard to Borrow Likely", "Micro-Cap Float (<10M float) with 100%+ intraday spike."
    elif short_pct_float > 0.20:
        return "🟠 Hard to Borrow Likely", "High Short Interest (>20% of float)."
    elif float_shares >= 50_000_000:
        return "🟢 Easy to Borrow Likely", "Large Float (>50M shares)."
    else:
        return "🟡 Check E*TRADE", "Medium Float. Verify available inventory on Power E*TRADE."


# ---------------------------------------------------------
# SHORT SCORING ENGINE
# ---------------------------------------------------------
def calculate_short_score(data: dict) -> dict:
    boosters = []
    penalties = []
    score = 0

    if data['gain_24h'] >= 1.50:
        score += 35
        boosters.append(("+35 pts", "Extreme Single-Day Surge (≥150% gain)"))
    elif data['gain_24h'] >= 1.00:
        score += 25
        boosters.append(("+25 pts", "Target Intraday Surge (≥100% gain)"))
    elif data['gain_24h'] >= 0.50:
        score += 10
        boosters.append(("+10 pts", "Moderate Surge (50%-99% gain)"))
    else:
        score -= 20
        penalties.append(("-20 pts", "Weak Gain (<50% gain) — Lacks parabolic extension"))

    if data['rvol'] >= 15:
        score += 20
        boosters.append(("+20 pts", "Massive Relative Volume (RVOL ≥15x)"))
    elif data['rvol'] >= 5:
        score += 10
        boosters.append(("+10 pts", "Elevated Relative Volume (RVOL ≥5x)"))

    vwap_diff = ((data['price'] - data['vwap']) / data['vwap']) * 100
    if data['price'] > data['vwap']:
        if vwap_diff >= 15:
            score += 25
            boosters.append(("+25 pts", f"Stretched Above VWAP ({vwap_diff:+.1f}%) — Optimal fade premium!"))
        else:
            score += 15
            boosters.append(("+15 pts", f"Trading Above VWAP ({vwap_diff:+.1f}%)"))
    else:
        score -= 15
        penalties.append(("-15 pts", f"Trading Below VWAP ({vwap_diff:+.1f}%) — Move already broke down"))

    if 0 < data['float_shares'] <= 10_000_000:
        score += 10
        boosters.append(("+10 pts", f"Low Float Micro-Cap ({data['float_shares']/1e6:.1f}M shares) — Classic pump candidate"))

    if data['drop_from_high'] <= 0.05:
        score -= 30
        penalties.append(("-30 pts", "Hugging Day Highs (Drop ≤5%) — Extreme continuation/squeeze risk!"))

    final_score = max(0, min(100, score))

    if final_score >= 75 and data['price'] > data['vwap']:
        status = "🔴 TRIGGER"
        status_msg = "Stock is up >100% and extended ABOVE VWAP. Ideal single-day fade location."
    elif final_score >= 55:
        status = "🟡 ARMED"
        status_msg = "Parabolic extension active. Monitor price action above VWAP for top exhaustion."
    else:
        status = "⚪ CANDIDATE"
        status_msg = "Lacks 100%+ extension or is already dumped below VWAP."

    return {
        "final_score": final_score,
        "status": status,
        "status_msg": status_msg,
        "boosters": boosters,
        "penalties": penalties,
        "vwap_diff": vwap_diff
    }


# ---------------------------------------------------------
# UI — MARKET MODE TOGGLE
# ---------------------------------------------------------
market_mode = st.toggle("Use Premarket List (4:00–9:30 a.m. ET)", value=True)

refresh_button = st.button("Refresh Top Gainers", type="secondary")
if refresh_button:
    st.cache_data.clear()

with st.spinner("Loading top gainers..."):
    top_tickers = get_top_gainers(force_premarket=market_mode)


# ---------------------------------------------------------
# UI — TICKER SELECTION
# ---------------------------------------------------------
col_input, col_btn = st.columns([3, 1])
with col_input:
    ticker_input = st.selectbox("Select Top Gainer:", options=top_tickers, index=0)
with col_btn:
    st.write(" ")
    analyze_click = st.button("Analyze Stock", width="stretch")


# ---------------------------------------------------------
# ANALYSIS ENGINE
# ---------------------------------------------------------
if ticker_input or analyze_click:
    with st.spinner(f"Fetching market data for {ticker_input}..."):
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

        st.info(f"**State Summary:** {eval_res['status_msg']} *(Session Date: {data['trade_date']})*")

        # 10-DAY CHART
        st.subheader("📈 10-Day Intraday Baseline Chart (Pre/Post Market Included)")
        hist_df = data['hist_10d']
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hist_df.index,
            y=hist_df['Close'],
            mode='lines',
            name='Price (5m Extended)',
            line=dict(color='#00B4D8', width=1.5)
        ))

        fig.add_hline(
            y=data['prev_close'],
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Prev Close (${data['prev_close']:.2f})"
        )

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
            xaxis=dict(type="category", nticks=10)
        )

        st.plotly_chart(fig, width="stretch")

        # METRIC BREAKDOWN
        st.subheader("📊 Intraday Metric Breakdown")
        d1, d2, d3, d4 = st.columns(4)
        d1.write(f"**Day High:** ${data['day_high']:.2f}")
        d1.write(f"**Previous Close:** ${data['prev_close']:.2f}")

        d2.write(f"**Drop From High:** {data['drop_from_high']*100:.1f}%")
        d2.write(f"**Spike Rejection:** {data['rejection_pct']*100:.1f}%")

        vwap_relation = "ABOVE 🟢" if data['price'] > data['vwap'] else "BELOW 🔴"
        d3.write(f"**Today VWAP:** ${data['vwap']:.2f} ({vwap_relation} {eval_res['vwap_diff']:+.1f}%)")
        d3.write(f"**RVOL Proxy:** {data['rvol']:.1f}x")

        float_str = f"{data['float_shares']/1e6:.1f}M" if data['float_shares'] else "N/A"
        d4.write(f"**Float Size:** {float_str}")
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
                st.write("No positive setup points triggered.")

        with c2:
            st.markdown("### 🔴 Warning Penalties (Continuation / Below VWAP)")
            if eval_res['penalties']:
                for pts, desc in eval_res['penalties']:
                    st.write(f"• **{pts}**: {desc}")
            else:
                st.write("No active warning penalties.")
