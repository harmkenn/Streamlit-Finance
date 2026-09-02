import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
import yfinance as yf

st.title("📉 Single-Ticker Parabolic Short Inspector")
st.markdown(
    "Optimized for **Single-Day Micro-Cap Spikes (100%+ Gains)** shorted **ABOVE VWAP** returning to baseline."
)

# --- READ SHARED SESSION STATE TICKERS ---
raw_tickers = st.session_state.get("user_tickers", "")
if not raw_tickers.strip():
    raw_tickers = st.session_state.get("tickers", "")

configured_tickers = [
    t.strip().upper() for t in raw_tickers.replace("\n", ",").split(",") if t.strip()
]

if not configured_tickers:
    configured_tickers = ["TQQQ", "UPRO", "UDOW", "^VIX", "SPHY"]


# --- ADVANCED SHORT METRICS COMPUTATION ENGINE ---
def compute_short_metrics(ticker_obj, info: dict) -> dict:
    """Computes Altman Z-Score, Piotroski F-Score, Cash Runway/Burn, CTB, and Share Growth YoY."""
    metrics = {
        "z_score": None,
        "f_score": None,
        "cash_runway_months": None,
        "cash_burn_monthly": None,
        "ctb_estimated": None,
        "share_growth_yoy": None,
    }

    try:
        # Financial Statements Extraction
        bs = getattr(ticker_obj, "quarterly_balance_sheet", None)
        inc = getattr(ticker_obj, "quarterly_financials", None)
        cf = getattr(ticker_obj, "quarterly_cashflow", None)

        if bs is None or bs.empty or inc is None or inc.empty:
            bs = getattr(ticker_obj, "balance_sheet", None)
            inc = getattr(ticker_obj, "financials", None)
            cf = getattr(ticker_obj, "cashflow", None)

        if (
            bs is not None
            and not bs.empty
            and inc is not None
            and not inc.empty
            and len(bs.columns) >= 1
        ):
            tot_assets = bs.iloc[:, 0].get("Total Assets", np.nan)
            tot_liab = bs.iloc[:, 0].get(
                "Total Liabilities Net Minority Interest", np.nan
            )
            curr_assets = bs.iloc[:, 0].get("Current Assets", np.nan)
            curr_liab = bs.iloc[:, 0].get("Current Liabilities", np.nan)
            cash = bs.iloc[:, 0].get("Cash And Cash Equivalents", np.nan)
            re = bs.iloc[:, 0].get("Retained Earnings", np.nan)

            ebit = inc.iloc[:, 0].get("EBIT", np.nan)
            rev = inc.iloc[:, 0].get("Total Revenue", np.nan)
            net_income = inc.iloc[:, 0].get("Net Income", np.nan)
            mcap = info.get("marketCap", np.nan)

            # --- ALTMAN Z-SCORE ---
            if all(
                pd.notna(x) and x > 0
                for x in [tot_assets, tot_liab, curr_assets, curr_liab, rev]
            ):
                x1 = (curr_assets - curr_liab) / tot_assets
                x2 = (re if pd.notna(re) else 0.0) / tot_assets
                x3 = (ebit if pd.notna(ebit) else 0.0) / tot_assets
                x4 = (mcap if pd.notna(mcap) else 0.0) / tot_liab
                x5 = rev / tot_assets

                metrics["z_score"] = (
                    1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5
                )

            # --- CASH BURN & RUNWAY (MONTHS) ---
            if cf is not None and not cf.empty and pd.notna(cash):
                fcf_q = cf.iloc[:, 0].get("Free Cash Flow", np.nan)
                if pd.isna(fcf_q):
                    ocf_q = cf.iloc[:, 0].get("Operating Cash Flow", np.nan)
                    capex_q = cf.iloc[:, 0].get("Capital Expenditure", np.nan)
                    if pd.notna(ocf_q):
                        fcf_q = ocf_q + (capex_q if pd.notna(capex_q) else 0)

                if pd.notna(fcf_q) and fcf_q < 0:
                    monthly_burn = abs(fcf_q) / 3.0
                    metrics["cash_burn_monthly"] = monthly_burn
                    if monthly_burn > 0:
                        metrics["cash_runway_months"] = cash / monthly_burn
                elif pd.notna(fcf_q) and fcf_q >= 0:
                    metrics["cash_burn_monthly"] = 0.0
                    metrics["cash_runway_months"] = 999.0

            # --- PIOTROSKI F-SCORE ---
            if len(bs.columns) >= 2 and len(inc.columns) >= 2:
                f_score = 0
                roa_curr = (
                    (net_income / tot_assets)
                    if pd.notna(net_income) and pd.notna(tot_assets)
                    else 0
                )
                if roa_curr > 0:
                    f_score += 1

                if cf is not None and not cf.empty:
                    ocf_curr = cf.iloc[:, 0].get("Operating Cash Flow", np.nan)
                    if pd.notna(ocf_curr) and ocf_curr > 0:
                        f_score += 1
                    if (
                        pd.notna(ocf_curr)
                        and pd.notna(net_income)
                        and ocf_curr > net_income
                    ):
                        f_score += 1

                tot_assets_prev = bs.iloc[:, 1].get("Total Assets", np.nan)
                net_income_prev = inc.iloc[:, 1].get("Net Income", np.nan)
                if pd.notna(tot_assets_prev) and pd.notna(net_income_prev):
                    roa_prev = net_income_prev / tot_assets_prev
                    if roa_curr > roa_prev:
                        f_score += 1

                ltd_curr = bs.iloc[:, 0].get("Long Term Debt", 0.0) or 0.0
                ltd_prev = bs.iloc[:, 1].get("Long Term Debt", 0.0) or 0.0
                if (ltd_curr / tot_assets) < (ltd_prev / tot_assets_prev):
                    f_score += 1

                cr_curr = (
                    (curr_assets / curr_liab)
                    if pd.notna(curr_assets) and pd.notna(curr_liab)
                    else 0
                )
                curr_assets_p = bs.iloc[:, 1].get("Current Assets", np.nan)
                curr_liab_p = bs.iloc[:, 1].get("Current Liabilities", np.nan)
                if pd.notna(curr_assets_p) and pd.notna(curr_liab_p):
                    cr_prev = curr_assets_p / curr_liab_p
                    if cr_curr > cr_prev:
                        f_score += 1

                metrics["f_score"] = f_score

        # --- SHARES DILUTION ---
        shares_curr = info.get("sharesOutstanding", None)
        shares_prev = (
            bs.iloc[:, -1].get("Share Issued", None)
            if bs is not None and not bs.empty
            else None
        )
        if shares_curr and shares_prev and shares_prev > 0:
            metrics["share_growth_yoy"] = (
                (shares_curr - shares_prev) / shares_prev
            ) * 100.0

        # --- CTB ESTIMATION ---
        float_shares = info.get("floatShares", 0) or 0
        short_ratio = info.get("shortPercentOfFloat", 0) or 0
        if 0 < float_shares < 5_000_000:
            metrics["ctb_estimated"] = 85.0
        elif float_shares < 15_000_000 and short_ratio > 0.15:
            metrics["ctb_estimated"] = 45.0
        elif float_shares > 50_000_000:
            metrics["ctb_estimated"] = 1.5
        else:
            metrics["ctb_estimated"] = 12.0

    except Exception:
        pass

    return metrics


# --- CACHED DATA ENGINE TO PREVENT RATE LIMITING ---
@st.cache_data(ttl=120, show_spinner=False)
def fetch_stock_data(symbol: str) -> dict:
    try:
        ticker = yf.Ticker(symbol)
<<<<<<< HEAD

        # 1. Fetch up to 5 days of 1-minute bars with extended hours enabled
=======
        info = ticker.info or {}

>>>>>>> 93e7e584595dc03583f78411101b6f7d4ddf81e8
        hist_1d = ticker.history(period="5d", interval="1m", prepost=True)
        if hist_1d.empty:
            return None

<<<<<<< HEAD
        # Fundamentals are optional; Yahoo can provide prices even when info fails.
        try:
            info = ticker.info or {}
        except Exception:
            info = {}

        # Filter to the most recent trading session
=======
>>>>>>> 93e7e584595dc03583f78411101b6f7d4ddf81e8
        latest_date = hist_1d.index.max().date()
        session_df = hist_1d[hist_1d.index.date == latest_date]
        if session_df.empty:
            session_df = hist_1d

<<<<<<< HEAD
        curr_price = session_df['Close'].iloc[-1]
        day_high = session_df['High'].max()
        day_low = session_df['Low'].min()
        
        # Fallback for Previous Close
        prev_close = pd.to_numeric(info.get('previousClose'), errors="coerce")
        if pd.isna(prev_close) or prev_close <= 0:
            prev_close = session_df['Open'].iloc[0]
        
        # Today's Anchored Intraday VWAP
        tp = (session_df['High'] + session_df['Low'] + session_df['Close']) / 3
        vwap = (tp * session_df['Volume']).sum() / session_df['Volume'].sum() if session_df['Volume'].sum() > 0 else curr_price
        
        # 2. 10-Day history (5m bars) with extended hours
        hist_10d = ticker.history(period="10d", interval="5m", prepost=True)
        if hist_10d.empty:
            hist_10d = hist_1d
        
        # Ratios & Metrics
        gain_24h = (curr_price - prev_close) / prev_close if prev_close > 0 else 0
        drop_from_high = (day_high - curr_price) / day_high if day_high > 0 else 0
        
        denom = (day_high - prev_close)
=======
        curr_price = session_df["Close"].iloc[-1]
        day_high = session_df["High"].max()
        day_low = session_df["Low"].min()

        prev_close = info.get("previousClose", None)
        if not prev_close or np.isnan(prev_close):
            prev_close = session_df["Open"].iloc[0]

        tp = (session_df["High"] + session_df["Low"] + session_df["Close"]) / 3
        vwap = (
            (tp * session_df["Volume"]).sum() / session_df["Volume"].sum()
            if session_df["Volume"].sum() > 0
            else curr_price
        )

        hist_10d = ticker.history(period="10d", interval="5m", prepost=True)

        gain_24h = (
            (curr_price - prev_close) / prev_close if prev_close > 0 else 0
        )
        drop_from_high = (
            (day_high - curr_price) / day_high if day_high > 0 else 0
        )

        denom = day_high - prev_close
>>>>>>> 93e7e584595dc03583f78411101b6f7d4ddf81e8
        rejection_pct = (day_high - curr_price) / denom if denom > 0 else 0

        tot_vol = session_df["Volume"].sum()
        avg_vol = info.get("averageVolume10days", tot_vol)
        rvol = tot_vol / (avg_vol / 6.5) if avg_vol > 0 else 1.0

        float_shares = info.get("floatShares", 0) or 0
        short_percent_of_float = info.get("shortPercentOfFloat", 0) or 0

        extra_metrics = compute_short_metrics(ticker, info)

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
            "trade_date": latest_date,
            **extra_metrics,
        }
    except Exception:
        return None


# --- BORROW RISK ESTIMATOR ---
def estimate_borrow_status(
    float_shares: int,
    short_pct_float: float,
    gain_24h: float,
    ctb: float = None,
) -> tuple[str, str]:
    if ctb and ctb > 50.0:
        return (
            "🔴 Extremely Hard to Borrow",
            f"Estimated CTB is high ({ctb:.1f}% APY). Squeeze/borrow fee risk.",
        )
    elif float_shares > 0 and float_shares < 10_000_000 and gain_24h > 0.40:
        return (
            "🟠 Hard to Borrow Likely",
            "Micro-Cap Float (<10M float) with 100%+ intraday spike.",
        )
    elif short_pct_float > 0.20:
        return (
            "🟠 Hard to Borrow Likely",
            "High Short Interest (>20% of float).",
        )
    elif float_shares >= 50_000_000:
        return (
            "🟢 Easy to Borrow Likely",
            "Large Float (>50M shares). Low CTB rate.",
        )
    else:
        return (
            "🟡 Check Broker Inventory",
            "Medium Float. Verify available inventory.",
        )


# --- SHORT SCORING ENGINE ---
def calculate_short_score(data: dict) -> dict:
    boosters = []
    penalties = []
    score = 0

    if data["gain_24h"] >= 1.50:
        score += 35
        boosters.append(("+35 pts", "Extreme Single-Day Surge (≥150% gain)"))
    elif data["gain_24h"] >= 1.00:
        score += 25
        boosters.append(("+25 pts", "Target Intraday Surge (≥100% gain)"))
    elif data["gain_24h"] >= 0.50:
        score += 10
        boosters.append(("+10 pts", "Moderate Surge (50%-99% gain)"))
    else:
        score -= 20
        penalties.append(
            ("-20 pts", "Weak Gain (<50% gain) — Lacks parabolic extension")
        )

    if data["rvol"] >= 15:
        score += 20
        boosters.append(("+20 pts", "Massive Relative Volume (RVOL ≥15x)"))
    elif data["rvol"] >= 5:
        score += 10
        boosters.append(("+10 pts", "Elevated Relative Volume (RVOL ≥5x)"))

    vwap_diff = ((data["price"] - data["vwap"]) / data["vwap"]) * 100
    if data["price"] > data["vwap"]:
        if vwap_diff >= 15:
            score += 25
            boosters.append(
                (
                    "+25 pts",
                    f"Stretched Above VWAP ({vwap_diff:+.1f}%) — Optimal fade premium!",
                )
            )
        else:
            score += 15
            boosters.append(
                ("+15 pts", f"Trading Above VWAP ({vwap_diff:+.1f}%)")
            )
    else:
        score -= 15
        penalties.append(
            (
                "-15 pts",
                f"Trading Below VWAP ({vwap_diff:+.1f}%) — Move already broke down",
            )
        )

    if 0 < data["float_shares"] <= 10_000_000:
        score += 10
        boosters.append(
            (
                "+10 pts",
                f"Low Float Micro-Cap ({data['float_shares']/1e6:.1f}M shares)",
            )
        )

    if data["z_score"] is not None and data["z_score"] < 1.8:
        score += 10
        boosters.append(
            (
                "+10 pts",
                f"Altman Z-Score in Distress Zone ({data['z_score']:.2f} < 1.8)",
            )
        )

    if (
        data["cash_runway_months"] is not None
        and data["cash_runway_months"] < 6.0
    ):
        score += 10
        boosters.append(
            (
                "+10 pts",
                f"Severe Cash Burn (Runway: {data['cash_runway_months']:.1f} months)",
            )
        )

    if data["f_score"] is not None and data["f_score"] <= 2:
        score += 10
        boosters.append(
            (
                "+10 pts",
                f"Deteriorating Fundamentals (Piotroski F-Score: {data['f_score']}/9)",
            )
        )

    if data["drop_from_high"] <= 0.05:
        score -= 30
        penalties.append(
            (
                "-30 pts",
                "Hugging Day Highs (Drop ≤5%) — Extreme continuation/squeeze risk!",
            )
        )

    if data["ctb_estimated"] and data["ctb_estimated"] > 50.0:
        score -= 15
        penalties.append(
            (
                "-15 pts",
                f"High Borrow Cost ({data['ctb_estimated']:.1f}% CTB)",
            )
        )

    final_score = max(0, min(100, score))

    if final_score >= 75 and data["price"] > data["vwap"]:
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
        "vwap_diff": vwap_diff,
    }


# --- USER INPUT ---
col_input, col_btn = st.columns([3, 1])
with col_input:
    ticker_input = st.selectbox(
        "Select Ticker:",
        options=configured_tickers,
        index=0 if configured_tickers else None,
    )

with col_btn:
    st.write(" ")
    analyze_click = st.button("Analyze Stock", width="stretch")

if ticker_input or analyze_click:
    with st.spinner(f"Fetching market data for {ticker_input}..."):
        data = fetch_stock_data(ticker_input)

    if not data:
        st.error(
            f"Could not retrieve data for **{ticker_input}**. Verify symbol or check internet connection."
        )
    else:
        eval_res = calculate_short_score(data)
        borrow_status, borrow_reason = estimate_borrow_status(
            data["float_shares"],
            data["short_pct_float"],
            data["gain_24h"],
            data["ctb_estimated"],
        )

        st.divider()

        # METRIC CARDS
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Short Score", f"{eval_res['final_score']}/100")
        m2.metric("Signal State", eval_res["status"])
        m3.metric(
            "Current Price",
            f"${data['price']:.2f}",
            f"{data['gain_24h']*100:+.1f}%",
        )
        m4.metric("Borrow Outlook", borrow_status)

        st.info(
            f"**State Summary:** {eval_res['status_msg']} *(Session Date: {data['trade_date']})*"
        )

        # --- 10-DAY EXTENDED HOURS PLOTLY CHART ---
        st.subheader(
            "📈 10-Day Intraday Baseline Chart (Pre/Post Market Highlighted)"
        )

        hist_df = data["hist_10d"].copy()
        eastern_tz = pytz.timezone("US/Eastern")
        if hist_df.index.tz is None:
            hist_df.index = hist_df.index.tz_localize("UTC").tz_convert(
                eastern_tz
            )
        else:
            hist_df.index = hist_df.index.tz_convert(eastern_tz)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=hist_df.index,
                y=hist_df["Close"],
                mode="lines",
                name="Price (5m Extended)",
                line=dict(color="#00B4D8", width=1.5),
            )
        )

        fig.add_hline(
            y=data["prev_close"],
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Prev Close (${data['prev_close']:.2f})",
        )

        fig.add_hline(
            y=data["vwap"],
            line_dash="dot",
            line_color="orange",
            annotation_text=f"Today's VWAP (${data['vwap']:.2f})",
        )

        unique_dates = sorted(list(set(hist_df.index.date)))

        for d in unique_dates:
            pm_start = eastern_tz.localize(
                datetime.combine(d, datetime.min.time()).replace(
                    hour=4, minute=0
                )
            )
            pm_end = eastern_tz.localize(
                datetime.combine(d, datetime.min.time()).replace(
                    hour=9, minute=30
                )
            )
            am_start = eastern_tz.localize(
                datetime.combine(d, datetime.min.time()).replace(
                    hour=16, minute=0
                )
            )
            am_end = eastern_tz.localize(
                datetime.combine(d, datetime.min.time()).replace(
                    hour=20, minute=0
                )
            )

            fig.add_vrect(
                x0=pm_start,
                x1=pm_end,
                fillcolor="rgba(255, 215, 0, 0.08)",
                layer="below",
                line_width=0,
            )
            fig.add_vrect(
                x0=am_start,
                x1=am_end,
                fillcolor="rgba(138, 43, 226, 0.12)",
                layer="below",
                line_width=0,
            )

        fig.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="Date/Time (US/Eastern)",
            yaxis_title="Stock Price ($)",
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified",
            xaxis=dict(type="date", nticks=10),
        )

        st.plotly_chart(fig, width="stretch")

        # TECHNICAL BREAKDOWN
        st.subheader("📊 Intraday Metric Breakdown")
        d1, d2, d3, d4 = st.columns(4)
        d1.write(f"**Day High:** ${data['day_high']:.2f}")
        d1.write(f"**Previous Close:** ${data['prev_close']:.2f}")

        d2.write(f"**Drop From High:** {data['drop_from_high']*100:.1f}%")
        d2.write(f"**Spike Rejection:** {data['rejection_pct']*100:.1f}%")

        vwap_relation = (
            "ABOVE 🟢" if data["price"] > data["vwap"] else "BELOW 🔴"
        )
        d3.write(
            f"**Today VWAP:** ${data['vwap']:.2f} ({vwap_relation} {eval_res['vwap_diff']:+.1f}%)"
        )
        d3.write(f"**RVOL Proxy:** {data['rvol']:.1f}x")

        float_str = (
            f"{data['float_shares']/1e6:.1f}M"
            if data["float_shares"]
            else "N/A"
        )
        d4.write(f"**Float Size:** {float_str}")
        d4.write(f"**Borrow Note:** {borrow_reason}")

        st.divider()

        # ADVANCED SHORT METRICS & CHECKLIST
        st.subheader("🔍 Micro-Cap Short Fundamentals & Criteria Checklist")

        col_metrics, col_checklist = st.columns([1, 1])

        with col_metrics:
            st.markdown("#### 📊 Live Stock Metrics")

            z_val = (
                f"{data['z_score']:.2f}"
                if data["z_score"] is not None
                else "N/A"
            )
            f_val = (
                f"{data['f_score']}/9"
                if data["f_score"] is not None
                else "N/A"
            )
            ctb_val = (
                f"{data['ctb_estimated']:.1f}% APY"
                if data["ctb_estimated"] is not None
                else "N/A"
            )

            runway_val = (
                f"{data['cash_runway_months']:.1f} Months"
                if data["cash_runway_months"] is not None
                else "N/A"
            )
            if data["cash_runway_months"] == 999.0:
                runway_val = "Cash Positive 🟢"

            burn_val = (
                f"${data['cash_burn_monthly']/1e6:.2f}M / mo"
                if data["cash_burn_monthly"]
                else "N/A"
            )
            dilution_val = (
                f"{data['share_growth_yoy']:+.1f}% YoY"
                if data["share_growth_yoy"] is not None
                else "N/A"
            )

            m_c1, m_c2 = st.columns(2)
            m_c1.write(f"**Altman Z-Score:** {z_val}")
            m_c1.write(f"**Piotroski F-Score:** {f_val}")
            m_c1.write(f"**Cash Runway:** {runway_val}")

            m_c2.write(f"**Monthly Cash Burn:** {burn_val}")
            m_c2.write(f"**Est. Borrow Rate (CTB):** {ctb_val}")
            m_c2.write(f"**Share Dilution (YoY):** {dilution_val}")

        with col_checklist:
            st.markdown("#### 📋 Short Candidate Target Criteria")

            z_match = (
                "✅ Match"
                if (data["z_score"] is not None and data["z_score"] < 1.8)
                else "❌ No"
            )
            f_match = (
                "✅ Match"
                if (data["f_score"] is not None and data["f_score"] <= 2)
                else "❌ No"
            )
            runway_match = (
                "✅ Match"
                if (
                    data["cash_runway_months"] is not None
                    and data["cash_runway_months"] < 6.0
                )
                else "❌ No"
            )
            ctb_match = (
                "✅ Match"
                if (
                    data["ctb_estimated"] is not None
                    and data["ctb_estimated"] < 20.0
                )
                else "⚠️ High Fees"
            )
            dilution_match = (
                "✅ Match"
                if (
                    data["share_growth_yoy"] is not None
                    and data["share_growth_yoy"] > 20.0
                )
                else "❌ No"
            )

            chk_df = pd.DataFrame(
                [
                    {
                        "Metric": "Altman Z-Score",
                        "Target Signal": "< 1.8 (Distress)",
                        "Status": z_match,
                    },
                    {
                        "Metric": "Piotroski F-Score",
                        "Target Signal": "0 - 2 (Weak)",
                        "Status": f_match,
                    },
                    {
                        "Metric": "Cash Runway",
                        "Target Signal": "< 6 Months",
                        "Status": runway_match,
                    },
                    {
                        "Metric": "Cost to Borrow (CTB)",
                        "Target Signal": "< 20% APY",
                        "Status": ctb_match,
                    },
                    {
                        "Metric": "Shares Growth YoY",
                        "Target Signal": "> 20% (Active Dilution)",
                        "Status": dilution_match,
                    },
                    {
                        "Metric": "Short Interest / Float",
                        "Target Signal": "< 15% (Low Squeeze)",
                        "Status": "✅ Match"
                        if data["short_pct_float"] < 0.15
                        else "⚠️ High",
                    },
                ]
            )

            st.dataframe(chk_df, width="stretch", hide_index=True)

        st.divider()

        # SCORE BREAKDOWN
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🟢 Positive Exhaustion Points")
            if eval_res["boosters"]:
                for pts, desc in eval_res["boosters"]:
                    st.write(f"• **{pts}**: {desc}")
            else:
                st.write("No positive setup points triggered.")

        with c2:
            st.markdown("### 🔴 Warning Penalties")
            if eval_res["penalties"]:
                for pts, desc in eval_res["penalties"]:
                    st.write(f"• **{pts}**: {desc}")
            else:
                st.write("No active warning penalties.")
