import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os

st.set_page_config(layout="wide")

DATA_FILE = "data/watchlist.json"

# --------------------------
# Constants
# --------------------------
DAY_COL = "1D %"
WEEK_COL = "1W %"
MONTH_COL = "1M %"

# --------------------------
# File Handling
# --------------------------
def load_watchlist():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def save_watchlist(watchlist):
    os.makedirs("data", exist_ok=True)
    with open(DATA_FILE, 'w') as f:
        json.dump(watchlist, f)

# --------------------------
# Validation
# --------------------------
def validate_ticker(ticker):
    try:
        data = yf.download(ticker, period="1d", progress=False)
        return not data.empty
    except:
        return False

# --------------------------
# Helpers
# --------------------------
def safe_round(val):
    try:
        return round(val, 2) if val is not None else None
    except:
        return None

# --------------------------
# Metrics + Scoring
# --------------------------
@st.cache_data(ttl=300)
def get_all_metrics(tickers):
    results = []

    if not tickers:
        return results

    hist = yf.download(
        tickers,
        period="1mo",
        group_by="ticker",
        auto_adjust=True,
        progress=False
    )

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)

            # --- Price data
            fast = stock.fast_info or {}
            current = fast.get("last_price")
            opening = fast.get("open")

            # --- Fundamentals
            try:
                info = stock.get_info()
            except:
                info = {}

            pe = info.get("trailingPE")
            fpe = info.get("forwardPE")
            peg = info.get("pegRatio")
            ps = info.get("priceToSalesTrailing12Months")
            pb = info.get("priceToBook")
            sector = info.get("sector")
            industry = info.get("industry")

            # --- Historical gains
            try:
                if isinstance(hist.columns, pd.MultiIndex):
                    data = hist[ticker]
                else:
                    data = hist

                close = data["Close"].dropna()

                day_gain = week_gain = month_gain = None

                if len(close) > 1 and current:
                    yesterday = close.iloc[-2]
                    if yesterday != 0:
                        day_gain = (current - yesterday) / yesterday * 100

                if len(close) >= 7 and current:
                    week_price = close.iloc[-7]
                    if week_price != 0:
                        week_gain = (current / week_price - 1) * 100

                if len(close) >= 22 and current:
                    month_price = close.iloc[-22]
                    if month_price != 0:
                        month_gain = (current / month_price - 1) * 100

            except:
                day_gain = week_gain = month_gain = None

            # --- Scoring
            def score_metric(value, good, bad):
                if value is None:
                    return None
                if value <= good:
                    return 100
                if value >= bad:
                    return 0
                return 100 * (bad - value) / (bad - good)

            scores = []
            for val, good, bad in [
                (pe, 15, 30),
                (peg, 1, 2),
                (ps, 2, 5),
                (pb, 1.5, 4)
            ]:
                s = score_metric(val, good, bad)
                if s is not None:
                    scores.append(s)

            overall_score = round(sum(scores) / len(scores), 1) if scores else None

            if overall_score is None:
                label = "N/A"
            elif overall_score >= 70:
                label = "Undervalued"
            elif overall_score >= 40:
                label = "Fair"
            else:
                label = "Expensive"

            # --- Output
            results.append({
                "Ticker": ticker.upper(),
                "Price": safe_round(current),
                "PE": safe_round(pe),
                "Forward PE": safe_round(fpe),
                "PEG": safe_round(peg),
                "PS": safe_round(ps),
                "PB": safe_round(pb),
                "Sector": sector,
                "Industry": industry,
                DAY_COL: safe_round(day_gain),
                WEEK_COL: safe_round(week_gain),
                MONTH_COL: safe_round(month_gain),
                "Score": overall_score,
                "Valuation": label,
                "Updated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            })

        except:
            continue

    return results

# --------------------------
# Styling
# --------------------------
def color_gain(val):
    if val is None:
        return ''
    return 'color: green' if val > 0 else 'color: red'

def color_valuation(val):
    if val == "Undervalued":
        return "color: green; font-weight: bold"
    elif val == "Expensive":
        return "color: red; font-weight: bold"
    elif val == "Fair":
        return "color: orange"
    return ""

# --------------------------
# UI
# --------------------------
st.title("📈 Stock Watchlist")

watchlist = load_watchlist()

# --- Display
if watchlist:
    st.subheader("Your Watchlist")

    metrics = get_all_metrics(watchlist)

    if metrics:
        df = pd.DataFrame(metrics)

        # Safe sort
        if DAY_COL in df.columns:
            df = df.sort_values(by=DAY_COL, ascending=False)

        styled_df = df.style.applymap(
            color_gain,
            subset=[col for col in [DAY_COL, WEEK_COL, MONTH_COL] if col in df.columns]
        ).applymap(
            color_valuation,
            subset=["Valuation"]
        )

        st.dataframe(styled_df, use_container_width=True)
    else:
        st.write("No data available.")
else:
    st.write("Your watchlist is empty. Add some tickers below.")

# --- Add ticker
st.subheader("➕ Add Ticker")

ticker_input = st.text_input("Enter stock ticker (e.g., AAPL)")

if st.button("Add Ticker"):
    if ticker_input:
        ticker = ticker_input.upper().strip()

        if ticker in watchlist:
            st.warning("Ticker already in watchlist.")
        elif validate_ticker(ticker):
            watchlist.append(ticker)
            save_watchlist(watchlist)
            st.success(f"Added {ticker}")
            st.rerun()
        else:
            st.error("Invalid ticker.")
    else:
        st.warning("Please enter a ticker.")

# --- Delete tickers
if watchlist:
    st.subheader("🗑️ Delete Tickers")

    selected = [
        ticker for ticker in watchlist
        if st.checkbox(f"Delete {ticker}", key=f"del_{ticker}")
    ]

    if st.button("Delete Selected"):
        if selected:
            watchlist = [t for t in watchlist if t not in selected]
            save_watchlist(watchlist)
            st.success(f"Deleted: {', '.join(selected)}")
            st.rerun()
        else:
            st.warning("No tickers selected.")
