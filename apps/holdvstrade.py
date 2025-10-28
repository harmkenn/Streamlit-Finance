import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Stock Summary", layout="wide")
st.title("📊 Stock Summary Dashboard")

# --- Input for tickers ---
st.markdown("Enter a comma-separated list of stock symbols (e.g. AAPL, MSFT, NVDA):")
tickers_input = st.text_input("Tickers", value=st.session_state.get("tickers", "AAPL, MSFT, NVDA"))
st.session_state["tickers"] = tickers_input  # store tickers in session state
tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

refresh_button = st.button("🔄 Refresh Data")

# --- Main Table ---
if tickers_list:
    rows = []

    for t in tickers_list:
        try:
            yf_t = yf.Ticker(t)
            t_data = yf_t.history(period="1d", interval="1m", prepost=True)
            month_data = yf_t.history(period="1mo", interval="1d")

            if not t_data.empty and not month_data.empty:
                latest = t_data["Close"].iloc[-1]
                prev_close = month_data["Close"].iloc[-2] if len(month_data) > 1 else month_data["Close"].iloc[-1]
                high_30d = month_data["High"].max()
                low_30d = month_data["Low"].min()

                price_diff = latest - prev_close
                percent_diff = (price_diff / prev_close) * 100 if prev_close != 0 else 0
                position = (latest - low_30d) / (high_30d - low_30d) * 100 if high_30d != low_30d else 50

                rows.append({
                    "Ticker": t,
                    "Last Price": f"${latest:.2f}",
                    "Change": f"{price_diff:+.2f}",
                    "Change %": f"{percent_diff:+.2f}%",
                    "30d Low": f"${low_30d:.2f}",
                    "30d High": f"${high_30d:.2f}",
                    "Range Position %": f"{position:.1f}%"
                })
            else:
                rows.append({"Ticker": t, "Error": "No data"})

        except Exception as e:
            rows.append({"Ticker": t, "Error": str(e)})

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    if refresh_button:
        st.experimental_rerun()
else:
    st.info("👆 Enter at least one ticker symbol to display stock information.")
