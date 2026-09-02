import streamlit as st
from yahooquery import Ticker
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.title("📈 Historical Investment Growth (DRIP Simulation)")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_drip_history(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Fetch batch history, falling back to isolated requests for partial responses."""
    try:
        history = Ticker(list(tickers)).history(start=start, end=end)
        if isinstance(history, pd.DataFrame) and not history.empty:
            history = history.reset_index()
            required_columns = {"symbol", "date", "close", "dividends"}
            if required_columns.issubset(history.columns) and set(tickers).issubset(
                set(history["symbol"].dropna().unique())
            ):
                return history
    except Exception:
        pass

    ticker_frames = []
    for ticker in tickers:
        try:
            history = Ticker(ticker).history(start=start, end=end)
            if isinstance(history, pd.DataFrame) and not history.empty:
                ticker_frames.append(history.reset_index())
        except Exception:
            continue

    return pd.concat(ticker_frames, ignore_index=True) if ticker_frames else pd.DataFrame()


# --- Parse tickers from shared session state ---
raw_tickers = st.session_state.get("user_tickers", "")
ticker_list = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

if not ticker_list:
    st.warning("⚠️ No tickers found in the sidebar. Please enter comma-separated stock tickers in the sidebar.")
    st.stop()

# --- Controls Layout ---
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    tickers = st.multiselect(
        "Select Tickers to Compare",
        options=ticker_list,
        default=ticker_list[:min(5, len(ticker_list))]
    )

with col2:
    initial_investment = st.number_input(
        "Initial Investment ($)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=5000
    )

with col3:
    time_horizon = st.selectbox(
        "Time Horizon",
        options=["1 Year", "2 Years", "3 Years", "5 Years"],
        index=0
    )

if not tickers:
    st.info("Select at least one ticker above to simulate growth.")
    st.stop()

# Determine start date
end_date = datetime.today()
years_map = {"1 Year": 1, "2 Years": 2, "3 Years": 3, "5 Years": 5}
years = years_map.get(time_horizon, 1)
start_date = end_date - timedelta(days=365 * years)

fig = go.Figure()

# --- Batch Fetch Data for Selected Tickers ---
try:
    history = fetch_drip_history(
        tuple(tickers),
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

    if isinstance(history, pd.DataFrame) and not history.empty and 'close' in history.columns:
        history = history.reset_index()

        summary_data = []

        for ticker_symbol in tickers:
            df = history[history['symbol'] == ticker_symbol].copy()

            if df.empty or len(df) < 2:
                st.warning(f"Insufficient historical data for {ticker_symbol}.")
                continue

            df = df[['date', 'close', 'dividends']].sort_values('date').reset_index(drop=True)
            df['dividends'] = df['dividends'].fillna(0)

            # --- DRIP Calculation ---
            initial_close = df.loc[0, 'close']
            if initial_close == 0 or pd.isna(initial_close):
                continue

            current_shares = initial_investment / initial_close
            investment_values = []
            share_counts = []

            for idx, row in df.iterrows():
                close_price = row['close']
                dividend = row['dividends']

                if dividend > 0 and close_price > 0:
                    # Reinvest dividend payout into additional shares
                    new_shares = (dividend * current_shares) / close_price
                    current_shares += new_shares

                share_counts.append(current_shares)
                investment_values.append(current_shares * close_price)

            df['shares'] = share_counts
            df['investment_value'] = investment_values

            final_val = df['investment_value'].iloc[-1]
            pct_change = ((final_val / initial_investment) - 1) * 100

            summary_data.append({
                "Ticker": ticker_symbol,
                "Initial Value": f"${initial_investment:,.2f}",
                "Final Value": f"${final_val:,.2f}",
                "Total Return": f"{pct_change:+.2f}%",
                "Final Shares": f"{df['shares'].iloc[-1]:,.2f}"
            })

            # Line plot for portfolio growth
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['investment_value'],
                mode='lines',
                name=f"{ticker_symbol} ({pct_change:+.1f}%)",
                hovertemplate=(
                    f"<b>{ticker_symbol}</b><br>"
                    "Date: %{x}<br>"
                    "Value: $%{y:,.2f}<br>"
                    f"Total Return: {pct_change:+.1f}%<extra></extra>"
                )
            ))

            # Dividend markers
            dividend_days = df[df['dividends'] > 0]
            if not dividend_days.empty:
                fig.add_trace(go.Scatter(
                    x=dividend_days['date'],
                    y=dividend_days['investment_value'],
                    mode='markers',
                    name=f"{ticker_symbol} Dividend",
                    marker=dict(size=8, symbol='star'),
                    hovertemplate=(
                        f"<b>{ticker_symbol} Dividend Paid</b><br>"
                        "Date: %{x}<br>"
                        "Payout/Share: $%{text:.2f}<br>"
                        "Portfolio Value: $%{y:,.2f}<extra></extra>"
                    ),
                    text=dividend_days['dividends']
                ))

        fig.update_layout(
            title=f"📊 DRIP Performance ({time_horizon}) — Starting Capital: ${initial_investment:,.2f}",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (USD)",
            template="plotly_white",
            height=600,
            hovermode="x unified"
        )

        st.plotly_chart(fig, width="stretch")

        # Display Summary Breakdown Table
        if summary_data:
            st.subheader("📋 Performance Summary Table")
            st.dataframe(pd.DataFrame(summary_data), width="stretch", hide_index=True)

    else:
        st.error(f"⚠️ Could not retrieve historical dividend data for selected tickers: {', '.join(tickers)}")

except Exception as e:
    st.error(f"Error fetching dividend or price history: {e}")
