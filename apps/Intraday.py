import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
#v6.0
# Page configuration
st.set_page_config(page_title="Intraday Stock Prices", layout="wide")
st.title("📈 Intraday Stock Prices (Including Pre-market, After-hours & Overnight) v6.0")

# --- User input for stock symbol ---
col1, col2, col3 = st.columns(3)
with col1:
    # Get tickers from session state and split into a list
    tickers_list = [t.strip().upper() for t in st.session_state.get("tickers", "").split(",") if t.strip()]

    # Ticker selector
    ticker = st.selectbox("Select Stock Ticker", tickers_list) if tickers_list else ""
    refresh_button = st.button("Refresh")

# --- Main Chart Area ---
if ticker:
    try:
        # Fetch stock data (5-minute interval for 5 days to capture extended hours)
        yf_ticker = yf.Ticker(ticker)
        data = yf_ticker.history(period="5d", interval="5m", prepost=True)

        if data.empty:
            st.error(f"No data found for {ticker}. Please check the symbol and try again.")
        else:
            # Convert timestamps to Eastern Time for session filtering
            data = data.tz_convert("America/New_York")

        # -----------------------------
        # ⭐ ADD OVERNIGHT SESSION HERE
        # -----------------------------
        overnight = pd.concat([
            data.between_time("20:00", "23:59"),
            data.between_time("00:00", "04:00")
        ])

        # -----------------------------
        # Display Overnight Stats
        # -----------------------------
        with col2:
            # Get the latest price
            latest_price = data["Close"].iloc[-1]

            # Filter only regular trading hours
            regular_hours = data.between_time("09:30", "16:00")

            # Group by date and get the last close of each day
            daily_closes = regular_hours.groupby(regular_hours.index.date).last()

            if len(daily_closes) >= 4:
                recent_closes = daily_closes.tail(4)
            else:
                recent_closes = daily_closes

            close_dates = recent_closes.index.tolist()
            close_values = recent_closes["Close"].tolist()

            # Display previous 3 closes and changes
            for i in range(1, len(close_values)):
                date_str = close_dates[i].strftime("%Y-%m-%d")
                close_price = close_values[i]
                prev_close_price = close_values[i - 1]
                price_change = close_price - prev_close_price
                percent_change = (price_change / prev_close_price) * 100 if prev_close_price != 0 else 0
                color = "green" if percent_change >= 0 else "red"

                st.markdown(
                    f"### {date_str}: ${close_price:.2f} "
                    f"<span style='color:{color}; font-size:16px'>({price_change:+.2f}, {percent_change:+.2f}%)</span>",
                    unsafe_allow_html=True
                )

            # Display current price and change from most recent close
            last_close_price = close_values[-1]
            price_diff = latest_price - last_close_price
            percent_diff = (price_diff / last_close_price) * 100 if last_close_price != 0 else 0
            color = "green" if percent_diff >= 0 else "red"

            st.markdown(
                f"### Current Price: ${latest_price:.2f} "
                f"<span style='color:{color}; font-size:20px'>({percent_diff:+.2f}%)</span>",
                unsafe_allow_html=True
            )

            # -----------------------------
            # ⭐ OVERNIGHT PRICE DISPLAY
            # -----------------------------
            st.markdown("### 🌙 Overnight Session (8 PM → 4 AM)")

            if not overnight.empty:
                overnight_low = overnight["Low"].min()
                overnight_high = overnight["High"].max()
                overnight_last = overnight["Close"].iloc[-1]

                st.write(f"**Overnight Low:** ${overnight_low:.2f}")
                st.write(f"**Overnight High:** ${overnight_high:.2f}")
                st.write(f"**Last Overnight Price:** ${overnight_last:.2f}")
            else:
                st.write("No overnight data available.")

        # --- Stats Table in Col3 ---
        with col3:
            stats_data = yf_ticker.history(period="3mo", interval="1d")
            
            if not stats_data.empty:
                w1 = stats_data.tail(5)
                w3 = stats_data.tail(15)
                w5 = stats_data.tail(25)

                stats_df = pd.DataFrame({
                    "Metric": ["5 Week High", "3 Week High", "1 Week High", "5 Week Avg", "1 Week Low", "3 Week Low", "5 Week Low"],
                    "Value": [w5["High"].max(), w3["High"].max(), w1["High"].max(), w5["Close"].mean(), w1["Low"].min(), w3["Low"].min(), w5["Low"].min()]
                })
                
                stats_df["Value"] = stats_df["Value"].apply(lambda x: f"${x:,.2f}")
                
                st.dataframe(stats_df, hide_index=True, width='stretch')

        # --- Price Chart ---
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Price",
            line=dict(color="blue")
        ))
        price_fig.update_layout(
            title=f"{ticker} Intraday Price (Including Pre-market, After-hours & Overnight)",
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True
        )
        st.plotly_chart(price_fig)

        # --- Volume Chart ---
        volume_fig = go.Figure()
        volume_fig.add_trace(go.Bar(
            x=data.index,
            y=data["Volume"],
            name="Volume",
            marker=dict(color="grey")
        ))
        volume_fig.update_layout(
            title=f"{ticker} Intraday Volume (Including Pre-market, After-hours & Overnight)",
            xaxis_title="Time",
            yaxis_title="Volume",
            showlegend=True
        )
        st.plotly_chart(volume_fig)

        # Show raw data (reversed for most recent on top)
        st.write(data[["Close", "Volume"]][::-1])

        # Refresh button logic
        if refresh_button:
            st.experimental_rerun()

    except Exception as e:
        st.error(f"Error fetching data: {e}")

# --- Sidebar Section ---
st.sidebar.header("📊 Current Prices & 60-Day Range")

if tickers_list:
    for t in tickers_list:
        try:
            yf_t = yf.Ticker(t)

            # Today's intraday with pre/post
            t_data = yf_t.history(period="1d", interval="1m", prepost=True)

            # 60-day daily candles
            month_data = yf_t.history(period="2mo", interval="1d")

            # Build yesterday's close from intraday
            intraday = yf_t.history(period="10d", interval="5m", prepost=True)

            if not intraday.empty:
                intraday = intraday.tz_convert("America/New_York")
                regular = intraday.between_time("09:30", "16:00")
                daily_from_intraday = regular.groupby(regular.index.date).last()

                if len(daily_from_intraday) >= 2:
                    prev_close = daily_from_intraday["Close"].iloc[-2]
                else:
                    prev_close = None
            else:
                prev_close = None

            if not t_data.empty and prev_close is not None and not month_data.empty:
                latest = t_data["Close"].iloc[-1]

                price_diff = latest - prev_close
                percent_diff = (price_diff / prev_close) * 100 if prev_close != 0 else 0
                color = "green" if percent_diff >= 0 else "red"

                high_60d = month_data["High"].max()
                low_60d = month_data["Low"].min()

                position = (latest - low_60d) / (high_60d - low_60d) * 100 if high_60d != low_60d else 50

                bar_html = f"""
                <div style='width:100%; height:6px; background:#ddd; border-radius:3px; margin-top:2px; margin-bottom:6px; position:relative;'>
                    <div style='position:absolute; left:{position}%; top:0; transform:translateX(-50%);
                                width:4px; height:6px; background:{color}; border-radius:2px;'></div>
                </div>
                <small>60-day range: ${low_60d:.2f} – ${high_60d:.2f}</small>
                """

                st.sidebar.markdown(
                    f"**{t}**: ${latest:.2f} "
                    f"<span style='color:{color}'>({percent_diff:+.2f}%)</span>",
                    unsafe_allow_html=True
                )
                st.sidebar.markdown(bar_html, unsafe_allow_html=True)

            else:
                st.sidebar.write(f"**{t}**: No data")

        except Exception as e:
            st.sidebar.write(f"**{t}**: Error ({e})")
