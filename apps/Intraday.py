import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

st.title("📈 Intraday Stock Prices (Including Pre-market & After-hours) v5.0")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker: str, period: str, interval: str, prepost: bool = False) -> pd.DataFrame:
    """Cache Yahoo history so one rerun does not issue duplicate requests."""
    return yf.Ticker(ticker).history(
        period=period,
        interval=interval,
        prepost=prepost,
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily_histories(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Fetch all daily sidebar data in one Yahoo request."""
    return yf.download(
        tickers=list(tickers),
        period="3mo",
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by="ticker",
    )


def get_ticker_daily_data(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            return data[ticker].dropna(how="all")
        if ticker in data.columns.get_level_values(1):
            return data.xs(ticker, axis=1, level=1).dropna(how="all")
        return pd.DataFrame()
    return data.dropna(how="all")


# --- Parse tickers from shared session state ---
raw_tickers = st.session_state.get("user_tickers", "")
tickers_list = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.selectbox("Select Stock Ticker", tickers_list) if tickers_list else ""
    refresh_button = st.button("Refresh")

# --- Main Chart Area ---
if ticker:
    try:
        if refresh_button:
            fetch_history.clear()
            fetch_daily_histories.clear()

        daily_histories = fetch_daily_histories(tuple(dict.fromkeys(tickers_list))) if tickers_list else pd.DataFrame()

        data = fetch_history(ticker, "10d", "5m", prepost=True)

        if data.empty:
            st.error(f"No data found for {ticker}. Please check the symbol and try again.")
            data = None
        elif data.index.tz is not None:
            data = data.tz_convert("America/New_York")

        if data is None:
            st.stop()

        with col2:
            latest_price = data["Close"].iloc[-1]
            regular_hours = data.between_time("09:30", "16:00")
            daily_closes = regular_hours.groupby(regular_hours.index.date).last()

            if len(daily_closes) >= 4:
                recent_closes = daily_closes.tail(4)
            else:
                recent_closes = daily_closes

            close_dates = recent_closes.index.tolist()
            close_values = recent_closes["Close"].tolist()

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

            last_close_price = close_values[-1]
            price_diff = latest_price - last_close_price
            percent_diff = (price_diff / last_close_price) * 100 if last_close_price != 0 else 0
            color = "green" if percent_diff >= 0 else "red"

            st.markdown(
                f"### Current Price: ${latest_price:.2f} "
                f"<span style='color:{color}; font-size:20px'>({percent_diff:+.2f}%)</span>",
                unsafe_allow_html=True
            )

        # --- Stats Table ---
        with col3:
            stats_data = get_ticker_daily_data(daily_histories, ticker)
            
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
            x=data.index, y=data["Close"], mode="lines", name="Price", line=dict(color="blue")
        ))
        price_fig.update_layout(
            title=f"{ticker} Intraday Price (Including Pre-market & After-hours)",
            xaxis_title="Time", yaxis_title="Price", showlegend=True
        )
        st.plotly_chart(price_fig)

        # --- Volume Chart ---
        volume_fig = go.Figure()
        volume_fig.add_trace(go.Bar(
            x=data.index, y=data["Volume"], name="Volume", marker=dict(color="grey")
        ))
        volume_fig.update_layout(
            title=f"{ticker} Intraday Volume (Including Pre-market & After-hours)",
            xaxis_title="Time", yaxis_title="Volume", showlegend=True
        )
        st.plotly_chart(volume_fig)

        st.write(data[["Close", "Volume"]][::-1])

    except Exception as e:
        st.error(f"Error fetching data: {e}")

# --- Sidebar Section ---
st.sidebar.header("📊 Current Prices & 60-Day Range")

if tickers_list:
    daily_histories = fetch_daily_histories(tuple(dict.fromkeys(tickers_list)))
    for t in tickers_list:
        try:
            month_data = get_ticker_daily_data(daily_histories, t)
            if len(month_data) >= 2:
                latest = month_data["Close"].iloc[-1]
                prev_close = month_data["Close"].iloc[-2]
                price_diff = latest - prev_close
                percent_diff = (price_diff / prev_close) * 100 if prev_close != 0 else 0
                color = "green" if percent_diff >= 0 else "red"

                high_60d = month_data["High"].max()
                low_60d = month_data["Low"].min()
                position = (latest - low_60d) / (high_60d - low_60d) * 100 if high_60d != low_60d else 50

                bar_html = f"""
                <div style='width:100%; height:6px; background:#ddd; border-radius:3px; margin-top:2px; margin-bottom:6px; position:relative;'>
                    <div style='position:absolute; left:{position}%; top:0; transform:translateX(-50%); width:4px; height:6px; background:{color}; border-radius:2px;'></div>
                </div>
                <small>60-day range: ${low_60d:.2f} – ${high_60d:.2f}</small>
                """

                st.sidebar.markdown(
                    f"**{t}**: ${latest:.2f} <span style='color:{color}'>({percent_diff:+.2f}%)</span>",
                    unsafe_allow_html=True
                )
                st.sidebar.markdown(bar_html, unsafe_allow_html=True)

            else:
                st.sidebar.write(f"**{t}**: No data")

        except Exception as e:
            st.sidebar.write(f"**{t}**: Error ({e})")
