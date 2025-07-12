import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import mibian

# Set the app title
st.title("TQQQ Covered Call Analyzer with Greeks")

# Layout input controls
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
# Get tickers from session state and split into a list
    tickers_list = [t.strip().upper() for t in st.session_state.get("tickers", "").split(",") if t.strip()]

    # Ticker selector
    ticker = st.selectbox("Select Stock Ticker", tickers_list) if tickers_list else ""

if ticker:
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.info.get("regularMarketPrice", None)
        options = stock.options

        if current_price is None:
            st.error("Could not retrieve current stock price.")
        elif not options:
            st.error("No options data available for this stock.")
        else:
            with col2:
                st.metric("Current Price", f"${current_price:.2f}")
            with col3:
                selected_date = st.selectbox("Expiration Date", options)

            # Get call option chain
            calls = stock.option_chain(selected_date).calls

            # Filter by strike range: 90% to 120% of current price
            lower_bound = 0.9 * current_price
            upper_bound = 1.2 * current_price
            filtered_calls = calls[
                (calls["strike"] >= lower_bound) & 
                (calls["strike"] <= upper_bound)
            ]

            # Calculate time to expiry in days
            today = datetime.datetime.today()
            expiry = datetime.datetime.strptime(selected_date, "%Y-%m-%d")
            days_to_expiry = (expiry - today).days

            # Risk-free rate (you can adjust or fetch from online)
            risk_free_rate = 5  # in percent

            # Calculate Greeks using mibian for each option
            def compute_greeks(row):
                try:
                    S = current_price
                    K = row["strike"]
                    t = days_to_expiry
                    iv = row["impliedVolatility"] * 100  # Convert to percent
                    if iv <= 0 or t <= 0:
                        return pd.Series([None, None])
                    bs = mibian.BS([S, K, risk_free_rate, t], volatility=iv)
                    return pd.Series([round(bs.callDelta, 3), round(bs.callTheta, 3)])
                except:
                    return pd.Series([None, None])

            greeks = filtered_calls.apply(compute_greeks, axis=1)
            filtered_calls["Delta"] = greeks[0]
            filtered_calls["Theta"] = greeks[1]

            # Display final dataframe
            df = filtered_calls[[
                "strike", "lastPrice", "openInterest", "volume", "impliedVolatility", "Delta", "Theta"
            ]].rename(columns={
                "strike": "Strike",
                "lastPrice": "Premium",
                "openInterest": "OI",
                "volume": "Vol",
                "impliedVolatility": "IV"
            })

            df["Strike/Price"] = (df["Strike"] / current_price).round(3)
            df["Premium/Price"] = (df["Premium"] / current_price).round(3)
            df["IV"] = (df["IV"] * 100).round(2)

            st.subheader("Filtered Call Options with Greeks")
            # Select and format columns
            df_display = df[["Strike", "Premium", "Delta", "Strike/Price", "Premium/Price"]].copy()
            df_display = df_display[df_display["Delta"] <= 0.5]
            
            d1,d2 = st.columns((2,3))
            with d1:
                # Display in Streamlit
                st.dataframe(df_display)

            with st.expander("📘 Column Descriptions"):
                st.markdown("""
                **Strike** – The price at which the option allows you to buy the stock.<br>
                **Premium** – The current price you receive for selling the option (per share).<br>
                **OI (Open Interest)** – The total number of open contracts for that strike.<br>
                **Vol (Volume)** – The number of contracts traded today.<br>
                **IV (%)** – Implied Volatility: the market's forecast of the stock's future volatility.<br>
                **Delta** – The expected change in option price for a $1 move in the stock.<br>
                **Theta** – The amount the option price decays per day (time decay).
                """, unsafe_allow_html=True)


    except Exception as e:
        st.error(f"An error occurred: {e}")

    
