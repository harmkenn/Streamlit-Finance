import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Nasdaq Analysis and Prediction App",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Nasdaq Analysis and Prediction App")
st.markdown("""
This app analyzes the Nasdaq Composite index using technical indicators, macroeconomic data, market sentiment, and external events, and provides 5-day price predictions using the Prophet model.
""")

# Sidebar for user input
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Enter Ticker Symbol:", "^IXIC", help="Use ^IXIC for Nasdaq Composite")
    analysis_period = st.number_input("Analysis Period (Days):", min_value=30, max_value=365, value=90)
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Enter the ticker symbol (e.g., ^IXIC for Nasdaq Composite).
    2. Specify the analysis period (e.g., last 90 days).
    3. Click 'Analyze Market' to get insights and predictions.
    """)

# Analyze button
if st.button("🚀 Analyze Market"):
    if not ticker:
        st.error("⚠️ Please enter a ticker symbol.")
    else:
        with st.spinner("🔄 Fetching data and analyzing..."):
            try:
                # Fetch historical data using yfinance
                stock = yf.Ticker(ticker)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=analysis_period)
                data = stock.history(start=start_date, end=end_date)

                if data.empty:
                    st.error("❌ No data found for the given ticker symbol. Please try another.")
                else:
                    # --- Old Analysis: Technical Indicators ---
                    # Calculate key metrics
                    current_price = data['Close'][-1]
                    avg_price = data['Close'].mean()
                    high_price = data['High'].max()
                    low_price = data['Low'].min()

                    # RSI calculation
                    delta = data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))

                    # Moving averages
                    data['MA_10'] = data['Close'].rolling(window=10).mean()
                    data['MA_50'] = data['Close'].rolling(window=50).mean()

                    # Display old analysis results
                    st.success(f"✅ Analysis complete for {ticker.upper()}!")
                    st.markdown(f"### Current Price: ${current_price:.2f}")
                    st.markdown(f"### Key Metrics:")
                    st.markdown(f"- **Average Price (Last {analysis_period} Days)**: ${avg_price:.2f}")
                    st.markdown(f"- **High Price**: ${high_price:.2f}")
                    st.markdown(f"- **Low Price**: ${low_price:.2f}")
                    st.markdown(f"- **RSI**: {rsi[-1]:.2f} (Overbought > 70, Oversold < 30)")

                    # Plot historical prices and moving averages
                    st.markdown("### Price Trend:")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['MA_10'], mode='lines', name='10-Day MA'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], mode='lines', name='50-Day MA'))
                    fig.update_layout(title="Nasdaq Price Trend", xaxis_title="Date", yaxis_title="Price (USD)")
                    st.plotly_chart(fig)

                    # --- New Analysis: Price Predictions ---
                    # Prepare data for Prophet
                    df = data.reset_index()
                    df = df[['Date', 'Close']]
                    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

                    # Remove timezone from the 'ds' column
                    df['ds'] = df['ds'].dt.tz_localize(None)  # Fix timezone issue

                    # Train Prophet model
                    model = Prophet(daily_seasonality=True)
                    model.fit(df)

                    # Make predictions for the next 5 days
                    future = model.make_future_dataframe(periods=5)
                    forecast = model.predict(future)

                    # Extract predictions
                    predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)

                    # Display predictions
                    st.markdown("### 5-Day Price Predictions:")
                    st.dataframe(predictions)

                    # Plot predictions
                    st.markdown("### Price Trend and Predictions:")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Prices'))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Prices'))
                    fig.update_layout(title="Nasdaq Price Trend and Predictions", xaxis_title="Date", yaxis_title="Price (USD)")
                    st.plotly_chart(fig)

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit, yFinance, Prophet, and Plotly | Powered by Python</p>
</div>
""", unsafe_allow_html=True)