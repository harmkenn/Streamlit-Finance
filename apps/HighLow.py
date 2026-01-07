import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# v 1.4
st.set_page_config(page_title="TQQQ Trading Analyzer", layout="wide")

st.title("🎯 TQQQ Buy/Sell Target Price Analyzer")
st.caption("⚠️ For educational purposes only. Not financial advice.")

# Sidebar parameters
st.sidebar.header("Strategy Parameters")
lookback_period = st.sidebar.slider("Analysis Period (months)", 6, 36, 12)

st.sidebar.subheader("Core Indicators")
rsi_oversold = st.sidebar.slider("RSI Oversold (Buy Signal)", 20, 40, 35)
rsi_overbought = st.sidebar.slider("RSI Overbought (Sell Signal)", 60, 80, 65)
ma_short = st.sidebar.slider("Short Moving Average (days)", 5, 20, 10)
ma_long = st.sidebar.slider("Long Moving Average (days)", 20, 60, 30)

st.sidebar.subheader("Strategy Enhancements")
use_trend_filter = st.sidebar.checkbox("Use Trend Filter (50-day MA)", value=True)
use_volatility_filter = st.sidebar.checkbox("Use Volatility Filter", value=True)
use_momentum_confirm = st.sidebar.checkbox("Require MA Crossover Confirmation", value=True)
use_ml_prediction = st.sidebar.checkbox("Use ML Price Prediction", value=True)
profit_target = st.sidebar.slider("Profit Target (%)", 5, 30, 15)
ml_confidence_threshold = st.sidebar.slider("ML Confidence Threshold (%)", 50, 80, 60)

# Fetch TQQQ data
@st.cache_data(ttl=3600)
def fetch_data(months):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months*30)
    ticker = yf.Ticker("TQQQ")
    data = ticker.history(start=start_date, end=end_date)
    return data

# Calculate technical indicators
def calculate_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['MA_Short'] = df['Close'].rolling(window=ma_short).mean()
    df['MA_Long'] = df['Close'].rolling(window=ma_long).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()  # Trend filter
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # ATR for volatility
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
    
    return df

# Create ML features
def create_ml_features(df):
    """Create features for machine learning prediction"""
    ml_df = df.copy()
    
    # Price-based features
    ml_df['Returns_1d'] = df['Close'].pct_change(1)
    ml_df['Returns_3d'] = df['Close'].pct_change(3)
    ml_df['Returns_5d'] = df['Close'].pct_change(5)
    ml_df['Returns_10d'] = df['Close'].pct_change(10)
    
    # Volatility features
    ml_df['Volatility_5d'] = df['Close'].pct_change().rolling(5).std()
    ml_df['Volatility_10d'] = df['Close'].pct_change().rolling(10).std()
    
    # Volume features
    ml_df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    ml_df['Volume_Change'] = df['Volume'].pct_change()
    
    # Technical indicator features
    ml_df['RSI_Change'] = df['RSI'].diff()
    ml_df['Price_to_MA20'] = df['Close'] / df['BB_Middle']
    ml_df['Price_to_MA50'] = df['Close'] / df['MA_50']
    ml_df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    ml_df['Distance_to_BB_Upper'] = (df['BB_Upper'] - df['Close']) / df['Close']
    ml_df['Distance_to_BB_Lower'] = (df['Close'] - df['BB_Lower']) / df['Close']
    
    # Momentum features
    ml_df['MA_Short_Slope'] = df['MA_Short'].diff()
    ml_df['MA_Long_Slope'] = df['MA_Long'].diff()
    ml_df['MACD'] = df['MA_Short'] - df['MA_Long']
    
    # Target: 1 if price goes up tomorrow, 0 if down
    ml_df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Target for regression: actual next day return
    ml_df['Target_Return'] = df['Close'].pct_change().shift(-1)
    
    return ml_df

# Train ML model
@st.cache_data(ttl=3600)
def train_ml_model(df):
    """Train Random Forest to predict price direction"""
    ml_df = create_ml_features(df)
    
    # Select features
    feature_cols = ['Returns_1d', 'Returns_3d', 'Returns_5d', 'Returns_10d',
                    'Volatility_5d', 'Volatility_10d', 'Volume_Ratio', 'Volume_Change',
                    'RSI', 'RSI_Change', 'Price_to_MA20', 'Price_to_MA50', 
                    'BB_Width', 'Distance_to_BB_Upper', 'Distance_to_BB_Lower',
                    'MA_Short_Slope', 'MA_Long_Slope', 'MACD', 'ATR_Pct']
    
    # Remove NaN rows
    ml_df = ml_df.dropna()
    
    if len(ml_df) < 100:
        return None, None, None, None
    
    X = ml_df[feature_cols]
    y = ml_df['Target']
    
    # Split data (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, scaler, feature_importance, test_accuracy

# Get ML prediction
def get_ml_prediction(df, model, scaler):
    """Get ML prediction for next day"""
    if model is None:
        return None, None
    
    ml_df = create_ml_features(df)
    
    feature_cols = ['Returns_1d', 'Returns_3d', 'Returns_5d', 'Returns_10d',
                    'Volatility_5d', 'Volatility_10d', 'Volume_Ratio', 'Volume_Change',
                    'RSI', 'RSI_Change', 'Price_to_MA20', 'Price_to_MA50', 
                    'BB_Width', 'Distance_to_BB_Upper', 'Distance_to_BB_Lower',
                    'MA_Short_Slope', 'MA_Long_Slope', 'MACD', 'ATR_Pct']
    
    # Get last row (most recent data)
    last_row = ml_df[feature_cols].iloc[-1:].values
    
    if np.isnan(last_row).any():
        return None, None
    
    # Scale and predict
    last_row_scaled = scaler.transform(last_row)
    prediction = model.predict(last_row_scaled)[0]
    probabilities = model.predict_proba(last_row_scaled)[0]
    
    # Confidence is the probability of the predicted class
    confidence = probabilities[prediction] * 100
    
    return prediction, confidence

# Generate signals
def generate_signals(df, rsi_low, rsi_high, ml_model=None, ml_scaler=None):
    df['Signal'] = 0
    
    # Get ML predictions for each row if model provided
    ml_predictions = []
    ml_confidences = []
    
    if use_ml_prediction and ml_model is not None:
        for i in range(len(df)):
            if i < 50:  # Need enough history for features
                ml_predictions.append(None)
                ml_confidences.append(None)
            else:
                df_slice = df.iloc[:i+1]
                pred, conf = get_ml_prediction(df_slice, ml_model, ml_scaler)
                ml_predictions.append(pred)
                ml_confidences.append(conf)
        
        df['ML_Prediction'] = ml_predictions
        df['ML_Confidence'] = ml_confidences
    
    # Base buy condition: RSI oversold AND price near lower Bollinger Band
    buy_condition = (df['RSI'] < rsi_low) & (df['Close'] < df['BB_Lower'] * 1.05)
    
    # Add trend filter: only buy if above 50-day MA (in uptrend)
    if use_trend_filter:
        buy_condition = buy_condition & (df['Close'] > df['MA_50'])
    
    # Add volatility filter: avoid buying in extreme volatility
    if use_volatility_filter:
        atr_threshold = df['ATR_Pct'].quantile(0.75)  # Top 25% volatility
        buy_condition = buy_condition & (df['ATR_Pct'] < atr_threshold)
    
    # Add momentum confirmation: short MA should be turning up
    if use_momentum_confirm:
        buy_condition = buy_condition & (df['MA_Short'] > df['MA_Short'].shift(1))
    
    # Add ML prediction filter: only buy if ML predicts price will go up
    if use_ml_prediction and ml_model is not None:
        ml_buy_condition = (df['ML_Prediction'] == 1) & (df['ML_Confidence'] >= ml_confidence_threshold)
        buy_condition = buy_condition & ml_buy_condition
    
    df.loc[buy_condition, 'Signal'] = 1
    
    # Enhanced sell conditions
    sell_condition_1 = (df['RSI'] > rsi_high) & (df['Close'] > df['BB_Upper'] * 0.98)  # Overbought
    sell_condition_2 = (df['Close'] < df['MA_50']) & (df['RSI'] < 50)  # Trend break with weak momentum
    
    # ML-based sell: if ML predicts down with high confidence
    if use_ml_prediction and ml_model is not None:
        ml_sell_condition = (df['ML_Prediction'] == 0) & (df['ML_Confidence'] >= ml_confidence_threshold)
        sell_condition_1 = sell_condition_1 | ml_sell_condition
    
    df.loc[sell_condition_1 | sell_condition_2, 'Signal'] = -1
    
    return df

# Backtest strategy
def backtest_strategy(df):
    position = 0
    trades = []
    entry_price = 0
    
    for i in range(len(df)):
        if df['Signal'].iloc[i] == 1 and position == 0:  # Buy signal
            position = 1
            entry_price = df['Close'].iloc[i]
            trades.append({
                'Date': df.index[i],
                'Type': 'BUY',
                'Price': entry_price,
                'Return': 0
            })
        elif df['Signal'].iloc[i] == -1 and position == 1:  # Sell signal
            position = 0
            exit_price = df['Close'].iloc[i]
            ret = ((exit_price - entry_price) / entry_price) * 100
            trades.append({
                'Date': df.index[i],
                'Type': 'SELL',
                'Price': exit_price,
                'Return': ret
            })
    
    return pd.DataFrame(trades)

# Portfolio simulation
def simulate_portfolio(df, initial_capital=100000):
    cash = initial_capital
    shares = 0
    entry_price = 0
    portfolio_value = []
    
    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        
        # Check profit target if we have shares
        if shares > 0 and entry_price > 0:
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            if profit_pct >= profit_target:
                # Take profit
                cash = shares * current_price
                shares = 0
                entry_price = 0
        
        # Calculate current portfolio value
        current_value = cash + (shares * current_price)
        portfolio_value.append({
            'Date': df.index[i],
            'Value': current_value,
            'Cash': cash,
            'Shares': shares,
            'Price': current_price
        })
        
        # Execute trades based on signals
        if df['Signal'].iloc[i] == 1 and shares == 0:  # Buy signal
            shares = cash / current_price
            entry_price = current_price
            cash = 0
        elif df['Signal'].iloc[i] == -1 and shares > 0:  # Sell signal
            cash = shares * current_price
            shares = 0
            entry_price = 0
    
    portfolio_df = pd.DataFrame(portfolio_value)
    return portfolio_df, cash, shares

# Main app
try:
    with st.spinner("Fetching TQQQ data..."):
        data = fetch_data(lookback_period)
    
    if data.empty:
        st.error("Unable to fetch data. Please try again.")
    else:
        # Calculate indicators
        data = calculate_indicators(data)
        
        # Train ML model if enabled
        ml_model = None
        ml_scaler = None
        ml_feature_importance = None
        ml_test_accuracy = None
        
        if use_ml_prediction:
            with st.spinner("Training ML model..."):
                ml_model, ml_scaler, ml_feature_importance, ml_test_accuracy = train_ml_model(data)
        
        data = generate_signals(data, rsi_oversold, rsi_overbought, ml_model, ml_scaler)
        
        # Current metrics
        current_price = data['Close'].iloc[-1]
        current_rsi = data['RSI'].iloc[-1]
        current_bb_lower = data['BB_Lower'].iloc[-1]
        current_bb_upper = data['BB_Upper'].iloc[-1]
        
        st.header("📊 Current Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            st.metric("RSI", f"{current_rsi:.1f}")
        with col3:
            st.metric("Buy Target", f"${current_bb_lower:.2f}", 
                     delta=f"{((current_bb_lower - current_price) / current_price * 100):.1f}%")
        with col4:
            st.metric("Sell Target", f"${current_bb_upper:.2f}",
                     delta=f"{((current_bb_upper - current_price) / current_price * 100):.1f}%")
        
        # ML Prediction Section
        if use_ml_prediction and ml_model is not None:
            st.header("🤖 AI Price Prediction")
            
            # Get current prediction
            ml_prediction, ml_confidence = get_ml_prediction(data, ml_model, ml_scaler)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if ml_prediction == 1:
                    st.success(f"**Prediction: UP** ⬆️")
                else:
                    st.error(f"**Prediction: DOWN** ⬇️")
            with col2:
                confidence_color = "success" if ml_confidence >= 70 else "warning" if ml_confidence >= 60 else "error"
                getattr(st, confidence_color)(f"**Confidence: {ml_confidence:.1f}%**")
            with col3:
                st.metric("Model Accuracy", f"{ml_test_accuracy*100:.1f}%")
            
            st.caption(f"The ML model predicts tomorrow's price direction based on {len(ml_feature_importance)} features.")
            
            # Feature importance
            with st.expander("🔍 What is the AI Learning?"):
                st.write("**Top 10 Most Important Features:**")
                top_features = ml_feature_importance.head(10)
                
                # Create a simple bar chart
                st.bar_chart(top_features.set_index('Feature')['Importance'])
                
                st.caption("""
                **Feature Importance** shows which technical indicators the AI finds most predictive:
                - Higher values = more important for predictions
                - The model learns patterns humans might miss
                - Combines 19 different technical and momentum indicators
                """)
        elif use_ml_prediction:
            st.warning("⚠️ ML model training failed. Need more data or check for errors.")
        
        # Trading recommendation
        st.subheader("🎲 Current Recommendation")
        
        # Check all conditions
        is_oversold = current_rsi < rsi_oversold
        near_lower_band = current_price < current_bb_lower * 1.05
        above_trend = current_price > data['MA_50'].iloc[-1] if use_trend_filter else True
        atr_ok = data['ATR_Pct'].iloc[-1] < data['ATR_Pct'].quantile(0.75) if use_volatility_filter else True
        ma_turning_up = data['MA_Short'].iloc[-1] > data['MA_Short'].iloc[-2] if use_momentum_confirm else True
        
        # ML condition
        ml_says_buy = False
        ml_says_sell = False
        if use_ml_prediction and ml_model is not None:
            ml_prediction, ml_confidence = get_ml_prediction(data, ml_model, ml_scaler)
            ml_says_buy = (ml_prediction == 1) and (ml_confidence >= ml_confidence_threshold)
            ml_says_sell = (ml_prediction == 0) and (ml_confidence >= ml_confidence_threshold)
        
        buy_signal = is_oversold and near_lower_band and above_trend and atr_ok and ma_turning_up
        if use_ml_prediction and ml_model is not None:
            buy_signal = buy_signal and ml_says_buy
        
        is_overbought = current_rsi > rsi_overbought
        near_upper_band = current_price > current_bb_upper * 0.98
        
        sell_signal = (is_overbought and near_upper_band) or (use_ml_prediction and ml_says_sell)
        
        if buy_signal:
            st.success(f"🟢 **BUY SIGNAL** - All conditions met!")
            st.write(f"- Price: ${current_price:.2f} (near support ${current_bb_lower:.2f})")
            st.write(f"- RSI: {current_rsi:.1f} (oversold)")
            if use_trend_filter:
                st.write(f"✓ Above 50-day MA: ${data['MA_50'].iloc[-1]:.2f}")
            if use_volatility_filter:
                st.write(f"✓ Volatility acceptable: {data['ATR_Pct'].iloc[-1]:.1f}%")
            if use_ml_prediction and ml_model is not None:
                st.write(f"✓ AI predicts UP with {ml_confidence:.1f}% confidence")
        elif sell_signal:
            st.warning(f"🔴 **SELL SIGNAL** - Take profits!")
            st.write(f"- Price: ${current_price:.2f}")
            if is_overbought and near_upper_band:
                st.write(f"- RSI: {current_rsi:.1f} (overbought)")
            if use_ml_prediction and ml_says_sell:
                st.write(f"- AI predicts DOWN with {ml_confidence:.1f}% confidence")
        else:
            st.info(f"⚪ **HOLD** - Wait for better setup")
            reasons = []
            if not is_oversold and not is_overbought:
                reasons.append(f"RSI neutral ({current_rsi:.1f})")
            if use_trend_filter and not above_trend:
                reasons.append(f"Below trend (50-MA: ${data['MA_50'].iloc[-1]:.2f})")
            if use_volatility_filter and not atr_ok:
                reasons.append("Volatility too high")
            if use_ml_prediction and ml_model is not None and not ml_says_buy:
                reasons.append(f"AI not confident (needs {ml_confidence_threshold}%, has {ml_confidence:.1f}%)")
            if reasons:
                st.write("Waiting for: " + ", ".join(reasons))
        
        # Price chart
        st.subheader("📈 Price Chart with Indicators")
        chart_data = data[['Close', 'MA_Short', 'MA_Long', 'BB_Upper', 'BB_Lower']].tail(90)
        st.line_chart(chart_data)
        
        # RSI chart
        st.subheader("📉 RSI Indicator")
        rsi_chart = data[['RSI']].tail(90)
        st.line_chart(rsi_chart)
        st.caption(f"Buy when RSI < {rsi_oversold} | Sell when RSI > {rsi_overbought}")
        
        # Backtest results
        st.subheader("🔬 Strategy Backtest")
        trades_df = backtest_strategy(data)
        
        if not trades_df.empty:
            completed_trades = trades_df[trades_df['Type'] == 'SELL']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Trades", f"{len(completed_trades)}")
            with col2:
                avg_return = completed_trades['Return'].mean()
                st.metric("Avg Return per Trade", f"{avg_return:.2f}%")
            with col3:
                win_rate = (completed_trades['Return'] > 0).sum() / len(completed_trades) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            # Trades per year
            days_analyzed = (data.index[-1] - data.index[0]).days
            trades_per_year = (len(completed_trades) / days_analyzed) * 365
            st.info(f"📅 Historical trading frequency: ~{trades_per_year:.0f} trades per year")
            
            # Show recent trades
            with st.expander("View Recent Trades"):
                st.dataframe(trades_df.tail(20).style.format({
                    'Price': '${:.2f}',
                    'Return': '{:.2f}%'
                }))
        else:
            st.warning("No completed trades found in backtest period. Try adjusting parameters.")
        
        # Portfolio simulation
        st.subheader("💰 Portfolio Simulation - 5 Year Performance")
        
        # Fetch 5 year data for simulation
        data_5yr = fetch_data(60)  # 60 months = 5 years
        data_5yr = calculate_indicators(data_5yr)
        
        # Train ML model on 5-year data if enabled
        ml_model_5yr = None
        ml_scaler_5yr = None
        if use_ml_prediction:
            with st.spinner("Training ML model on 5-year data..."):
                ml_model_5yr, ml_scaler_5yr, _, _ = train_ml_model(data_5yr)
        
        data_5yr = generate_signals(data_5yr, rsi_oversold, rsi_overbought, ml_model_5yr, ml_scaler_5yr)
        
        initial_capital = 100000
        portfolio_df, final_cash, final_shares = simulate_portfolio(data_5yr, initial_capital)
        
        # Calculate final values
        final_price = data_5yr['Close'].iloc[-1]
        final_portfolio_value = final_cash + (final_shares * final_price)
        total_return = ((final_portfolio_value - initial_capital) / initial_capital) * 100
        years = len(data_5yr) / 252  # Trading days in a year
        annualized_return = ((final_portfolio_value / initial_capital) ** (1 / years) - 1) * 100
        
        # Calculate max drawdown for strategy
        portfolio_df['Peak'] = portfolio_df['Value'].cummax()
        portfolio_df['Drawdown'] = (portfolio_df['Value'] - portfolio_df['Peak']) / portfolio_df['Peak'] * 100
        max_drawdown = portfolio_df['Drawdown'].min()
        
        # Buy and hold comparison
        buy_hold_shares = initial_capital / data_5yr['Close'].iloc[0]
        buy_hold_value = buy_hold_shares * final_price
        buy_hold_return = ((buy_hold_value - initial_capital) / initial_capital) * 100
        
        # Buy and hold max drawdown
        buy_hold_values = buy_hold_shares * data_5yr['Close']
        buy_hold_peak = buy_hold_values.cummax()
        buy_hold_drawdown = ((buy_hold_values - buy_hold_peak) / buy_hold_peak * 100).min()
        
        st.markdown(f"**Starting Capital:** ${initial_capital:,.0f} invested on {data_5yr.index[0].strftime('%Y-%m-%d')}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Final Value", f"${final_portfolio_value:,.0f}", 
                     delta=f"${final_portfolio_value - initial_capital:,.0f}")
        with col2:
            st.metric("Total Return", f"{total_return:.1f}%")
        with col3:
            st.metric("Annualized Return", f"{annualized_return:.1f}%")
        with col4:
            st.metric("Max Drawdown", f"{max_drawdown:.1f}%",
                     delta=f"{max_drawdown - buy_hold_drawdown:.1f}% vs B&H",
                     delta_color="inverse")
        with col5:
            outperformance = total_return - buy_hold_return
            st.metric("vs Buy & Hold", f"{outperformance:+.1f}%",
                     delta=f"${final_portfolio_value - buy_hold_value:,.0f}")
        
        # Portfolio value chart
        st.subheader("📊 Portfolio Value Over Time")
        portfolio_chart = portfolio_df.set_index('Date')[['Value']]
        
        # Add buy and hold line for comparison
        buy_hold_line = pd.DataFrame({
            'Date': data_5yr.index,
            'Buy & Hold': buy_hold_shares * data_5yr['Close']
        }).set_index('Date')
        
        comparison_df = pd.concat([portfolio_chart, buy_hold_line], axis=1)
        st.line_chart(comparison_df)
        
        st.caption(f"🟦 Strategy Portfolio | 🟧 Buy & Hold Comparison")
        
        # Summary comparison
        col1, col2 = st.columns(2)
        with col1:
            color = "success" if total_return > buy_hold_return else "warning"
            getattr(st, color)(f"""
            **Strategy Results:**
            - Final Value: ${final_portfolio_value:,.0f}
            - Total Return: {total_return:.1f}%
            - Annualized: {annualized_return:.1f}%
            - Max Drawdown: {max_drawdown:.1f}%
            - Current: {f'${final_cash:,.0f} cash' if final_shares == 0 else f'{final_shares:.2f} shares @ ${final_price:.2f}'}
            """)
        with col2:
            st.info(f"""
            **Buy & Hold Results:**
            - Final Value: ${buy_hold_value:,.0f}
            - Total Return: {buy_hold_return:.1f}%
            - Max Drawdown: {buy_hold_drawdown:.1f}%
            - Position: {buy_hold_shares:.2f} shares held
            """)
        
        # Key insight
        if total_return > buy_hold_return:
            st.success(f"🎉 **Strategy Wins!** Outperformed buy-and-hold by {outperformance:.1f}% with {max_drawdown - buy_hold_drawdown:.1f}% better drawdown protection!")
        else:
            st.warning(f"⚠️ Strategy underperformed by {abs(outperformance):.1f}%. Try adjusting the parameters or enabling more filters.")
            st.info("💡 **Tip:** For TQQQ, the trend filter and volatility filter are crucial to avoid prolonged drawdowns.")
            
except Exception as e:
    st.error(f"Error: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.info("""
**Enhanced Strategy Logic:**

**BUY when:**
- RSI oversold (< threshold)
- Price near lower Bollinger Band
- Optional: Above 50-day MA (uptrend)
- Optional: Volatility not extreme
- Optional: Short MA turning up
- **NEW: ML predicts price UP with high confidence**

**SELL when:**
- RSI overbought OR
- Price breaks below 50-day MA with weak RSI OR
- Profit target reached OR
- **NEW: ML predicts price DOWN with high confidence**

**ML Model:**
- Random Forest trained on 19 technical features
- Learns from historical patterns
- Predicts next day direction
- Only trades when confidence > threshold
""")
