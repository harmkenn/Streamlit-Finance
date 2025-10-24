# Streamlit app: ETF 1-month target price prediction using LSTM + Technical Indicators
# File: tqqq_etf_target_lstm.py
# Save and run: pip install -r requirements.txt
# then: streamlit run tqqq_etf_target_lstm.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
import os
import joblib

st.set_page_config(layout="wide", page_title="ETF 1-Month Target (LSTM)")
st.title("ETF 1-Month Target Price — LSTM with Technical Indicators")

# ------------------ App configuration ------------------
DEFAULT_TICKERS = ["TQQQ", "MAIN", "SCHG"]
LOOKBACK = st.sidebar.number_input("Lookback (trading days used as input)", min_value=5, max_value=60, value=21)
PRED_DAYS = st.sidebar.number_input("Prediction horizon (trading days ahead)", min_value=5, max_value=60, value=21)
EPOCHS = st.sidebar.number_input("Training epochs", min_value=1, max_value=200, value=40)
BATCH = st.sidebar.number_input("Batch size", min_value=1, max_value=256, value=32)
LR = st.sidebar.number_input("Validation split (fraction)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

st.sidebar.markdown("---")

tickers_input = st.sidebar.text_input("Tickers (comma separated)", value=",".join(DEFAULT_TICKERS))
TICKERS = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

use_cached = st.sidebar.checkbox("Use cached models if available", value=True)

# Where to store models/scalers
MODEL_DIR = "models_etf_lstm"
SCALER_DIR = "scalers_etf_lstm"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# ------------------ Utility functions ------------------

def get_history(ticker, period_years=5):
    """Get historical adjusted close (and other columns) for ticker"""
    end = datetime.today()
    start = end - timedelta(days=int(365 * period_years))
    df = yf.download(ticker, start=start.date(), end=end.date(), progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    df = df.dropna()
    return df


def add_technical_indicators(df):
    """Add a set of technical indicators to dataframe. Operates in place and returns df copy."""
    df = df.copy()
    close = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']

    # Moving averages
    df['SMA_7'] = close.rolling(window=7, min_periods=1).mean()
    df['SMA_21'] = close.rolling(window=21, min_periods=1).mean()
    df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = close.ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(span=14).mean()
    roll_down = down.ewm(span=14).mean()
    rs = roll_up / roll_down
    df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))

    # Bollinger Bands
    df['BB_mid'] = close.rolling(window=20, min_periods=1).mean()
    df['BB_std'] = close.rolling(window=20, min_periods=1).std()
    df['BB_up'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_dn'] = df['BB_mid'] - 2 * df['BB_std']

    # Returns
    df['Return_1'] = close.pct_change()
    df['Return_5'] = close.pct_change(5)

    # Fill NaN
    df = df.fillna(method='bfill').fillna(method='ffill')

    return df


def create_sequences(data, lookback, pred_days):
    """Create X, y sequences. X shape: (n_samples, lookback, n_features). y shape: (n_samples, pred_days)
    We predict future Adjusted Close as sequence of pred_days ahead.
    """
    X, y = [], []
    n = len(data)
    for i in range(n - lookback - pred_days + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+pred_days, 0])  # assuming column 0 is target (Adj Close)
    X = np.array(X)
    y = np.array(y)
    return X, y


def build_lstm_model(input_shape, pred_days):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(pred_days))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_for_ticker(ticker, lookback, pred_days, epochs, batch, val_split, use_cached=True):
    """Fetch data, compute indicators, create sequences, scale, train LSTM. Returns model, scaler, last_df (for plotting)"""
    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.h5")
    scaler_path = os.path.join(SCALER_DIR, f"{ticker}_scaler.gz")

    # If cached and exists
    if use_cached and os.path.exists(model_path) and os.path.exists(scaler_path):
        st.info(f"Loading cached model and scaler for {ticker}")
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        df = get_history(ticker, period_years=5)
        df = add_technical_indicators(df)
        return model, scaler, df

    st.info(f"Fetching data for {ticker} (this may take a few seconds)")
    df = get_history(ticker, period_years=5)
    df = add_technical_indicators(df)

    # Create features matrix
    # We'll predict 'Adj Close' if available else 'Close'
    target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    feature_cols = [target_col, 'SMA_7', 'SMA_21', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'RSI_14', 'BB_up', 'BB_dn', 'Return_1', 'Return_5']
    data = df[feature_cols].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = create_sequences(data_scaled, lookback, pred_days)
    st.write(f"Created {X.shape[0]} training samples for {ticker}.")

    # Build model
    model = build_lstm_model((lookback, data.shape[1]), pred_days)

    early = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    with st.spinner(f"Training LSTM for {ticker} ..."):
        history = model.fit(X, y, epochs=epochs, batch_size=batch, validation_split=val_split, callbacks=[early], verbose=0)

    # Save model and scaler
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    st.success(f"Trained and saved model for {ticker}")
    return model, scaler, df


def predict_future_prices(model, scaler, df, lookback, pred_days):
    """Use the last `lookback` days from df to predict pred_days ahead. Returns predicted prices (unscaled) and index dates."""
    target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    feature_cols = [target_col, 'SMA_7', 'SMA_21', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'RSI_14', 'BB_up', 'BB_dn', 'Return_1', 'Return_5']
    recent = df[feature_cols].tail(lookback).values
    recent_scaled = scaler.transform(recent)
    recent_scaled = np.expand_dims(recent_scaled, axis=0)  # shape (1, lookback, nfeatures)

    preds_scaled = model.predict(recent_scaled)[0]  # shape (pred_days,)

    # We predicted scaled target values relative to first column scaling; to invert, we need to reconstruct full feature rows
    # Simpler: We'll create dummy rows where only the first column is preds_scaled and inverse transform via scaler's min/max.
    # But because MinMaxScaler works across features, we can invert only the first column using scaler min_/scale_ values.

    # Invert scaling for the first column only
    # MinMaxScaler: X_std = (X - min) / (max - min); X_scaled = X_std * (max-min) + min? Actually scaler stores min_ and scale_
    data_min = scaler.min_[0]
    data_scale = scaler.scale_[0]
    # scaler.scale_ = (max - min) / (data_range) ??? scikit-learn stores scale_ = data_range / (max-min)?
    # To invert: original = (scaled - min_)/scale_
    # double-check: scikit-learn: X_std = (X - min) * scale_ ; where scale_ = 1/(data_max - data_min)
    # Actually safer to use inverse_transform on constructed arrays

    # Build placeholder array
    n_features = recent.shape[1]
    inv_scaled = []
    for s in preds_scaled:
        row = np.zeros(n_features)
        row[0] = s
        # For other features, fill with last available row scaled values (so inverse_transform doesn't produce nonsense)
        last_scaled_row = scaler.transform(recent[-1:].reshape(1, -1))[0]
        row[1:] = last_scaled_row[1:]
        inv_scaled.append(row)
    inv_scaled = np.array(inv_scaled)
    inv = scaler.inverse_transform(inv_scaled)
    preds = inv[:, 0]

    # Build date index for predicted trading days by extending business days from last date
    last_date = df.index[-1]
    dates = []
    curr = last_date
    while len(dates) < pred_days:
        curr = curr + pd.Timedelta(days=1)
        if curr.weekday() < 5:  # Mon-Fri
            dates.append(curr)
    dates = pd.to_datetime(dates)
    return pd.Series(preds, index=dates)

# ------------------ Streamlit UI ------------------

st.markdown("### Configuration")
st.write("Tickers:", TICKERS)
st.write(f"Using lookback = {LOOKBACK} trading days → predict next {PRED_DAYS} trading days")

if st.button("Train / Load Models for All Tickers"):
    results = {}
    progress = st.progress(0)
    total = len(TICKERS)
    i = 0
    for t in TICKERS:
        try:
            model, scaler, df = train_for_ticker(t, LOOKBACK, PRED_DAYS, EPOCHS, BATCH, LR, use_cached)
            results[t] = (model, scaler, df)
        except Exception as e:
            st.error(f"Failed for {t}: {e}")
        i += 1
        progress.progress(int(i/total * 100))
    st.session_state['models'] = results
    st.success("Models ready")


# If models in session state, allow predictions
if 'models' in st.session_state and st.session_state['models']:
    st.markdown("---")
    st.markdown("### Predictions and Plots")
    cols = st.columns(len(TICKERS))
    for idx, t in enumerate(TICKERS):
        with cols[idx]:
            st.subheader(t)
            if t not in st.session_state['models']:
                st.warning("Model not trained/loaded for this ticker")
                continue
            model, scaler, df = st.session_state['models'][t]

            # Predict
            preds = predict_future_prices(model, scaler, df, LOOKBACK, PRED_DAYS)

            # Plot historical last 60 days and predicted
            hist = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
            hist = hist.tail(60)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode='lines', name='Recent Close'))
            fig.add_trace(go.Scatter(x=preds.index, y=preds.values, mode='lines+markers', name='Predicted Close'))
            fig.update_layout(title=f"{t}: Recent + {PRED_DAYS}-day Prediction", xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig, use_container_width=True)

            # Show predicted summary
            last_close = hist.values[-1]
            pred_price_1m = preds.values[-1]
            pct = (pred_price_1m - last_close) / last_close * 100
            st.metric(label=f"Target in ~{PRED_DAYS} trading days", value=f"{pred_price_1m:.2f}", delta=f"{pct:.2f}%")

            # Show table of predicted values
            df_preds = preds.reset_index()
            df_preds.columns = ['Date', 'Predicted_Close']
            st.dataframe(df_preds)

# Allow single quick prediction without training (load cached if available)
st.markdown("---")
st.markdown("### Quick predict for a single ticker")
quick_ticker = st.text_input("Ticker to predict (single)", value="TQQQ")
if st.button("Quick predict"):
    try:
        model, scaler, df = train_for_ticker(quick_ticker.upper(), LOOKBACK, PRED_DAYS, epochs=1, batch=1, val_split=LR, use_cached=use_cached)
        preds = predict_future_prices(model, scaler, df, LOOKBACK, PRED_DAYS)
        fig = go.Figure()
        hist = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        fig.add_trace(go.Scatter(x=hist.tail(60).index, y=hist.tail(60).values, mode='lines', name='Recent Close'))
        fig.add_trace(go.Scatter(x=preds.index, y=preds.values, mode='lines+markers', name='Predicted Close'))
        fig.update_layout(title=f"{quick_ticker.upper()}: Recent + {PRED_DAYS}-day Prediction", xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)

        last_close = hist.values[-1]
        pred_price_1m = preds.values[-1]
        pct = (pred_price_1m - last_close) / last_close * 100
        st.metric(label=f"Target in ~{PRED_DAYS} trading days", value=f"{pred_price_1m:.2f}", delta=f"{pct:.2f}%")

    except Exception as e:
        st.error(f"Quick predict failed: {e}")

st.markdown("---")
st.markdown("### Notes and next steps")
st.write("This demo builds one LSTM per ticker using historical data and typical technical indicators. Training an LSTM can be slow on CPU — consider using a GPU or reduce epochs/batch size for quick experiments.")
st.write("The approach uses a sliding-window training set built from historical data (default 5 years) where each sample uses `lookback` days to predict `pred_days` ahead.")

st.markdown("**Requirements (suggested)**")
st.code("""
pip install streamlit yfinance pandas numpy scikit-learn tensorflow plotly joblib
""")

st.markdown("If you'd like, I can:")
st.write("- Reduce the model size / change architecture for faster CPU training")
st.write("- Use a walk-forward evaluation and show historical backtest error")
st.write("- Export predictions to CSV or Excel automatically")

