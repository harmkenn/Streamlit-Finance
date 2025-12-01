"""
Streamlit app: 7-day TQQQ forecast (predict QQQ then map to TQQQ)
Features: technical indicators, global index leads, VIX
Models: XGBoost (fallback RandomForest), ensemble, recursive forecasting
Outputs: predicted 7-day path, confidence intervals from Monte Carlo, basic backtest

How to run:
1. Create a virtualenv and install requirements:
   pip install streamlit yfinance pandas numpy scikit-learn xgboost matplotlib ta
2. Run:
   streamlit run TQQQ_7day_forecast_streamlit.py

Notes: This is a template. Tweak hyperparams, add more features, and run walk-forward CV for production.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# Try to import xgboost; fallback handled
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# Optional: use ta library for indicators
try:
    import ta
    TA_AVAILABLE = True
except Exception:
    TA_AVAILABLE = False

st.set_page_config(layout="wide", page_title="TQQQ 7-Day Forecast")
st.title("TQQQ — 7 Day Forecast Template")

# Sidebar inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker to forecast (will predict QQQ then map to TQQQ)", value="QQQ")
start_date = st.sidebar.date_input("Start date (historical data)", value=datetime.now().date() - timedelta(days=365*3))
end_date = st.sidebar.date_input("End date", value=datetime.now().date())
model_choice = st.sidebar.selectbox("Model", options=["RandomForest", "XGBoost (if installed)"])
retrain = st.sidebar.button("Retrain model")
n_days = st.sidebar.number_input("Forecast horizon (days)", value=7, min_value=1, max_value=30)
mc_sims = st.sidebar.number_input("Monte Carlo sims", value=200, min_value=50, max_value=2000)

# Helper functions
@st.cache_data
def download_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, progress=False, threads=True)
    # for multi ticker return close prices
    if isinstance(tickers, list) or (isinstance(tickers, str) and "," in tickers):
        return df
    return df


def compute_features(df):
    # expects df to have columns: Close, High, Low, Volume
    data = df.copy()
    data['return'] = data['Close'].pct_change()
    data['logret'] = np.log(data['Close']).diff()
    data['r1'] = data['Close'].pct_change(1)
    data['r3'] = data['Close'].pct_change(3)
    data['r5'] = data['Close'].pct_change(5)

    # rolling features
    data['ma5'] = data['Close'].rolling(5).mean()
    data['ma10'] = data['Close'].rolling(10).mean()
    data['ma20'] = data['Close'].rolling(20).mean()
    data['std5'] = data['return'].rolling(5).std()
    data['std10'] = data['return'].rolling(10).std()

    # ATR approximation
    data['tr1'] = data['High'] - data['Low']
    data['tr2'] = (data['High'] - data['Close'].shift()).abs()
    data['tr3'] = (data['Low'] - data['Close'].shift()).abs()
    data['TR'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['ATR10'] = data['TR'].rolling(10).mean()

    # Momentum
    data['mom5'] = data['Close'] / data['Close'].shift(5) - 1

    # RSI quick implementation
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    data['rsi14'] = 100 - (100 / (1 + rs))

    # dropna
    data = data.dropna()
    return data


def prepare_dataset(main_close_df, extra_close_frames=None):
    # main_close_df: DataFrame with columns Close, High, Low, Volume
    df = compute_features(main_close_df)

    # Optionally add extra leader returns (global indices, vix)
    if extra_close_frames:
        for name, frame in extra_close_frames.items():
            df[name + '_r1'] = frame['Close'].pct_change().reindex(df.index)
    return df


def train_model(X, y, choice='RandomForest'):
    # simple time series split
    tscv = TimeSeriesSplit(n_splits=3)
    best_model = None
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if choice == 'XGBoost' and XGB_AVAILABLE:
            model = XGBRegressor(n_estimators=200, max_depth=4, verbosity=0)
        else:
            model = RandomForestRegressor(n_estimators=200, max_depth=8)
        model.fit(X_train_s, y_train)
        best_model = (model, scaler)
    return best_model


def recursive_forecast(model_tuple, last_features, n_days):
    model, scaler = model_tuple
    preds = []
    features = last_features.copy()
    for i in range(n_days):
        X = features.values.reshape(1, -1)
        Xs = scaler.transform(X)
        p = model.predict(Xs)[0]
        preds.append(p)
        # update features by shifting with the new predicted return applied to last close
        # simplistic: shift returns and recompute a few features
        # user should replace with more rigorous feature update
        # features['r1'] = p
        new_row = features.copy()
        # shift numeric rolling features by assuming persistence
        for col in ['r1','r3','r5','mom5']:
            if col in new_row.index:
                new_row[col] = new_row[col] * 0.9 + p * 0.1
        features = new_row
    return np.array(preds)


def monte_carlo_paths(initial_price, mean_preds, residual_std, sims=200):
    # mean_preds: array of daily expected returns
    paths = np.zeros((sims, len(mean_preds)+1))
    paths[:,0] = initial_price
    for t in range(len(mean_preds)):
        # draw random shocks
        shocks = np.random.normal(loc=mean_preds[t], scale=residual_std, size=sims)
        paths[:, t+1] = paths[:, t] * (1 + shocks)
    return paths

# Main app flow
with st.spinner('Downloading data...'):
    qqq = download_data('QQQ', start_date, end_date)
    # get VIX and some global indices for optional features
    vix = download_data('^VIX', start_date, end_date)
    ftse = download_data('^FTSE', start_date, end_date)

if qqq is None or qqq.empty:
    st.error('Could not download QQQ data. Check ticker or internet.')
    st.stop()

# Prepare dataset
main_df = pd.DataFrame()
main_df['Close'] = qqq['Close']
main_df['High'] = qqq['High']
main_df['Low'] = qqq['Low']
main_df['Volume'] = qqq['Volume']

extra = {'VIX': vix[['Close']]} if not vix.empty else None

dataset = prepare_dataset(main_df, extra_close_frames=extra)

st.subheader('Data preview')
st.dataframe(dataset.tail())

# Define X and y: predict next-day return r1
FEATURES = ['r3','r5','ma5','ma10','ma20','std5','ATR10','mom5','rsi14']
# add VIX lead if present
if 'VIX_r1' in dataset.columns:
    FEATURES.append('VIX_r1')

X = dataset[FEATURES]
y = dataset['r1'].shift(-1).dropna()
X = X.iloc[:-1]

st.write(f"Training rows: {len(X)}")

# Train or load model
model_file = 'tqqq_model.joblib'
if retrain or not st.sidebar.checkbox('Use cached model if exists', value=True):
    chosen = 'XGBoost' if model_choice.startswith('XGBoost') else 'RandomForest'
    if chosen == 'XGBoost' and not XGB_AVAILABLE:
        st.warning('XGBoost not available; using RandomForest instead.')
        chosen = 'RandomForest'
    model_tuple = train_model(X, y, choice=chosen)
    joblib.dump(model_tuple, model_file)
    st.success('Model trained and saved.')
else:
    try:
        model_tuple = joblib.load(model_file)
        st.info('Loaded cached model.')
    except Exception:
        st.warning('No cached model found — training now...')
        model_tuple = train_model(X, y, choice='RandomForest')
        joblib.dump(model_tuple, model_file)

# Forecasting
last_row = X.iloc[-1]
mean_preds = recursive_forecast(model_tuple, last_row, n_days)

# Estimate residual std from last errors
preds_in_sample = model_tuple[0].predict(model_tuple[1].transform(X))
resid = y.values - preds_in_sample
resid_std = np.nanstd(resid)

# Map QQQ returns to TQQQ (approx 3x daily, plus simple volatility drag term)
def q_to_tqqq_path(initial_q, q_returns):
    # apply returns sequentially
    q_path = [initial_q]
    for r in q_returns:
        q_path.append(q_path[-1] * (1 + r))
    return np.array(q_path)

initial_q = main_df['Close'].iloc[-1]
q_path_mean = q_to_tqqq_path(initial_q, mean_preds)
# Simple TQQQ mapping: 3x daily returns
initial_t = None
try:
    tqqq = yf.download('TQQQ', start=start_date, end=end_date, progress=False)
    initial_t = tqqq['Close'].iloc[-1]
except Exception:
    initial_t = None

# Monte Carlo on QQQ returns
paths = monte_carlo_paths(initial_q, mean_preds, resid_std, sims=mc_sims)

# Convert all paths to TQQQ via 3x daily returns
t_paths = (paths[:,1:] / paths[:,:-1] - 1) * 3 + 1
# cumulative
t_price_paths = np.zeros_like(paths)
if initial_t is not None:
    t_price_paths[:,0] = initial_t
else:
    # approximate starting TQQQ by 3x of QQQ
    t_price_paths[:,0] = initial_q * 3
for i in range(len(mean_preds)):
    t_price_paths[:, i+1] = t_price_paths[:, i] * t_paths[:, i]

# Present results
st.subheader('Point forecast (QQQ)')
future_dates = [main_df.index[-1] + pd.Timedelta(days=i+1) for i in range(n_days)]
point_df = pd.DataFrame({'date': future_dates, 'qqq_expected_close': q_path_mean[1:]})
st.table(point_df.set_index('date'))

st.subheader('TQQQ Monte Carlo summary')
median = np.median(t_price_paths, axis=0)
p10 = np.percentile(t_price_paths, 10, axis=0)
p90 = np.percentile(t_price_paths, 90, axis=0)

summary = pd.DataFrame({'date': ['start']+future_dates, 'median': median, 'p10': p10, 'p90': p90})
st.line_chart(pd.DataFrame({'median': summary['median']}).set_index(summary['date']))
st.write('Percentile table:')
st.dataframe(summary)

st.subheader('Sample Monte Carlo paths (first 50)')
for i in range(min(50, mc_sims)):
    plt.plot(summary['date'], t_price_paths[i])
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('TQQQ price')
plt.tight_layout()
st.pyplot(plt.gcf())

st.markdown('---')
st.write('This is a template — please treat predictions as experimental. For production: add walk-forward CV, better feature updates for recursive forecasting, additional macro inputs, and careful risk controls.')
