# OFX File Processor with Smart Category Prediction

## Overview
`apps/BankStatement.py` is a Streamlit app that processes OFX files (bank statements) and automatically categorizes transactions using a machine learning model.

## Features
- **Parse OFX Files**: Upload multiple OFX files to extract transactions
- **Smart Sub-Category Prediction**: Automatically predicts transaction "Sub" category (Shopping, Groceries, Travel, etc.)
- **Monthly Summaries**: View transaction summaries by category and month
- **Account Breakdown**: Analyze spending by account and Sub-category
- **CSV Export**: Download consolidated transactions and summaries

## Workflow

### Step 1: Prepare Training Data (CSV)
You need a CSV file with past transaction history that includes:
- **Memo** column: transaction description (e.g., "AMAZON MKTPL*ZF9XA5SF3")
- **Sub** column: the category label you've manually assigned (e.g., "Shopping", "Groceries", "Travel")

Example:
```
Memo,Sub
Direct Debit - AMAZON,Shopping
SUPERMERCATO IPERTOSANO,Groceries
OMIO USD BERLIN,Travel
```

### Step 2: Upload Training CSV
1. Launch the app: `streamlit run apps/BankStatement.py`
2. In **Step 1**, upload your training CSV
3. The app will train a model on the Memo→Sub mappings

**Note**: A pre-trained model is already provided at `models/sub_model.joblib` (trained on system transaction history). If you upload a new CSV, it will override this model.

### Step 3: Upload OFX Files
1. In **Step 2**, upload one or more OFX files from your bank
2. The app will parse the files and extract transactions

### Step 4: Review Predictions
The app will:
- Predict the "Sub" category for each transaction using the trained model
- Display consolidated transactions with predicted Sub categories
- Show monthly summaries and account breakdowns
- Allow you to download the data as CSV

## Pre-Trained Model
A pre-trained model is available at `models/sub_model.joblib` (accuracy: **83.47%**).

This model was trained on 1,206 transactions with 19 different Sub-categories:
- **Auto** (vehicle repairs, gas, insurance)
- **Clothing** (apparel, shoes)
- **Dining** (restaurants, food delivery)
- **Entertain** (entertainment, events)
- **Groceries** (supermarkets, food)
- **Housing** (utilities, rent, home services)
- **Shopping** (general retail, Amazon, etc.)
- **Travel** (flights, hotels, parking, transit)
- **Subscription** (Netflix, Spotify, etc.)
- **Tithing** (donations)
- **Venmo** (peer transfers)
- **Medical** (healthcare)
- **Insurance** (insurance payments)
- **Internet** (ISP, phone)
- **Shipping** (postal, delivery fees)
- **Paypal** (PayPal transfers)
- **House** (home maintenance)

## Re-Training the Model
To train a fresh model from your own transaction history:

```bash
cd /home/ksh/Documents/Github/Streamlit-Finance
python3 scripts/sub_predictor.py
```

This script:
1. Loads `20260131 All Statement - Sheet1.csv`
2. Trains a TF-IDF + Logistic Regression model
3. Saves the model to `models/sub_model.joblib`
4. Reports accuracy and per-category performance

## Requirements
- **Python 3.8+**
- **Dependencies**: pandas, scikit-learn, ofxparse, streamlit, joblib

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App
```bash
streamlit run apps/BankStatement.py
```

Then open `http://localhost:8501` in your browser.

## Tips
- **Better Predictions**: Train with at least 100+ samples per Sub-category for best results
- **Memo Consistency**: Ensure merchant names in Memo are consistent (e.g., "AMAZON" vs "Amazon.com" might confuse the model)
- **Category Coverage**: Include diverse transaction types to help the model generalize
