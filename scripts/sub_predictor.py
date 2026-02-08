import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os


CSV_PATH = "20260131 All Statement - Sheet1.csv"


def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows with columns: {list(df.columns)}")
    return df


def prepare_dataframe(df):
    # Keep rows with a non-empty Sub for supervised training
    df = df.copy()
    df['Sub'] = df['Sub'].fillna("")
    df['Name'] = df['Name'].fillna("")
    df['Memo'] = df['Memo'].fillna("")
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0.0)

    df['text'] = (df['Name'] + ' ' + df['Memo']).str.strip().str.lower()
    df['abs_amount'] = df['Amount'].abs()
    df['sign'] = np.where(df['Amount'] < 0, 'debit', 'credit')

    # Filter rows with a labeled Sub
    labeled = df[df['Sub'].str.strip() != ""].reset_index(drop=True)
    print(f"Using {len(labeled):,} labeled rows (for training/evaluation). Unique Subs: {labeled['Sub'].nunique()}")
    return labeled


def baseline_lookup(train, test):
    # Exact (Name,Memo) -> most common Sub
    mapping = train.groupby(['Name','Memo'])['Sub'].agg(lambda x: x.value_counts().idxmax()).to_dict()

    def predict_row(r):
        key = (r['Name'], r['Memo'])
        return mapping.get(key, None)

    preds = test.apply(predict_row, axis=1)
    # fallback to most common Sub from training
    most_common = train['Sub'].mode().iloc[0]
    preds = preds.fillna(most_common)
    return preds


def train_tfidf_model(train, test):
    text_col = 'text'
    cat_cols = ['Account Type', 'Transaction Type', 'Category', 'Month', 'sign']
    num_cols = ['abs_amount']

    # Column transformer
    preprocessor = ColumnTransformer([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2)), text_col),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('scale', StandardScaler(), num_cols)
    ], remainder='drop')

    clf = LogisticRegression(max_iter=1000)

    pipeline = Pipeline([
        ('pre', preprocessor),
        ('clf', clf)
    ])

    X_train = train[[text_col] + cat_cols + num_cols]
    y_train = train['Sub']

    X_test = test[[text_col] + cat_cols + num_cols]
    y_test = test['Sub']

    # Fit using DataFrame inputs so ColumnTransformer can select columns by name
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)

    return pipeline, preds, probs


def main():
    df = load_data(CSV_PATH)
    labeled = prepare_dataframe(df)

    if labeled.empty:
        print("No labeled rows found. Exiting.")
        return

    # Collapse very-rare labels into 'OTHER' so stratified split is possible
    label_counts = labeled['Sub'].value_counts()
    rare = label_counts[label_counts < 2].index.tolist()
    if rare:
        print(f"Collapsing {len(rare)} rare Sub labels into 'OTHER' to allow stratified split")
        labeled['Sub'] = labeled['Sub'].apply(lambda x: 'OTHER' if x in rare else x)

    # Train/test split (attempt stratify by Sub; fall back if still not possible)
    try:
        train, test = train_test_split(labeled, test_size=0.2, random_state=42, stratify=labeled['Sub'])
    except ValueError:
        print("Stratified split not possible; falling back to random split without stratify")
        train, test = train_test_split(labeled, test_size=0.2, random_state=42)

    print("\n--- Baseline (exact lookup) ---")
    base_preds = baseline_lookup(train, test)
    print(f"Baseline accuracy: {accuracy_score(test['Sub'], base_preds):.4f}")

    print("\n--- TF-IDF + Logistic Regression ---")
    try:
        model, preds, probs = train_tfidf_model(train, test)
        print(f"Model accuracy: {accuracy_score(test['Sub'], preds):.4f}")
        print("\nClassification report:\n")
        print(classification_report(test['Sub'], preds, zero_division=0))

        # Show sample predictions
        sample = test.sample(n=min(10, len(test)), random_state=1).copy()
        sample['baseline_pred'] = baseline_lookup(train, sample)
        sample_features = sample[['text','Account Type','Transaction Type','Category','Month','sign','abs_amount']]
        sample['model_pred'] = model.predict(sample_features)
        sample['model_prob'] = [max(p) for p in model.predict_proba(sample_features)]

        print("\nSample predictions (Name | Memo | True Sub | Baseline | Model | Prob):")
        for _, r in sample.iterrows():
            print(f"- {r['Name'][:40]} | {r['Memo'][:40]} | {r['Sub']} | {r['baseline_pred']} | {r['model_pred']} | {r['model_prob']:.2f}")

        # Save trained pipeline
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'sub_model.joblib')
        joblib.dump(model, model_path)
        print(f"Saved trained model to: {model_path}")

    except Exception as e:
        print(f"Model training failed: {e}")


if __name__ == '__main__':
    main()
