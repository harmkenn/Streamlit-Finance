import streamlit as st
import pandas as pd
import numpy as np
from ofxparse import OfxParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import os

# Function to parse OFX file and extract transactions
def parse_ofx(file):
    ofx = OfxParser.parse(file)
    transactions = []
    for account in ofx.accounts:
        for transaction in account.statement.transactions:
            transactions.append({
                "Account ID": account.number,
                "Account Type": account.account_type,
                "Transaction Type": transaction.type,
                "Date Posted": transaction.date,
                "Amount": transaction.amount,
                "FITID": transaction.id,
                "Name": transaction.payee,
                "Memo": transaction.memo
            })
    return pd.DataFrame(transactions)

# Streamlit app
def main():
    st.title("📊 OFX File Processor with Smart Category Prediction")
    st.write("Upload a training CSV and OFX files to automatically categorize transactions by Sub.")

    # Step 1: Training data section
    st.header("Step 1: Upload Training Data (CSV)")
    st.write("Upload a CSV with 'Memo' and 'Sub' columns. The app will learn to predict 'Sub' from transaction details.")
    
    training_file = st.file_uploader("Upload History CSV (with 'Memo' and 'Sub' columns)", type=["csv"], key="training_csv")
    
    model = None
    model_source = None
    
    # Try to load pre-trained model first
    model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'models', 'sub_model.joblib'))
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            model_source = "saved"
            st.info(f"✓ Loaded pre-trained model from: `{os.path.relpath(model_path)}`")
        except Exception as e:
            st.warning(f"⚠️ Pre-trained model found but failed to load: {e}")
    
    # If training file uploaded, train a new model
    if training_file:
        try:
            train_df = pd.read_csv(training_file)
            if "Memo" in train_df.columns and "Sub" in train_df.columns:
                train_df["Memo"] = train_df["Memo"].fillna("").astype(str)
                train_df = train_df.dropna(subset=["Sub"])
                
                # Train simple Memo->Sub model
                simple_model = make_pipeline(CountVectorizer(), MultinomialNB())
                simple_model.fit(train_df["Memo"], train_df["Sub"])
                model = simple_model
                model_source = "uploaded"
                
                st.success(f"✓ Trained model on {len(train_df)} records from uploaded CSV.")
            else:
                st.error("❌ CSV must contain 'Memo' and 'Sub' columns.")
        except Exception as e:
            st.error(f"❌ Error training model: {e}")

    # Step 2: OFX file upload
    st.header("Step 2: Upload OFX Files")
    st.write("Upload one or more OFX files. Transactions will be categorized using the trained model.")
    
    uploaded_files = st.file_uploader("Upload OFX Files", type=["ofx"], accept_multiple_files=True, key="ofx_files")

    if uploaded_files:
        # Check if a model is available
        if model is None:
            st.error("❌ No model available. Please upload a training CSV in Step 1 or ensure a pre-trained model is available.")
            st.stop()
        
        st.success(f"✓ Using {model_source} model to predict Sub categories.")
        
        all_transactions = pd.DataFrame()  # Initialize an empty DataFrame

        for uploaded_file in uploaded_files:
            try:
                # Parse each uploaded OFX file
                df = parse_ofx(uploaded_file)
                all_transactions = pd.concat([all_transactions, df], ignore_index=True)
            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {e}")

        if not all_transactions.empty:
            # Remove duplicate transactions based on FITID
            all_transactions = all_transactions.drop_duplicates(subset=["FITID"])

            # Add the "Category" column based on the Memo content
            all_transactions["Category"] = all_transactions["Memo"].apply(
                lambda memo: "Card Pay" if any(keyword in str(memo) for keyword in [
                    "Direct Debit - CHASE CREDIT CRD EPAY",
                    "Withdrawal ACH CITI CARD ONLINE TYPE: PAYMENT ID: CITICTP CO: CI"
                ]) else ("Housing" if any(keyword in str(memo) for keyword in ["Wise", "AGSM"]) else (
                    "Transfer" if "Transfer" in str(memo) else "Expense"))
            )

            # Update the "Category" to "Deposit" if the Category is "Expense" and the Amount is positive
            all_transactions["Category"] = all_transactions.apply(
                lambda row: "Deposit" if row["Category"] == "Expense" and row["Amount"] > 0 else row["Category"],
                axis=1
            )

            # Extract the month from the "Date Posted" column
            all_transactions["Month"] = all_transactions["Date Posted"].dt.to_period("M")

            # Predict "Sub" category using the trained model
            st.header("Step 3: Review Predictions")
            
            with st.spinner("Predicting Sub categories..."):
                try:
                    # Prepare memo for prediction
                    all_transactions['Memo'] = all_transactions['Memo'].fillna("").astype(str)
                    # Use the model to predict Sub from Memo
                    sub_values = model.predict(all_transactions['Memo'])
                    all_transactions.insert(all_transactions.columns.get_loc("Category"), "Sub", sub_values)
                    st.success(f"✓ Predicted Sub category for {len(all_transactions)} transactions.")
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    st.stop()

            # Reorder columns to move Amount to the end
            cols = all_transactions.columns.tolist()
            if "Amount" in cols:
                cols.remove("Amount")
                cols.append("Amount")
                all_transactions = all_transactions[cols]
            
            # Display the consolidated DataFrame
            st.subheader("Consolidated Transactions Data")
            st.dataframe(all_transactions, use_container_width=True)

            # Calculate total amounts per category per month
            monthly_summary = all_transactions.groupby(["Month", "Category"])["Amount"].sum().reset_index()
            monthly_summary = monthly_summary.pivot(index="Month", columns="Category", values="Amount").fillna(0)
            monthly_summary["Total Expenses"] = monthly_summary.get("Expense", 0)  # Add a column for total expenses

            # Display the monthly summary table
            st.subheader("Monthly Summary by Category")
            st.dataframe(monthly_summary)

            # Display expenses by account and month
            st.subheader("Expenses by Account and Month")
            expenses_data = all_transactions[all_transactions["Category"] == "Expense"]
            if not expenses_data.empty:
                expenses_by_account_month = expenses_data.groupby(["Account ID", "Month"])["Amount"].sum().reset_index()
                expenses_pivot = expenses_by_account_month.pivot(index="Month", columns="Account ID", values="Amount").fillna(0)
                st.dataframe(expenses_pivot)
            else:
                st.info("No expense transactions found.")

            # Sub-category Breakdown by Account
            st.subheader("Sub-category Breakdown by Account")
            account_list = all_transactions["Account ID"].unique()
            selected_accounts = st.multiselect("Select Account(s)", account_list, default=account_list)
            category_list = all_transactions["Category"].unique()
            selected_categories = st.multiselect("Select Category(s)", category_list, default=category_list)

            if selected_accounts and selected_categories:
                account_filtered = all_transactions[all_transactions["Account ID"].isin(selected_accounts)]
                account_filtered = account_filtered[account_filtered["Category"].isin(selected_categories)]
                if not account_filtered.empty:
                    sub_pivot = account_filtered.pivot_table(
                        index="Month", columns="Sub", values="Amount", aggfunc="sum", fill_value=0
                    )
                    sub_pivot["Total"] = sub_pivot.sum(axis=1)
                    st.dataframe(sub_pivot)

            # Option to download the consolidated DataFrame as CSV
            csv = all_transactions.to_csv(index=False)
            st.download_button(
                label="Download Consolidated CSV",
                data=csv,
                file_name="consolidated_transactions.csv",
                mime="text/csv"
            )

            # Option to download the monthly summary table as CSV
            summary_csv = monthly_summary.to_csv(index=True)
            st.download_button(
                label="Download Monthly Summary CSV",
                data=summary_csv,
                file_name="monthly_summary.csv",
                mime="text/csv"
            )
        else:
            st.warning("No transactions found in the uploaded files.")

main()
