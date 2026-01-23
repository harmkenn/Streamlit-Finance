import streamlit as st
import pandas as pd
from ofxparse import OfxParser

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
    st.title("OFX File Processor")
    st.write("Upload one or more OFX files to process them into a Pandas DataFrame.")

    # File uploader for multiple files
    uploaded_files = st.file_uploader("Upload OFX Files", type=["ofx"], accept_multiple_files=True)

    if uploaded_files:
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

            # Calculate total amounts per category per month
            monthly_summary = all_transactions.groupby(["Month", "Category"])["Amount"].sum().reset_index()
            monthly_summary = monthly_summary.pivot(index="Month", columns="Category", values="Amount").fillna(0)
            monthly_summary["Total Expenses"] = monthly_summary.get("Expense", 0)  # Add a column for total expenses

            st.success("Files processed successfully!")
            
            # Add blank "Sub" column to the left of "Category"
            all_transactions.insert(all_transactions.columns.get_loc("Category"), "Sub", "")
            
            # Reorder columns to move Amount to the end
            cols = all_transactions.columns.tolist()
            cols.remove("Amount")
            cols.append("Amount")
            all_transactions = all_transactions[cols]
            
            # Display the consolidated DataFrame
            st.write("### Consolidated Transactions Data")
            st.dataframe(all_transactions)

            # Display the monthly summary table
            st.write("### Monthly Summary by Category")
            st.dataframe(monthly_summary)

            # Display expenses by account and month
            st.write("### Expenses by Account and Month")
            expenses_data = all_transactions[all_transactions["Category"] == "Expense"]
            if not expenses_data.empty:
                expenses_by_account_month = expenses_data.groupby(["Account ID", "Month"])["Amount"].sum().reset_index()
                expenses_pivot = expenses_by_account_month.pivot(index="Month", columns="Account ID", values="Amount").fillna(0)
                st.dataframe(expenses_pivot)
            else:
                st.info("No expense transactions found.")

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

