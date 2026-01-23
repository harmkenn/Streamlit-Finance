import streamlit as st
import pandas as pd
from ofxparse import OfxParser

# Set the page layout to wide
st.set_page_config(layout="wide", page_title="Finance")

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

            # Replace specific Memo items with "Card Pay"
            all_transactions["Memo"] = all_transactions["Memo"].apply(
                lambda memo: "Card Pay" if memo in [
                    "Direct Debit - CHASE CREDIT CRD EPAY",
                    "Withdrawal ACH CITI CARD ONLINE TYPE: PAYMENT ID: CITICTP CO: CI"
                ] else memo
            )

            # Add the "Category" column based on the Memo content
            all_transactions["Category"] = all_transactions["Memo"].apply(
                lambda memo: "Transfer" if "Transfer" in str(memo) else "Expense"
            )

            # Update the "Category" to "Deposit" if the Category is "Expense" and the Amount is positive
            all_transactions["Category"] = all_transactions.apply(
                lambda row: "Deposit" if row["Category"] == "Expense" and row["Amount"] > 0 else row["Category"],
                axis=1
            )

            # Extract the month from the "Date Posted" column
            all_transactions["Month"] = all_transactions["Date Posted"].dt.to_period("M")

            # Calculate total expenses per month
            monthly_expenses = all_transactions[all_transactions["Category"] == "Expense"].groupby("Month")["Amount"].sum().reset_index()
            monthly_expenses.rename(columns={"Amount": "Total Expenses"}, inplace=True)

            st.success("Files processed successfully!")
            
            # Display the consolidated DataFrame
            st.write("### Consolidated Transactions Data")
            st.dataframe(all_transactions)

            # Display the monthly expenses table
            st.write("### Total Expenses Per Month")
            st.dataframe(monthly_expenses)

            # Option to download the consolidated DataFrame as CSV
            csv = all_transactions.to_csv(index=False)
            st.download_button(
                label="Download Consolidated CSV",
                data=csv,
                file_name="consolidated_transactions.csv",
                mime="text/csv"
            )
        else:
            st.warning("No transactions found in the uploaded files.")

if __name__ == "__main__":
    main()
