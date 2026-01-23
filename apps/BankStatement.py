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

            # Add the "Category" column based on the Memo content
            all_transactions["Category"] = all_transactions["Memo"].apply(
                lambda memo: "Transfer" if "Transfer" in str(memo) else "Expense"
            )

            st.success("Files processed successfully!")
            
            # Display the DataFrame
            st.write("### Consolidated Transactions Data")
            st.dataframe(all_transactions)

            # Option to download the DataFrame as CSV
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
