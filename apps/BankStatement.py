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
    st.write("Upload an OFX file to process it into a Pandas DataFrame.")

    uploaded_file = st.file_uploader("Upload OFX File", type=["ofx"])
    if uploaded_file is not None:
        try:
            # Parse the uploaded OFX file
            df = parse_ofx(uploaded_file)
            st.success("File processed successfully!")
            
            # Display the DataFrame
            st.write("### Transactions Data")
            st.dataframe(df)

            # Option to download the DataFrame as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="transactions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
