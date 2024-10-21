import streamlit as st
import importlib

# Set the page layout to wide
st.set_page_config(layout="wide", page_title=f"Finance")

# Set up the Streamlit interface
st.title("Finance")

def main():
    # Define a dictionary to map radio button values to .py file names
    sub_app_map = {
        "Buy Sell or Hold": "BuySellHold.py",
        "Sub App 2": "subapp2.py"
    }

    # Use sidebar for radio button navigation
    with st.sidebar:
        selected_app = st.radio("Choose a sub-app", list(sub_app_map.keys()))

    # Load the selected sub-app from the "SubApps" folder
    sub_app_module_name = f"SubApps.{sub_app_map[selected_app]}"
    try:
        sub_app_module = importlib.import_module(sub_app_module_name)
        sub_app_module.run_sub_app()
    except ImportError:
        st.error(f"Sub-app '{selected_app}' not found.")

if __name__ == "__main__":
    main()