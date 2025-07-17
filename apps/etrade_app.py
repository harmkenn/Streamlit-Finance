import streamlit as st
import configparser
import webbrowser
import json
from rauth import OAuth1Service
from accounts.accounts import Accounts
from market.market import Market

# Load config
config = configparser.ConfigParser()
config.read("config.ini")

st.set_page_config(layout="wide")
st.title("📊 E*TRADE API Streamlit App")

# Mode selector
mode = st.sidebar.radio("Select Mode", ["Sandbox", "Production"])
base_url = config["DEFAULT"]["SANDBOX_BASE_URL"] if mode == "Sandbox" else config["DEFAULT"]["PROD_BASE_URL"]

consumer_key = config["DEFAULT"]["CONSUMER_KEY"]
consumer_secret = config["DEFAULT"]["CONSUMER_SECRET"]

# Auth button
if "session" not in st.session_state:
    if st.button("🔐 Authenticate with E*TRADE"):
        etrade = OAuth1Service(
            name="etrade",
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            request_token_url=base_url + "/oauth/request_token",
            access_token_url=base_url + "/oauth/access_token",
            authorize_url="https://us.etrade.com/e/t/etws/authorize?key={}&token={}",
            base_url=base_url,
        )

        request_token, request_token_secret = etrade.get_request_token(params={"oauth_callback": "oob", "format": "json"})
        auth_url = etrade.authorize_url.format(consumer_key, request_token)
        st.session_state["oauth_temp"] = (etrade, request_token, request_token_secret)

        st.markdown(f"🔗 [Click here to authorize E*TRADE and get a PIN]({auth_url})")
        pin = st.text_input("Enter PIN:")
        if pin:
            etrade, req_token, req_secret = st.session_state["oauth_temp"]
            session = etrade.get_auth_session(req_token, req_secret, params={"oauth_verifier": pin})
            st.session_state["session"] = session
            st.success("✅ Authenticated!")

# Once authenticated, show features
if "session" in st.session_state:
    session = st.session_state["session"]

    option = st.radio("What do you want to do?", ["Quotes", "Account List"])
    
    if option == "Quotes":
        symbol = st.text_input("Enter a stock symbol", "MSTY").upper()
        if st.button("Get Quote"):
            market = Market(session, base_url)
            data = market.get_quote(symbol)
            if data:
                st.json(data)
            else:
                st.error("❌ No quote data found.")
    
    elif option == "Account List":
        accounts = Accounts(session, base_url)
        data = accounts.get_accounts()
        if data:
            st.json(data)
        else:
            st.error("❌ No account data found.")
