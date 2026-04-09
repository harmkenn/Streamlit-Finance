import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import quopri
import re
#1.9
st.set_page_config(page_title="Factba.se HTML → CSV", layout="wide")

st.title("Factba.se / Truth Social HTML → CSV")
st.write("Upload a **Webpage, Complete (.htm)** file saved from Chrome to extract posts into CSV.")

uploaded_file = st.file_uploader("Upload the .htm file", type=["htm", "html"])

def clean_text(t):
    if not t:
        return ""
    return re.sub(r"\s+", " ", t).strip()

def decode_html(raw_bytes):
    """Decode quoted-printable if present."""
    try:
        raw = raw_bytes.decode("utf-8", errors="ignore")
        return quopri.decodestring(raw).decode("utf-8", errors="ignore")
    except:
        return raw_bytes.decode("utf-8", errors="ignore")

def extract_timestamp(block):
    # Match both classes: "hidden" and "md:inline"
    ts_el = block.find("span", class_=["hidden", "md:inline"])
    if ts_el and ts_el.get_text(strip=True):
        return ts_el.get_text(strip=True)

    # Fallback: any span containing @ and ET
    for span in block.find_all("span"):
        txt = span.get_text(strip=True)
        if "@" in txt and "ET" in txt:
            return txt

    return None


def extract_post_text(block):
    html_block = block.select_one('[x-html="item.social.post_html"]')
    if html_block:
        txt = html_block.get_text(separator=" ", strip=True)
        if txt:
            return clean_text(txt)
    return None

def extract_url(block):
    link_el = block.select_one("a[href*='truthsocial.com']")
    if link_el and link_el.has_attr("href"):
        return link_el["href"]
    return None

def extract_platform(block):
    if block.select_one('img[alt="Truth Social icon"]'):
        return "Truth Social"
    if block.select_one(".fa-x-twitter"):
        return "Twitter"
    return None

def extract_posts_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    posts = []

    # Each post is inside a <div class="block mb-8 ...">
    for block in soup.select("div.block.mb-8"):
        timestamp = extract_timestamp(block)
        text = extract_post_text(block)
        url = extract_url(block)
        platform = extract_platform(block)

        if timestamp or text or url:
            posts.append({
                "timestamp": timestamp,
                "platform": platform,
                "url": url,
                "text": text
            })

    return pd.DataFrame(posts)

if uploaded_file:
    raw_bytes = uploaded_file.read()
    decoded_html = decode_html(raw_bytes)

    df = extract_posts_from_html(decoded_html)

    if df.empty:
        st.warning("No posts found. Make sure you uploaded a **Webpage, Complete (.htm)** file, not an MHTML file.")
    else:
        st.success(f"Extracted {len(df)} posts.")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="factbase_truth_posts.csv",
            mime="text/csv",
        )
