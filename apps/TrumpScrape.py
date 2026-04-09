import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import quopri
import re
#1.7
st.set_page_config(page_title="Factba.se / Truth Social HTML → CSV", layout="wide")

st.title("Factba.se / Truth Social HTML → CSV")
st.write("Paste rendered HTML (from Elements panel or saved page) to extract posts into CSV.")

st.markdown("---")

html_input = st.text_area(
    "Paste the HTML code here:",
    height=400,
    placeholder="Paste the HTML that contains the posts…"
)

extract_button = st.button("Extract Posts")


def clean_text(t):
    if not t:
        return ""
    return re.sub(r"\s+", " ", t).strip()


def decode_html(raw):
    try:
        return quopri.decodestring(raw).decode("utf-8", errors="ignore")
    except:
        return raw


def extract_timestamp(block):
    # Factba.se timestamp format
    ts_el = block.select_one("span.hidden.md\\:inline")
    if ts_el and ts_el.get_text(strip=True):
        return clean_text(ts_el.get_text())

    # Fallback: any span containing @ and ET
    for span in block.find_all("span"):
        txt = clean_text(span.get_text())
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
    decoded = decode_html(html)
    soup = BeautifulSoup(decoded, "html.parser")

    posts = []

    # Each post is inside a <div class="block ...">
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


if extract_button:
    if not html_input.strip():
        st.error("Please paste HTML first.")
    else:
        df = extract_posts_from_html(html_input)

        if df.empty:
            st.warning("No posts found. Make sure you pasted *rendered* HTML from the Elements panel.")
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
