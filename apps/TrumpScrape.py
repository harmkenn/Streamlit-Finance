import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import re
#
st.set_page_config(page_title="HTML → CSV Extractor", layout="wide")

st.title("HTML → CSV Extractor for Truth Social Posts")
st.write("Paste HTML from RollCall/Factba.se and extract posts into a CSV.")

st.markdown("---")

html_input = st.text_area(
    "Paste the HTML code here:",
    height=400,
    placeholder="Paste the HTML from the page…"
)

extract_button = st.button("Extract Posts")

def clean_text(t):
    if not t:
        return ""
    return re.sub(r"\s+", " ", t).strip()

def extract_posts_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    posts = []

    # Each post block is inside a div.fb-result or similar
    for block in soup.select("div.fb-result, div.block"):
        # Platform
        platform_el = block.select_one(".fb-result-platform")
        platform = clean_text(platform_el.get_text()) if platform_el else None

        # Timestamp
        date_el = block.select_one(".fb-result-date")
        timestamp = clean_text(date_el.get_text()) if date_el else None

        # URL
        link_el = block.select_one("a[href*='truthsocial.com']")
        url = link_el["href"] if link_el else None

        # Post text (inside x-html or fallback)
        text_el = block.select_one("[x-html], .fb-result-text")
        text = clean_text(text_el.get_text()) if text_el else None

        # Only add if we found meaningful content
        if text or timestamp or url:
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
            st.warning("No posts found in the HTML.")
        else:
            st.success(f"Extracted {len(df)} posts.")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="extracted_posts.csv",
                mime="text/csv"
            )
