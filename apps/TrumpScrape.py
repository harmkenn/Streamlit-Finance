import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import quopri
import re
###
st.set_page_config(page_title="HTML → CSV Extractor", layout="wide")

st.title("HTML → CSV Extractor for Factba.se / Truth Social")
st.write("Paste HTML from RollCall/Factba.se (even encoded) and extract posts into a CSV.")

st.markdown("---")

html_input = st.text_area(
    "Paste the HTML code here:",
    height=400,
    placeholder="Paste the HTML from the page or email source…"
)

extract_button = st.button("Extract Posts")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def clean_text(t):
    if not t:
        return ""
    return re.sub(r"\s+", " ", t).strip()


def decode_html(raw):
    """Decode quoted-printable HTML like class=3D, =E2=80=9C, =\n, etc."""
    try:
        decoded = quopri.decodestring(raw).decode("utf-8", errors="ignore")
        return decoded
    except Exception:
        return raw


def extract_timestamp(block):
    """
    Extract timestamps like:
    <span class="hidden md:inline">April 8, 2026 @ 11:46 PM ET</span>
    """
    # Tailwind class with colon must be escaped
    ts_el = block.select_one("span.hidden.md\\:inline")
    if ts_el:
        return clean_text(ts_el.get_text())

    # Fallback: any span containing "@" and "ET"
    for span in block.find_all("span"):
        txt = clean_text(span.get_text())
        if "@" in txt and "ET" in txt:
            return txt

    return None


def extract_post_text(block):
    """
    Extract the actual Truth Social post text.
    Factba.se embeds it inside:
    <div x-html="item.social.post_html"><p>...</p></div>
    """
    html_block = block.select_one('[x-html="item.social.post_html"]')
    if html_block:
        txt = html_block.get_text(separator=" ", strip=True)
        if txt:
            return clean_text(txt)

    # Fallback for older Factba.se
    text_el = block.select_one(".fb-result-text")
    if text_el:
        return clean_text(text_el.get_text(separator=" ", strip=True))

    return None


def extract_posts_from_html(html):
    decoded = decode_html(html)
    soup = BeautifulSoup(decoded, "html.parser")

    posts = []

    # Look for any div that contains the post HTML
    for html_block in soup.select('[x-html="item.social.post_html"]'):
        # Walk up a few levels to get the full post container
        block = html_block
        for _ in range(3):
            if block.parent:
                block = block.parent

        timestamp = extract_timestamp(block)

        platform_el = block.select_one(".fb-result-platform")
        platform = clean_text(platform_el.get_text()) if platform_el else None

        link_el = block.select_one("a[href*='truthsocial.com']")
        url = link_el["href"] if link_el and link_el.has_attr("href") else None

        text = extract_post_text(block)

        if timestamp or text or url:
            posts.append({
                "timestamp": timestamp,
                "platform": platform,
                "url": url,
                "text": text
            })

    return pd.DataFrame(posts)


# ---------------------------------------------------------
# UI Logic
# ---------------------------------------------------------

if extract_button:
    if not html_input.strip():
        st.error("Please paste HTML first.")
    else:
        df = extract_posts_from_html(html_input)

        if df.empty:
            st.warning("No posts found in the HTML. Try pasting more of the page/source.")
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
