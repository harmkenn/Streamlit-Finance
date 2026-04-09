import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import quopri
import re
# v1.6
st.set_page_config(page_title="Factba.se / Truth Social HTML → CSV", layout="wide")

st.title("Factba.se / Truth Social HTML → CSV")
st.write("Paste HTML (from page source, email, or Elements) and extract posts into a CSV.")

st.markdown("---")

html_input = st.text_area(
    "Paste the HTML code here:",
    height=400,
    placeholder="Paste the HTML that contains the posts…"
)

extract_button = st.button("Extract Posts")


def clean_text(t: str | None) -> str:
    if not t:
        return ""
    return re.sub(r"\s+", " ", t).strip()


def decode_html(raw: str) -> str:
    """Decode quoted-printable HTML like class=3D, =E2=80=9C, =\n, etc."""
    try:
        decoded = quopri.decodestring(raw).decode("utf-8", errors="ignore")
        return decoded
    except Exception:
        return raw


def extract_timestamp(block) -> str | None:
    """
    Extract timestamps like:
    <span class="hidden md:inline">April 8, 2026 @ 11:46 PM ET</span>
    """
    ts_el = block.select_one("span.hidden.md\\:inline")
    if ts_el and ts_el.get_text(strip=True):
        return clean_text(ts_el.get_text())

    # Fallback: any span containing "@" and "ET"
    for span in block.find_all("span"):
        txt = clean_text(span.get_text())
        if "@" in txt and "ET" in txt:
            return txt

    return None


def extract_post_text(block) -> str | None:
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

    # Fallback: text-only variant
    text_el = block.select_one('[x-html="item.text"], .fb-result-text')
    if text_el:
        return clean_text(text_el.get_text(separator=" ", strip=True))

    return None


def extract_url(block) -> str | None:
    """
    Extract Truth Social URL like:
    <a href="https://truthsocial.com/@realDonaldTrump/posts/...">View on Truth Social</a>
    """
    link_el = block.select_one("a[href*='truthsocial.com']")
    if link_el and link_el.has_attr("href"):
        return link_el["href"]
    return None


def extract_platform(block) -> str | None:
    """
    Try to infer platform (Truth Social / Twitter) from icon/text.
    """
    # Explicit platform text if present
    plat_el = block.select_one(".fb-result-platform")
    if plat_el:
        return clean_text(plat_el.get_text())

    # Look for Truth Social icon
    if block.select_one('img[alt="Truth Social icon"]'):
        return "Truth Social"

    # Look for X/Twitter icon
    if block.select_one(".fa-x-twitter") or block.select_one(".fa-brands.fa-x-twitter"):
        return "Twitter"

    return None


def extract_posts_from_html(html: str) -> pd.DataFrame:
    decoded = decode_html(html)
    soup = BeautifulSoup(decoded, "html.parser")

    posts = []

    # Each post has a div with x-html="item.social.post_html"
    for html_block in soup.select('[x-html="item.social.post_html"]'):
        # Walk up a few levels to get the full post container
        block = html_block
        for _ in range(4):
            if block.parent:
                block = block.parent

        timestamp = extract_timestamp(block)
        text = extract_post_text(block)
        url = extract_url(block)
        platform = extract_platform(block)

        if timestamp or text or url:
            posts.append(
                {
                    "timestamp": timestamp,
                    "platform": platform,
                    "url": url,
                    "text": text,
                }
            )

    return pd.DataFrame(posts)


if extract_button:
    if not html_input.strip():
        st.error("Please paste HTML first.")
    else:
        df = extract_posts_from_html(html_input)

        if df.empty:
            st.warning(
                "No posts found. Make sure the HTML includes the rendered blocks "
                "with x-html=\"item.social.post_html\" and the timestamp span."
            )
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
