import os
import io
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
#v2.2
# ---------- CONFIG ----------

MASTER_CSV = "data/posts.csv"
os.makedirs("data", exist_ok=True)

# ---------- HELPERS ----------

def load_master():
    if os.path.exists(MASTER_CSV):
        return pd.read_csv(MASTER_CSV)
    return pd.DataFrame(columns=["timestamp", "platform", "url", "text"])

def save_master(df):
    df.to_csv(MASTER_CSV, index=False)

def extract_timestamp(block):
    # Get all spans that have both classes: "hidden" and "md:inline"
    spans = block.find_all("span", class_=["hidden", "md:inline"])
    for span in spans:
        txt = span.get_text(strip=True)
        # Skip bullet "•", keep real timestamp
        if "@" in txt and "ET" in txt:
            return txt
    return ""

def extract_platform(block):
    # Look for Truth Social / Twitter indicator in the header area
    # First try explicit platform span
    platform_span = block.find("span", string=lambda s: s and "Truth Social" in s)
    if platform_span:
        return "Truth Social"
    # Fallback: check for X / Twitter icon context
    x_icon = block.find("i", class_="fa-brands fa-x-twitter")
    if x_icon:
        return "Twitter"
    return ""

def extract_url(block):
    # Prefer the "View on Truth Social" or "View on X" link
    link = block.find("a", string=lambda s: s and "View on" in s)
    if link and link.get("href"):
        return link["href"].strip()
    # Fallback: any link that looks like a Truth Social or X URL
    for a in block.find_all("a", href=True):
        href = a["href"]
        if "truthsocial.com" in href or "twitter.com" in href or "x.com" in href:
            return href.strip()
    return ""

def extract_text(block):
    # Prefer item.social.post_html container
    post_div = block.find("div", attrs={"x-html": "item.social.post_html"})
    if post_div:
        return post_div.get_text(" ", strip=True)

    # Fallback: generic text container used in your snippet
    post_div = block.find("div", class_="text-sm")
    if post_div:
        return post_div.get_text(" ", strip=True)

    return ""

def extract_posts_from_html(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")

    # Each post block: div.block.mb-8.rounded-xl.border...
    post_blocks = soup.find_all("div", class_="block mb-8 rounded-xl border border-[#2F3C4B]/20")

    records = []
    for block in post_blocks:
        timestamp = extract_timestamp(block)
        platform = extract_platform(block)
        url = extract_url(block)
        text = extract_text(block)

        # Only keep rows that have at least a URL
        if url:
            records.append(
                {
                    "timestamp": timestamp,
                    "platform": platform,
                    "url": url,
                    "text": text,
                }
            )

    return pd.DataFrame(records, columns=["timestamp", "platform", "url", "text"])

# ---------- STREAMLIT APP ----------

st.title("Trump Social Media HTML → CSV Archive")

st.write("Upload one or more `.htm` files from Roll Call / Factba.se, and I'll extract posts, "
         "merge them into a master CSV, and remove duplicates by URL.")

uploaded_files = st.file_uploader(
    "Upload .htm files",
    type=["htm", "html"],
    accept_multiple_files=True
)

if uploaded_files:
    all_new_rows = []

    for f in uploaded_files:
        st.subheader(f"File: {f.name}")
        content = f.read()
        try:
            decoded = content.decode("utf-8", errors="ignore")
        except Exception:
            decoded = content.decode("latin-1", errors="ignore")

        df_new = extract_posts_from_html(decoded)
        st.write(f"Extracted {len(df_new)} posts from this file.")
        st.dataframe(df_new.head())
        all_new_rows.append(df_new)

    if all_new_rows:
        df_new_all = pd.concat(all_new_rows, ignore_index=True)
        st.subheader("Combined new posts from this upload")
        st.write(f"Total new extracted posts (before dedupe vs repository): {len(df_new_all)}")
        st.dataframe(df_new_all.head())

        # Load master, merge, dedupe
        master = load_master()
        before_count = len(master)

        combined = pd.concat([master, df_new_all], ignore_index=True)
        combined = combined.drop_duplicates(subset=["url"], keep="first")

        added = len(combined) - before_count

        save_master(combined)

        st.success(f"Added {added} new posts to repository (duplicates removed).")
        st.write(f"Repository now contains {len(combined)} total posts.")

        # Offer download of just-this-upload and full master
        st.download_button(
            "Download this upload as CSV",
            data=df_new_all.to_csv(index=False).encode("utf-8"),
            file_name="this_upload_posts.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download full repository CSV",
            data=combined.to_csv(index=False).encode("utf-8"),
            file_name="posts_master.csv",
            mime="text/csv",
        )

else:
    st.info("Upload one or more `.htm` files to begin.")
