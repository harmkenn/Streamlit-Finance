import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Truth Social Scraper", layout="wide")

BASE_URL = "https://rollcall.com/factbase/trump/topic/social/?platform=all&sort=date&sort_order=desc&page={}"

headers = {
    "User-Agent": "Mozilla/5.0"
}

# ---------------------------------------------------------
# SCRAPER FUNCTIONS
# ---------------------------------------------------------
def scrape_page(page_number: int):
    """Scrape a single page of posts."""
    url = BASE_URL.format(page_number)
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    posts = []

    for item in soup.select("div.fb-result"):
        text_el = item.select_one(".fb-result-text")
        date_el = item.select_one(".fb-result-date")
        link_el = item.select_one("a")
        platform_el = item.select_one(".fb-result-platform")

        text = text_el.get_text(strip=True) if text_el else None
        timestamp = date_el.get_text(strip=True) if date_el else None
        url = link_el["href"] if link_el else None
        platform = platform_el.get_text(strip=True) if platform_el else None

        posts.append({
            "timestamp": timestamp,
            "text": text,
            "url": url,
            "platform": platform
        })

    return posts


def scrape_all_pages(max_pages: int, delay: float = 1.0):
    """Scrape multiple pages with progress bar."""
    all_posts = []
    progress = st.progress(0)

    for page in range(1, max_pages + 1):
        progress.progress(page / max_pages)
        st.write(f"Scraping page {page}…")

        posts = scrape_page(page)
        if not posts:
            st.write("No more posts found — stopping early.")
            break

        all_posts.extend(posts)
        time.sleep(delay)

    progress.progress(1.0)
    return pd.DataFrame(all_posts)


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("Truth Social Post Scraper")
st.write("Scrapes Trump posts from the RollCall/Factba.se archive and exports them to CSV.")

st.markdown("---")

max_pages = st.number_input(
    "How many pages to scrape?",
    min_value=1,
    max_value=200,
    value=10,
    step=1
)

delay = st.slider(
    "Delay between page requests (seconds)",
    min_value=0.0,
    max_value=3.0,
    value=1.0,
    step=0.1
)

start_button = st.button("Start Scraping")

if start_button:
    st.info("Scraping started…")
    df = scrape_all_pages(max_pages=max_pages, delay=delay)

    if df.empty:
        st.warning("No posts found.")
    else:
        st.success(f"Scraped {len(df)} posts.")
        st.dataframe(df, use_container_width=True)

        # Deduplicate
        df.drop_duplicates(subset=["timestamp", "text"], inplace=True)

        # Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="trump_truth_posts.csv",
            mime="text/csv"
        )
