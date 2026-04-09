import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import time
import asyncio
from playwright.async_api import async_playwright

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Trump Posts Scraper", layout="wide")

BASE_URL = "https://rollcall.com/factbase/trump/topic/social/?platform=all&sort=date&sort_order=desc&page={}"

# ---------------------------------------------------------
# SCRAPER FUNCTIONS
# ---------------------------------------------------------
async def scrape_page_async(page_number: int):
    """Scrape a single page using Playwright (handles JavaScript rendering)."""
    url = BASE_URL.format(page_number)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        try:
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(2000)  # Extra wait for dynamic content
            
            html = await page.content()
            soup = BeautifulSoup(html, "html.parser")
            
            posts = []
            
            # Find all <p> tags that contain post text
            p_tags = soup.find_all("p")
            
            for p_tag in p_tags:
                try:
                    # Extract text from the <p> tag
                    text = p_tag.get_text(strip=True)
                    
                    # Only process if the text is substantial enough
                    if not text or len(text) < 20:
                        continue
                    
                    # Find the parent container (go up from the p tag)
                    item = p_tag.parent
                    while item and item.name != "main" and len(item.get("class", [])) == 0:
                        item = item.parent
                    
                    if not item:
                        item = p_tag.parent
                    
                    # Extract link - look in the parent container
                    link = None
                    link_el = item.find("a", href=True) if item else None
                    if link_el:
                        link = link_el.get("href")
                    
                    # Extract date/time - look in the parent container
                    timestamp = None
                    
                    # Try to find time element first
                    date_el = item.find("time") if item else None
                    
                    # If not found, look for span tags that might contain date info
                    if not date_el and item:
                        # Look for span tags with datetime or title attributes
                        for span in item.find_all("span"):
                            if span.get("datetime") or span.get("title"):
                                date_el = span
                                break
                        
                        # If still not found, get all spans and look for date-like patterns
                        if not date_el:
                            all_spans = item.find_all("span")
                            # Usually the date is in one of the later spans
                            if len(all_spans) >= 5:
                                date_el = all_spans[4]  # Often around the 5th span based on your XPath
                    
                    if date_el:
                        timestamp = date_el.get("datetime") or date_el.get("title") or date_el.get_text(strip=True)
                    
                    posts.append({
                        "timestamp": timestamp,
                        "text": text,
                        "url": link,
                    })
                except Exception as e:
                    st.warning(f"Error parsing item: {e}")
                    continue
            
            return posts
            
        finally:
            await browser.close()


def scrape_page(page_number: int):
    """Wrapper to run async scraping in Streamlit."""
    try:
        return asyncio.run(scrape_page_async(page_number))
    except Exception as e:
        st.error(f"Scraping error on page {page_number}: {e}")
        return []


def scrape_all_pages(max_pages: int, delay: float = 1.0):
    """Scrape multiple pages with progress bar."""
    all_posts = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for page in range(1, max_pages + 1):
        progress_bar.progress(page / max_pages)
        status_text.write(f"Scraping page {page}…")

        posts = scrape_page(page)
        if not posts:
            status_text.write(f"No posts found on page {page} — stopping early.")
            break

        all_posts.extend(posts)
        status_text.write(f"Found {len(posts)} posts on page {page}. Total: {len(all_posts)}")
        time.sleep(delay)

    progress_bar.progress(1.0)
    return pd.DataFrame(all_posts) if all_posts else pd.DataFrame()


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
    value=1,
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
