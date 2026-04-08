import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://rollcall.com/factbase/trump/topic/social/?platform=all&sort=date&sort_order=desc&page={}"

headers = {
    "User-Agent": "Mozilla/5.0"
}

def scrape_page(page_number):
    url = BASE_URL.format(page_number)
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    posts = []

    # Each post is in a <div class="fb-result">
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


def scrape_all_pages(max_pages=10, delay=1.0):
    all_posts = []

    for page in range(1, max_pages + 1):
        print(f"Scraping page {page}...")
        posts = scrape_page(page)
        if not posts:
            print("No more posts found.")
            break
        all_posts.extend(posts)
        time.sleep(delay)

    return pd.DataFrame(all_posts)


# Run the scraper
df = scrape_all_pages(max_pages=20)
df.drop_duplicates(subset=["timestamp", "text"], inplace=True)

df.to_csv("trump_truth_posts.csv", index=False)
print("Saved trump_truth_posts.csv with", len(df), "posts.")
