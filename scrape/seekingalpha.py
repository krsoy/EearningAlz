# scraper/seekingalpha_scraper.py

import time
import json
import os
import random
import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EARLIEST_DATE = datetime(2020, 1, 1)          # ignore transcripts before this
OUTPUT_DIR    = Path("data/transcripts")
COOKIE_FILE   = Path("data/sa_cookies.pkl")
BASE_URL      = "https://seekingalpha.com"
TRANSCRIPT_LIST_URL = "https://seekingalpha.com/earnings/earnings-call-transcripts"

# Rate limit: wait between requests (seconds)
RATE_LIMIT_MIN = 3.0
RATE_LIMIT_MAX = 7.0

# How many transcript pages to scrape (set None for unlimited)
MAX_PAGES = 10

# ─────────────────────────────────────────────
# DRIVER SETUP
# ─────────────────────────────────────────────
def build_driver(headless: bool = False) -> webdriver.Chrome:
    import os
    options = Options()

    if headless:
        options.add_argument("--headless=new")

    # ── Anti-bot ──────────────────────────────────────────────
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    # ── Use a DEDICATED scraper profile (not your main Chrome) ─
    # This dir is created automatically on first run and reused after
    scraper_profile = Path(os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\SeekingAlphaScraper"))
    scraper_profile.mkdir(parents=True, exist_ok=True)
    options.add_argument(f"--user-data-dir={scraper_profile}")

    # ── Stability ─────────────────────────────────────────────
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")

    # ── Realistic user-agent ──────────────────────────────────
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # ── Patch webdriver fingerprints ─────────────────────────
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": """
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                window.chrome = { runtime: { onConnect: null, onMessage: null } };
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications'
                        ? Promise.resolve({ state: Notification.permission })
                        : originalQuery(parameters)
                );
            """
        },
    )

    return driver



# ─────────────────────────────────────────────
# COOKIE MANAGEMENT
# ─────────────────────────────────────────────
def save_cookies(driver: webdriver.Chrome):
    COOKIE_FILE.parent.mkdir(parents=True, exist_ok=True)
    cookies = driver.get_cookies()
    with open(COOKIE_FILE, "wb") as f:
        pickle.dump(cookies, f)
    print(f"[✓] Cookies saved to {COOKIE_FILE}")


def load_cookies(driver: webdriver.Chrome) -> bool:
    if not COOKIE_FILE.exists():
        return False
    with open(COOKIE_FILE, "rb") as f:
        cookies = pickle.load(f)
    driver.get(BASE_URL)
    time.sleep(2)
    for cookie in cookies:
        try:
            driver.add_cookie(cookie)
        except Exception:
            pass
    driver.refresh()
    time.sleep(3)
    print("[✓] Cookies loaded.")
    return True


def is_logged_in(driver: webdriver.Chrome) -> bool:
    """Check if the session is authenticated."""
    driver.get(BASE_URL)
    time.sleep(3)
    # SeekingAlpha shows user avatar or "Sign In" link
    try:
        driver.find_element(By.CSS_SELECTOR, "[data-test-id='header-user-menu']")
        return True
    except Exception:
        pass
    # fallback: check URL / page source
    return "sign-in" not in driver.current_url and "Sign In" not in driver.page_source[:2000]


# ─────────────────────────────────────────────
# MANUAL AUTH FLOW
# ─────────────────────────────────────────────
def manual_login(driver: webdriver.Chrome):
    print("\n[!] Opening SeekingAlpha login page...")
    driver.get("https://seekingalpha.com/login")
    print("[!] Please log in manually in the browser window.")
    print("[!] After you are fully logged in, come back here and press Enter.")
    input("    >>> Press Enter to continue scraping...")
    save_cookies(driver)
    print("[✓] Login confirmed, cookies saved.")


def authenticate(driver: webdriver.Chrome):
    """Try cookies first, fall back to manual login."""
    if load_cookies(driver):
        if is_logged_in(driver):
            print("[✓] Authenticated via saved cookies.")
            return
        else:
            print("[!] Saved cookies expired or invalid.")
    manual_login(driver)


# ─────────────────────────────────────────────
# RATE-LIMITED SLEEP
# ─────────────────────────────────────────────
def polite_sleep():
    t = random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX)
    print(f"    [~] Sleeping {t:.1f}s...")
    time.sleep(t)


# ─────────────────────────────────────────────
# SCRAPING LOGIC
# ─────────────────────────────────────────────
def parse_transcript_date(date_str: str) -> datetime | None:
    """Parse various date formats from SeekingAlpha."""
    for fmt in ("%b. %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def get_transcript_links(driver: webdriver.Chrome) -> list[dict]:
    """
    Scrape the transcript listing pages and return a list of
    {title, url, date} dicts filtered by EARLIEST_DATE.
    """
    results = []
    page = 1

    while True:
        url = f"{TRANSCRIPT_LIST_URL}?page={page}"
        print(f"\n[→] Fetching listing page {page}: {url}")
        driver.get(url)
        polite_sleep()

        # Wait for article cards to load
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "article, [data-test-id='post-list-item']")
                )
            )
        except Exception:
            print(f"[!] Timeout waiting for articles on page {page}, stopping.")
            break

        # Collect article cards
        cards = driver.find_elements(
            By.CSS_SELECTOR,
            "article, [data-test-id='post-list-item']"
        )
        if not cards:
            print("[!] No articles found, stopping pagination.")
            break

        stop_early = False
        for card in cards:
            try:
                link_el = card.find_element(By.CSS_SELECTOR, "a[href*='/article/']")
                title   = link_el.text.strip()
                href    = link_el.get_attribute("href")

                # Try to find date
                date_el  = card.find_element(By.CSS_SELECTOR, "time, [data-test-id='post-date']")
                date_str = date_el.get_attribute("datetime") or date_el.text
                pub_date = parse_transcript_date(date_str)

                if pub_date is None:
                    # include if we can't parse, to be safe
                    results.append({"title": title, "url": href, "date": None})
                    continue

                if pub_date < EARLIEST_DATE:
                    print(f"    [✗] Reached articles before {EARLIEST_DATE.date()}, stopping.")
                    stop_early = True
                    break

                results.append({"title": title, "url": href, "date": pub_date.strftime("%Y-%m-%d")})
                print(f"    [+] {pub_date.date()} | {title[:70]}")

            except Exception as e:
                print(f"    [?] Skipped one card: {e}")
                continue

        if stop_early:
            break

        page += 1
        if MAX_PAGES and page > MAX_PAGES:
            print(f"[!] Reached MAX_PAGES={MAX_PAGES}, stopping.")
            break

    return results


def scrape_transcript(driver: webdriver.Chrome, meta: dict) -> dict | None:
    """Open a single transcript page and extract the text."""
    print(f"\n  [→] Scraping: {meta['url']}")
    driver.get(meta["url"])
    polite_sleep()

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "[data-test-id='article-content'], .sa-art-article-body")
            )
        )
    except Exception:
        print("  [!] Timeout waiting for article body.")
        return None

    try:
        body_el = driver.find_element(
            By.CSS_SELECTOR,
            "[data-test-id='article-content'], .sa-art-article-body"
        )
        paragraphs = body_el.find_elements(By.TAG_NAME, "p")
        text = "\n".join(p.text for p in paragraphs if p.text.strip())

        return {
            "title":   meta["title"],
            "date":    meta.get("date"),
            "url":     meta["url"],
            "content": text,
        }
    except Exception as e:
        print(f"  [!] Failed to extract body: {e}")
        return None


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SeekingAlpha Earnings Call Transcript Scraper")
    print(f"  Earliest date : {EARLIEST_DATE.date()}")
    print(f"  Max pages     : {MAX_PAGES or 'unlimited'}")
    print(f"  Rate limit    : {RATE_LIMIT_MIN}–{RATE_LIMIT_MAX}s per request")
    print("=" * 60)

    driver = build_driver(headless=False)  # must be visible for manual login

    try:
        # ── Step 1: Authenticate ──────────────────────────
        authenticate(driver)

        # ── Step 2: Collect transcript URLs ──────────────
        print("\n[Phase 1] Collecting transcript links...")
        transcript_list = get_transcript_links(driver)
        print(f"\n[✓] Found {len(transcript_list)} transcripts to scrape.")

        if not transcript_list:
            print("[!] Nothing to scrape. Exiting.")
            return

        # Save the metadata list
        meta_path = OUTPUT_DIR / "transcript_metadata.csv"
        pd.DataFrame(transcript_list).to_csv(meta_path, index=False)
        print(f"[✓] Metadata saved to {meta_path}")

        # ── Step 3: Scrape each transcript ────────────────
        print("\n[Phase 2] Scraping transcript contents...")
        records = []
        for i, meta in enumerate(transcript_list, 1):
            print(f"\n  [{i}/{len(transcript_list)}]", end="")
            result = scrape_transcript(driver, meta)
            if result:
                records.append(result)
                # Save incrementally every 10 transcripts
                if i % 10 == 0:
                    _save_records(records)

        _save_records(records)
        print(f"\n[✓] Done! {len(records)} transcripts saved to {OUTPUT_DIR}")

    finally:
        driver.quit()


def _save_records(records: list[dict]):
    if not records:
        return
    out_path = OUTPUT_DIR / "transcripts_raw.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\n  [✓] Progress saved: {len(records)} records → {out_path}")


if __name__ == "__main__":
    main()
