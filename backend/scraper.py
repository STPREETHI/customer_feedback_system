import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time
import logging
import concurrent.futures
import os

# --- Setup logging ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- THE FIX, PART 1: A function to create a browser on demand ---
# This function creates a new, independent browser instance whenever called.
def create_driver():
    """Creates and returns a new Selenium WebDriver instance."""
    log.info("--- Creating new Selenium WebDriver instance... ---")
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-logging")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")

    try:
        if os.environ.get('RENDER'):
            service = Service()
        else:
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
        
        driver = webdriver.Chrome(service=service, options=options)
        log.info("--- WebDriver instance created successfully. ---")
        return driver
    except Exception as e:
        log.critical(f"Failed to create Selenium WebDriver instance: {e}")
        return None

# --- A single, global driver for non-concurrent tasks ---
# This will be used by the main "analyze" function.
main_driver = create_driver()

def get_product_name_from_url(url: str) -> str:
    """Extracts a clean product name from a URL's page title using requests."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        if soup.title and soup.title.string:
            title = soup.title.string
            cleaned_title = re.sub(r'(\s*\|.*)|(\s*-.*)|(\s*@.*)|(buy.*online)|(specs)|(review)|(price in india)|(flipkart)|(amazon)', '', title, flags=re.IGNORECASE)
            return cleaned_title.strip() if cleaned_title else "Unknown Product"
    except requests.RequestException as e:
        log.warning(f"Could not extract title from URL {url}: {e}")
    return url.split('/')[-1].replace('-', ' ').replace('_', ' ').title()

def scrape_text_from_url_with_requests(url: str) -> str:
    """Fast scraper using requests."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')[:15]
            return ' '.join([p.get_text(strip=True) for p in paragraphs])
    except requests.RequestException as e:
        log.warning(f"Fast scrape failed for {url}: {e}")
    return ""

# --- THE FIX, PART 2: The search function now accepts a driver ---
def perform_duckduckgo_search_with_selenium(driver, query: str, num_results: int = 5) -> List[str]:
    """Performs a search using a provided Selenium driver instance."""
    if not driver: return []
    search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}&ia=web"
    try:
        log.info(f"Searching for: {query}")
        driver.get(search_url)
        time.sleep(1.5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        links = [result['href'] for result in soup.find_all('a', {'data-testid': 'result-title-a'}, href=True, limit=num_results)]
        log.info(f"Found {len(links)} links.")
        return links
    except Exception as e:
        log.error(f"Selenium search failed: {e}")
        return []

def parallel_scrape_urls(urls: List[str]) -> List[str]:
    """Scrapes multiple URLs in parallel."""
    all_snippets = []
    def scrape_single_url(url):
        content = scrape_text_from_url_with_requests(url)
        if content:
            sentences = re.split(r'[.!?]+', content)
            return [s.strip() for s in sentences if 15 < len(s.split()) < 80][:10]
        return []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(scrape_single_url, url): url for url in urls}
        try:
            for future in concurrent.futures.as_completed(future_to_url, timeout=20):
                snippets = future.result()
                if snippets: all_snippets.extend(snippets)
        except concurrent.futures.TimeoutError:
            log.warning("Scraping timed out for some URLs, proceeding with partial results.")
    return all_snippets

# --- THE FIX, PART 3: The main function now accepts an optional driver ---
def get_general_reviews(query: str, num_results: int = 5, driver=None) -> Dict:
    """Aggregates reviews. Uses a provided driver or the main global driver."""
    # Use the provided driver if one is given, otherwise default to the global `main_driver`
    active_driver = driver if driver else main_driver
    
    product_name = query
    if query.startswith('http'):
        product_name = get_product_name_from_url(query)

    search_query = f'{product_name} review'
    search_results = perform_duckduckgo_search_with_selenium(active_driver, search_query, num_results=num_results)
    
    if not search_results:
        return {"status": "failure", "message": f"Could not find search results for '{product_name}'."}

    all_review_snippets = parallel_scrape_urls(search_results)

    if not all_review_snippets:
        return {"status": "failure", "message": "Could not extract review content."}
        
    return {
        "status": "success", 
        "reviews": all_review_snippets[:50],
        "product_name": product_name,
        "sources": search_results[:3]
    }