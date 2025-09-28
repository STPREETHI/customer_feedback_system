import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import logging
import concurrent.futures

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Definitive Selenium WebDriver Setup ---
try:
    log.info("--- Initializing Optimized Selenium WebDriver... ---")
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-logging")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    log.info("--- Selenium WebDriver initialized successfully. ---")
except Exception as e:
    log.critical(f"Failed to initialize Selenium WebDriver: {e}")
    driver = None

def get_product_name_from_url(url: str) -> str:
    """Extracts a clean product name from a URL's page title using requests."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=3)  # Reduced timeout
        soup = BeautifulSoup(response.content, 'html.parser')
        if soup.title and soup.title.string:
            title = soup.title.string
            cleaned_title = re.sub(r'(\s*\|.*)|(\s*-.*)|(\s*@.*)|(buy.*online)|(specs)|(review)|(price in india)|(flipkart)|(amazon)', '', title, flags=re.IGNORECASE)
            return cleaned_title.strip() if cleaned_title else "Unknown Product"
    except Exception as e:
        log.warning(f"Could not extract title from URL {url}: {e}")
    return url.split('/')[-1].replace('-', ' ').replace('_', ' ').title()

def scrape_text_from_url_with_requests(url: str) -> str:
    """Ultra-fast scraper using requests with aggressive timeouts."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=3)  # Very short timeout
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Quick text extraction - just get paragraphs
            paragraphs = soup.find_all('p')[:10]  # Limit to first 10 paragraphs
            return ' '.join([p.get_text(strip=True) for p in paragraphs])
    except Exception as e:
        log.warning(f"Fast scrape failed for {url}: {e}")
    return ""

def perform_duckduckgo_search_with_selenium(query: str, num_results: int = 3) -> List[str]:
    """Ultra-fast DuckDuckGo search with reduced results."""
    if not driver: return []
    search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}&ia=web"
    try:
        log.info(f"Quick search for: {query}")
        driver.get(search_url)
        time.sleep(1)  # Reduced wait time
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        links = [result['href'] for result in soup.find_all('a', {'data-testid': 'result-title-a'}, href=True, limit=num_results)]
        log.info(f"Found {len(links)} links quickly.")
        return links
    except Exception as e:
        log.error(f"Quick search failed: {e}")
        return []

def parallel_scrape_urls(urls: List[str]) -> List[str]:
    """Scrape multiple URLs in parallel for speed."""
    all_snippets = []
    
    def scrape_single_url(url):
        try:
            content = scrape_text_from_url_with_requests(url)
            if content:
                # Quick sentence splitting
                sentences = re.split(r'[.!?]+', content)
                return [s.strip() for s in sentences if 15 < len(s.split()) < 80][:10]  # Max 10 sentences per URL
        except:
            pass
        return []
    
    # Parallel scraping with short timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_url = {executor.submit(scrape_single_url, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url, timeout=10):
            try:
                snippets = future.result(timeout=3)
                all_snippets.extend(snippets)
            except:
                continue
    
    return all_snippets

def get_general_reviews(query: str, num_results: int = 5) -> Dict:
    """Ultra-fast review aggregation with parallel processing."""
    product_name = query
    if query.startswith('http'):
        log.info("Extracting product name from URL...")
        product_name = get_product_name_from_url(query)
        log.info(f"Product name: '{product_name}'")

    # Quick search queries - reduced complexity
    search_queries = [
        f'{product_name} review',
        f'{product_name} user opinion'
    ]
    
    search_results = []
    for s_query in search_queries:
        results = perform_duckduckgo_search_with_selenium(s_query, num_results=num_results)
        if results:
            search_results = results[:num_results]  # Take first successful search
            break
    
    if not search_results:
        return {"status": "no_data", "message": f"Could not find search results for '{product_name}'."}

    # Parallel scraping for maximum speed
    log.info(f"Parallel scraping {len(search_results)} URLs...")
    all_review_snippets = parallel_scrape_urls(search_results)

    if not all_review_snippets:
        return {"status": "no_data", "message": "No meaningful content found in search results."}
        
    log.info(f"Ultra-fast aggregation: {len(all_review_snippets)} snippets")
    return {
        "status": "success", 
        "reviews": all_review_snippets[:50],  # Hard limit for speed
        "product_name": product_name,
        "sources": search_results[:2]  # Limit sources shown
    }