import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# --- WebDriver Setup ---
try:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize Selenium WebDriver: {e}")
    driver = None

def get_product_name_from_url(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        if soup.title and soup.title.string:
            title = soup.title.string
            cleaned_title = re.sub(r'(\s*\|.*)|(\s*-.*)|(\s*@.*)|(buy.*online)|(specs)|(review)|(price in india)|(flipkart)|(amazon)', '', title, flags=re.IGNORECASE)
            return cleaned_title.strip() if cleaned_title else "Unknown Product"
    except Exception:
        pass
    return url.split('/')[-1].replace('-', ' ').replace('_', ' ').title()

def scrape_text_from_url_with_requests(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            paragraphs = main_content.find_all('p') if main_content else []
            return ' '.join([p.get_text(strip=True) for p in paragraphs])
    except Exception:
        return ""

def perform_duckduckgo_search_with_selenium(query: str, num_results: int = 5) -> List[str]:
    if not driver: return []
    search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}&ia=web"
    try:
        driver.get(search_url)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        return [result['href'] for result in soup.find_all('a', {'data-testid': 'result-title-a'}, href=True, limit=num_results)]
    except Exception:
        return []

def get_general_reviews(query: str, num_results: int = 5) -> Dict:
    product_name = query
    if query.startswith('http'):
        product_name = get_product_name_from_url(query)

    search_queries = [f'"{product_name}" review opinions', f'{product_name} user reviews']
    search_results = []
    for s_query in search_queries:
        results = perform_duckduckgo_search_with_selenium(s_query, num_results=num_results)
        if results:
            search_results = results
            break
    
    if not search_results:
        return {"status": "no_data", "message": f"Could not find any reliable search results for '{product_name}'."}

    all_review_text_snippets = []
    successful_sources = []
    for url in search_results:
        content = scrape_text_from_url_with_requests(url)
        if content:
            successful_sources.append(url)
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)
            all_review_text_snippets.extend([s.strip() for s in sentences if 20 < len(s.split()) < 150])

    if not all_review_text_snippets:
        return {"status": "no_data", "message": "Could not extract meaningful text content from the top search results."}
        
    return {
        "status": "success", 
        "reviews": all_review_text_snippets, 
        "product_name": product_name,
        "sources": successful_sources[:3] # Return top 3 successful sources
    }

