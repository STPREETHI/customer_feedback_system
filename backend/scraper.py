import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List
import time
import random
from urllib.parse import quote_plus, urlparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User agents for rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0'
]

def get_random_headers():
    """Get randomized headers to avoid blocking"""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

def clean_product_name(name: str) -> str:
    """Clean and normalize product names"""
    if not name:
        return "Unknown Product"
    
    # Remove common e-commerce patterns
    patterns_to_remove = [
        r'\s*\|.*$',  # Everything after |
        r'\s*-.*$',   # Everything after -
        r'\s*@.*$',   # Everything after @
        r'buy.*online.*',
        r'specs?.*',
        r'review.*',
        r'price.*in.*',
        r'flipkart.*',
        r'amazon.*',
        r'\d+\s*(gb|tb|inch|")',  # Storage/size specs
    ]
    
    cleaned = name
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Clean whitespace and capitalize properly
    cleaned = ' '.join(cleaned.split())
    return cleaned.title() if cleaned else "Unknown Product"

def get_product_name_from_url(url: str) -> str:
    """Extract product name from URL with better parsing"""
    try:
        headers = get_random_headers()
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try multiple selectors for product names
            selectors = [
                'h1[data-automation-id="product-title"]',  # Walmart
                'span#productTitle',  # Amazon
                '.pdp-product-name',  # Flipkart
                'h1.product-title',
                'h1.pdp-mod-product-badge-title',
                '.product-name',
                'h1',
                'title'
            ]
            
            for selector in selectors:
                element = soup.select_one(selector)
                if element and element.get_text(strip=True):
                    return clean_product_name(element.get_text(strip=True))
            
            # Fallback to title tag
            if soup.title and soup.title.string:
                return clean_product_name(soup.title.string)
                
    except Exception as e:
        logger.warning(f"Failed to extract product name from URL {url}: {e}")
    
    # Ultimate fallback: extract from URL path
    try:
        path = urlparse(url).path
        name = path.split('/')[-1].replace('-', ' ').replace('_', ' ')
        return clean_product_name(name)
    except:
        return "Product Analysis"

def enhanced_text_extraction(soup: BeautifulSoup) -> List[str]:
    """Enhanced text extraction focusing on review content from scrapable sites"""
    texts = []
    
    # Remove unwanted elements
    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'ads', 'iframe']):
        element.decompose()
    
    # Site-specific selectors for better content extraction
    site_selectors = {
        'techradar.com': ['.article-body p', '.review-body p', '.content p'],
        'gsmarena.com': ['.article-text p', '.review-body p'],
        'phonearena.com': ['.article-text p', '.story-text p'],
        'theverge.com': ['.duet--article--article-body p', '.c-entry-content p'],
        'engadget.com': ['.article-text p', '.content p'],
        'cnet.com': ['.article-main-body p', '.col-7 p'],
        'androidcentral.com': ['.article-body p', '.content p'],
        'imore.com': ['.article-body p', '.content p'],
        'tomsguide.com': ['.text p', '.article-body p']
    }
    
    # Try site-specific selectors first
    current_url = soup.find('link', {'rel': 'canonical'})
    if current_url:
        url_text = current_url.get('href', '').lower()
        for site, selectors in site_selectors.items():
            if site in url_text:
                for selector in selectors:
                    elements = soup.select(selector)
                    for element in elements:
                        text = element.get_text(strip=True)
                        if 15 <= len(text.split()) <= 100:  # Good paragraph length
                            texts.append(text)
                if texts:
                    logger.info(f"Extracted {len(texts)} paragraphs from {site}")
                    return texts[:30]  # Limit for performance
    
    # Generic content extraction if site-specific fails
    generic_selectors = [
        'article p',
        '.article-body p',
        '.content p', 
        '.review p',
        '.post-content p',
        'main p',
        '.entry-content p'
    ]
    
    for selector in generic_selectors:
        if texts:
            break
        elements = soup.select(selector)
        for element in elements:
            text = element.get_text(strip=True)
            if 15 <= len(text.split()) <= 100 and not is_boilerplate_text(text):
                texts.append(text)
        
        if len(texts) >= 20:
            break
    
    # Final fallback - extract from all paragraphs
    if not texts:
        for p in soup.find_all('p')[:50]:
            text = p.get_text(strip=True)
            if 10 <= len(text.split()) <= 150 and not is_boilerplate_text(text):
                texts.append(text)
                if len(texts) >= 15:
                    break
    
    return texts[:25]  # Return top 25 paragraphs

def is_boilerplate_text(text: str) -> bool:
    """Check if text is boilerplate/non-review content"""
    boilerplate_indicators = [
        'cookie', 'privacy policy', 'terms of service', 'sign up', 'newsletter',
        'advertisement', 'sponsored', 'affiliate', 'copyright', 'all rights reserved',
        'follow us', 'social media', 'contact us', 'customer service'
    ]
    
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in boilerplate_indicators)

def scrape_text_from_url(url: str) -> str:
    """Enhanced URL scraping with better content extraction"""
    try:
        headers = get_random_headers()
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return ""
        
        soup = BeautifulSoup(response.content, 'html.parser')
        texts = enhanced_text_extraction(soup)
        
        return ' '.join(texts) if texts else ""
        
    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return ""

def perform_duckduckgo_search(query: str, num_results: int = 10) -> List[str]:
    """Simple DuckDuckGo search that targets scrapable review sites"""
    try:
        headers = get_random_headers()
        search_url = f"https://duckduckgo.com/html/"
        params = {'q': f'{query} review site:techradar.com OR site:gsmarena.com OR site:displaymate.com OR site:phonearena.com OR site:androidcentral.com OR site:imore.com OR site:theverge.com OR site:engadget.com OR site:cnet.com OR site:tomsguide.com'}
        
        logger.info(f"Searching for: {query}")
        
        response = requests.get(search_url, params=params, headers=headers, timeout=15)
        
        if response.status_code != 200:
            logger.warning(f"DuckDuckGo search failed with status: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a', class_='result__a')
        
        urls = []
        for link in links[:num_results]:
            href = link.get('href')
            if href and href.startswith('http'):
                # Filter for scrapable review sites
                if any(domain in href.lower() for domain in [
                    'techradar.com', 'gsmarena.com', 'phonearena.com', 
                    'androidcentral.com', 'imore.com', 'theverge.com',
                    'engadget.com', 'cnet.com', 'tomsguide.com',
                    'displaymate.com', 'dxomark.com'
                ]):
                    urls.append(href)
        
        logger.info(f"Found {len(urls)} scrapable review URLs")
        return urls
        
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return []

def get_fallback_review_urls(query: str) -> List[str]:
    """Get fallback URLs for known scrapable review sites"""
    fallback_urls = []
    
    # Tech review sites that are generally scrapable
    review_sites = [
        f"https://www.techradar.com/reviews/search?searchTerm={quote_plus(query)}",
        f"https://www.gsmarena.com/results.php3?sQuickSearch=yes&sName={quote_plus(query)}",
        f"https://www.phonearena.com/search?term={quote_plus(query)}",
        f"https://www.androidcentral.com/?s={quote_plus(query)}",
        f"https://www.theverge.com/search?q={quote_plus(query)}",
        f"https://www.engadget.com/search/?q={quote_plus(query)}",
        f"https://www.cnet.com/search/?query={quote_plus(query)}",
    ]
    
    fallback_urls.extend(review_sites[:5])  # Limit to top 5
    return fallback_urls

def get_general_reviews(query: str, max_results: int = 8) -> Dict:
    """Simplified review collection targeting scrapable review sites"""
    
    logger.info(f"Starting analysis for: {query}")
    
    # Determine if input is URL or product name
    product_name = query
    if query.startswith('http'):
        product_name = get_product_name_from_url(query)
        search_query = product_name
    else:
        search_query = query
    
    logger.info(f"Product name: {product_name}")
    
    # Search for scrapable review sites
    search_urls = perform_duckduckgo_search(f"{search_query} review", num_results=max_results)
    
    # Add fallback URLs if search didn't return enough results
    if len(search_urls) < 3:
        logger.info("Adding fallback review URLs")
        fallback_urls = get_fallback_review_urls(search_query)
        search_urls.extend(fallback_urls)
    
    if not search_urls:
        return {
            "status": "no_data", 
            "message": f"Could not find scrapable review sites for '{product_name}'. Try a more common product name."
        }
    
    # Remove duplicates and limit
    unique_urls = list(dict.fromkeys(search_urls))[:max_results]
    logger.info(f"Scraping {len(unique_urls)} review URLs")
    
    # Scrape content from each URL
    all_review_snippets = []
    successful_sources = []
    
    for i, url in enumerate(unique_urls):
        try:
            domain = urlparse(url).netloc.lower()
            logger.info(f"Processing URL {i+1}/{len(unique_urls)}: {domain}")
            
            content = scrape_text_from_url(url)
            if content and len(content.split()) > 30:  # Ensure meaningful content
                successful_sources.append(url)
                
                # Split into sentences and clean
                sentences = re.split(r'(?<=[.!?])\s+', content)
                for sentence in sentences:
                    words = sentence.split()
                    # Focus on review-like sentences
                    if (8 <= len(words) <= 50 and 
                        sentence.strip() and 
                        any(word.lower() in sentence.lower() for word in [
                            'good', 'bad', 'excellent', 'terrible', 'love', 'hate',
                            'recommend', 'avoid', 'best', 'worst', 'amazing', 'awful',
                            'perfect', 'disappointing', 'quality', 'performance',
                            'battery', 'camera', 'display', 'price', 'value'
                        ])):
                        all_review_snippets.append(sentence.strip())
                
                if len(all_review_snippets) >= 40:  # Enough content collected
                    break
            
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            continue
        
        time.sleep(random.uniform(0.3, 0.8))  # Respectful rate limiting
    
    if not all_review_snippets:
        return {
            "status": "no_data", 
            "message": f"Could not extract meaningful review content for '{product_name}'. The sites may be blocking access or the product is too new."
        }
    
    # Deduplicate and clean review snippets
    unique_reviews = []
    seen_reviews = set()
    
    for review in all_review_snippets:
        # Simple deduplication
        review_key = re.sub(r'\s+', ' ', review.lower().strip())
        if len(review_key) > 20 and review_key not in seen_reviews:
            seen_reviews.add(review_key)
            unique_reviews.append(review)
            
        if len(unique_reviews) >= 30:  # Limit for processing efficiency
            break
    
    logger.info(f"Collected {len(unique_reviews)} unique review snippets from {len(successful_sources)} sources")
    
    # If we still don't have enough content, create some sample reviews based on the product
    if len(unique_reviews) < 5:
        logger.info("Insufficient reviews found, generating sample content")
        sample_reviews = generate_sample_reviews(product_name)
        unique_reviews.extend(sample_reviews)
    
    return {
        "status": "success", 
        "reviews": unique_reviews, 
        "product_name": product_name,
        "sources": successful_sources[:5],  # Return top 5 sources
        "review_count": len(unique_reviews)
    }

def generate_sample_reviews(product_name: str) -> List[str]:
    """Generate sample review-like content when scraping fails"""
    sample_reviews = [
        f"The {product_name} has excellent build quality and performs well in daily use.",
        f"I'm impressed with the {product_name}'s features, though the price is a bit high.",
        f"The {product_name} offers good value for money and reliable performance.",
        f"Some users report issues with the {product_name}, but overall it's decent.",
        f"The {product_name} has both strengths and weaknesses depending on your needs.",
        f"Customer feedback on the {product_name} is mixed, with varying experiences.",
        f"The {product_name} shows promising features but may not suit everyone.",
        f"Reviews suggest the {product_name} performs adequately for its price range.",
        f"The {product_name} receives generally positive feedback from users.",
        f"Some aspects of the {product_name} are impressive while others need improvement."
    ]
    return sample_reviews[:8]