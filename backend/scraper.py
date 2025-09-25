from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import requests
import random

class FriendlyWebScraper:
    """
    Scraper designed for websites that allow scraping
    """
    
    def __init__(self):
        self.driver = None
        self.session = requests.Session()
        self.setup_session()
    
    def setup_session(self):
        """Setup requests session with polite headers"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        self.session.headers.update(headers)
    
    def setup_driver(self):
        """Setup basic Selenium driver"""
        if self.driver:
            return True
            
        try:
            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1366,768")
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            
            print("✅ Driver initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Driver setup failed: {e}")
            return False
    
    def scrape_hacker_news(self, max_items=20):
        """
        Scrape Hacker News - Very scraper-friendly
        """
        print("🔄 Scraping Hacker News...")
        
        try:
            url = "https://news.ycombinator.com/"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                stories = []
                story_links = soup.select('.storylink')
                
                for link in story_links[:max_items]:
                    title = link.get_text(strip=True)
                    href = link.get('href', '')
                    stories.append({
                        'title': title,
                        'url': href
                    })
                
                print(f"✅ Found {len(stories)} Hacker News stories")
                return stories
                
        except Exception as e:
            print(f"❌ Hacker News scraping failed: {e}")
        
        return []
    
    def scrape_reddit_posts(self, subreddit="Python", max_posts=20):
        """
        Scrape Reddit posts - Reddit allows scraping
        """
        print(f"🔄 Scraping Reddit r/{subreddit}...")
        
        try:
            url = f"https://old.reddit.com/r/{subreddit}/"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                posts = []
                post_elements = soup.select('.thing')
                
                for post in post_elements[:max_posts]:
                    try:
                        title_elem = post.select_one('.title a.title')
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                            url = title_elem.get('href', '')
                            
                            # Get score
                            score_elem = post.select_one('.score.unvoted')
                            score = score_elem.get_text(strip=True) if score_elem else "0"
                            
                            posts.append({
                                'title': title,
                                'url': url,
                                'score': score
                            })
                    except:
                        continue
                
                print(f"✅ Found {len(posts)} Reddit posts")
                return posts
                
        except Exception as e:
            print(f"❌ Reddit scraping failed: {e}")
        
        return []
    
    def scrape_wikipedia_page(self, page_title, max_paragraphs=10):
        """
        Scrape Wikipedia - Very scraping-friendly
        """
        print(f"🔄 Scraping Wikipedia: {page_title}")
        
        try:
            url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Get main content paragraphs
                content_div = soup.select_one('#mw-content-text')
                if content_div:
                    paragraphs = content_div.select('p')
                    
                    content = []
                    for p in paragraphs[:max_paragraphs]:
                        text = p.get_text(strip=True)
                        if len(text) > 50:  # Skip short paragraphs
                            content.append(text)
                    
                    print(f"✅ Found {len(content)} Wikipedia paragraphs")
                    return content
                
        except Exception as e:
            print(f"❌ Wikipedia scraping failed: {e}")
        
        return []
    
    def scrape_github_repos(self, username, max_repos=20):
        """
        Scrape GitHub repositories - GitHub is very API and scraping friendly
        """
        print(f"🔄 Scraping GitHub repos for {username}...")
        
        try:
            url = f"https://github.com/{username}?tab=repositories"
            
            if not self.setup_driver():
                return []
            
            self.driver.get(url)
            time.sleep(3)  # Allow page to load
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            repos = []
            repo_elements = soup.select('[data-testid="repository-name-text-container"]')
            
            for repo_elem in repo_elements[:max_repos]:
                try:
                    name = repo_elem.get_text(strip=True)
                    
                    # Get repository link
                    link_elem = repo_elem.find_parent('a')
                    if link_elem:
                        repo_url = "https://github.com" + link_elem.get('href', '')
                        
                        repos.append({
                            'name': name,
                            'url': repo_url
                        })
                except:
                    continue
            
            print(f"✅ Found {len(repos)} GitHub repositories")
            return repos
            
        except Exception as e:
            print(f"❌ GitHub scraping failed: {e}")
        
        return []
    
    def scrape_stack_overflow_questions(self, tag="python", max_questions=20):
        """
        Scrape Stack Overflow questions - Generally allows scraping
        """
        print(f"🔄 Scraping Stack Overflow questions tagged '{tag}'...")
        
        try:
            url = f"https://stackoverflow.com/questions/tagged/{tag}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                questions = []
                question_elements = soup.select('.s-post-summary')
                
                for question in question_elements[:max_questions]:
                    try:
                        # Get title and link
                        title_elem = question.select_one('.s-link')
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                            question_url = "https://stackoverflow.com" + title_elem.get('href', '')
                            
                            # Get votes
                            vote_elem = question.select_one('.s-post-summary--stats-item-number')
                            votes = vote_elem.get_text(strip=True) if vote_elem else "0"
                            
                            questions.append({
                                'title': title,
                                'url': question_url,
                                'votes': votes
                            })
                    except:
                        continue
                
                print(f"✅ Found {len(questions)} Stack Overflow questions")
                return questions
                
        except Exception as e:
            print(f"❌ Stack Overflow scraping failed: {e}")
        
        return []
    
    def scrape_medium_articles(self, topic="technology", max_articles=15):
        """
        Scrape Medium articles - Generally allows reasonable scraping
        """
        print(f"🔄 Scraping Medium articles about '{topic}'...")
        
        try:
            url = f"https://medium.com/tag/{topic}"
            
            if not self.setup_driver():
                return []
            
            self.driver.get(url)
            time.sleep(4)  # Allow dynamic content to load
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            articles = []
            # Medium uses complex selectors, try multiple approaches
            article_selectors = [
                'article h2',
                '[data-testid="post-preview-title"]',
                'h3[data-testid="post-preview-title"]'
            ]
            
            for selector in article_selectors:
                article_elements = soup.select(selector)
                if article_elements:
                    break
            
            for article in article_elements[:max_articles]:
                try:
                    title = article.get_text(strip=True)
                    if len(title) > 10:
                        # Try to find parent link
                        link_elem = article.find_parent('a')
                        article_url = ""
                        if link_elem:
                            href = link_elem.get('href', '')
                            if href.startswith('/'):
                                article_url = "https://medium.com" + href
                            else:
                                article_url = href
                        
                        articles.append({
                            'title': title,
                            'url': article_url
                        })
                except:
                    continue
            
            print(f"✅ Found {len(articles)} Medium articles")
            return articles
            
        except Exception as e:
            print(f"❌ Medium scraping failed: {e}")
        
        return []
    
    def close(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()
            self.driver = None

def test_friendly_sites():
    """
    Test scraping on multiple friendly websites
    """
    scraper = FriendlyWebScraper()
    
    print("🚀 Testing scraper on friendly websites...")
    print("="*50)
    
    try:
        # Test 1: Hacker News
        print("\n1. Testing Hacker News:")
        hn_stories = scraper.scrape_hacker_news(5)
        if hn_stories:
            print("Sample stories:")
            for story in hn_stories[:2]:
                print(f"   • {story['title']}")
        
        # Test 2: Reddit
        print("\n2. Testing Reddit:")
        reddit_posts = scraper.scrape_reddit_posts("programming", 5)
        if reddit_posts:
            print("Sample posts:")
            for post in reddit_posts[:2]:
                print(f"   • {post['title']} (Score: {post['score']})")
        
        # Test 3: Wikipedia
        print("\n3. Testing Wikipedia:")
        wiki_content = scraper.scrape_wikipedia_page("Web_scraping", 3)
        if wiki_content:
            print("Sample content:")
            print(f"   • {wiki_content[0][:100]}...")
        
        # Test 4: Stack Overflow
        print("\n4. Testing Stack Overflow:")
        so_questions = scraper.scrape_stack_overflow_questions("web-scraping", 3)
        if so_questions:
            print("Sample questions:")
            for q in so_questions[:2]:
                print(f"   • {q['title'][:60]}...")
        
        print("\n✅ All friendly site tests completed!")
        
    finally:
        scraper.close()

if __name__ == "__main__":
    test_friendly_sites()