import os
import re
import time
from typing import Optional, Tuple
from argparse import ArgumentParser
from bs4 import BeautifulSoup, Comment
from fake_useragent import UserAgent
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from newspaper import Article
import logging
from urllib.parse import urlparse
import requests
from selenium import webdriver
from playwright.sync_api import sync_playwright

CONTENT_CLASS_PATTERN = re.compile(r'content|main-content|post-content|body|main-body|body-content|main', re.IGNORECASE)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_article_text(url: str) -> Optional[Tuple[str, str, Optional[str], str]]:
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        title = article.title
        author = ', '.join(article.authors)
        pub_date = article.publish_date
        article_text = article.text
        
        if all([title, article_text]):
            return title, author, pub_date, article_text
        return None
    except Exception as e:
        logger.error(f"Error fetching article: {e}")
        return None
    
class WebContentExtractor:
    _driver = None
    @classmethod
    def _initialize_driver(cls):
        edge_options = EdgeOptions()
        edge_options.add_argument('--headless')
        edge_options.add_argument('--disable-gpu')
        edge_options.add_argument('--no-sandbox')
        edge_options.add_argument('--disable-dev-shm-usage')
        edge_options.add_argument(f'user-agent={UserAgent().random}')
    
        service = EdgeService(executable_path="C:\\webdrivers\\edgedriver_win64\\msedgedriver.exe")
        cls._driver = webdriver.Edge(service=service, options=edge_options)
    
    def __init__(self, max_retries: int = 3, timeout: int = 10):
        self.max_retries = max_retries
        self.timeout = timeout

    @classmethod
    def get_driver(cls):
        if cls._driver is None:
            cls._initialize_driver()
        return cls._driver
    
    @classmethod
    def quit_driver(cls):
        if cls._driver:
            cls._driver.quit()
            cls._driver = None
    
    def __del__(self):
        self.quit_driver()
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def _extract_content_from_soup(soup: BeautifulSoup) -> str:
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_with_requests(self, url: str) -> Optional[str]:
        try:
            headers = {
                'User-Agent': UserAgent().random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0'
            }
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            return self._extract_content_from_soup(soup)
        except Exception as e:
            logger.error(f"Error extracting content with requests: {e}")
            return None
    
    def _extract_with_newspaper(self, url: str) -> Optional[str]:
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text.strip()
        except Exception as e:
            logger.error(f"Error extracting content with newspaper3k: {e}")
            return None
    @classmethod
    def _extract_with_playwright(cls, url: str) -> str:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=60000)
                page.wait_for_load_state('networkidle')
                html_content = page.content()
                browser.close()

                soup = BeautifulSoup(html_content, 'html.parser')
                main_content = soup.find(['div', 'main', 'article'], class_=CONTENT_CLASS_PATTERN) or soup.body
                main_text = main_content.get_text(separator=' ', strip=True) if main_content else ''
                return re.sub(r'\s+', ' ', main_text)
        except Exception as e:
            logging.error(f"Playwright extraction failed for {url}: {e}")
            return ""
        
    def extract_with_selenium(self, url: str) -> Optional[str]:
        try:
            driver = self.get_driver()
            driver.get(url)
            
            WebDriverWait(driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            return self._extract_content_from_soup(soup)
        except Exception as e:
            logger.error(f"Error extracting content with Selenium: {e}")
            return None
    
    
    def extract_content(self, url: str, retry_count: int = 0) -> Optional[str]:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        if not self.is_valid_url(url):
            url = input("URL Invalid. Enter URL and press Enter: ")
            if not self.is_valid_url(url):
                logger.error(f"Invalid URL: {url}")
                return None
        
        content = self._extract_with_playwright(url)
        if content and len(content.strip()) >= 200:
            return content
        
        content = self._extract_with_requests(url)
        if content and len(content.strip()) >= 200:
            return content
        
        content = self._extract_with_newspaper(url)
        if content and len(content.strip()) >= 200:
            return content
        
        content = self.extract_with_selenium(url)
        if content and len(content.strip()) >= 200:
            return content
        
        if retry_count < self.max_retries:
            logger.info(f"Retrying content extraction for {url} (attempt {retry_count + 1})")
            time.sleep(1)  # Add delay between retries
            return self.extract_content(url, retry_count + 1)
        logger.error(f"All content extraction methods failed for {url}")
        return None
    
if __name__ == "__main__":
    parser = ArgumentParser(description='Extract content from a web page.')
    parser.add_argument('--url', type=str, help='URL of the web page to extract content from')
    parser.add_argument('--timeout', type=int, default=10, help='Timeout in seconds for requests (default: 10)')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retry attempts (default: 3)')
    
    args = parser.parse_args()
    
    if not args.url:
        url = "https://seekingalpha.com/news/4230736-openai-brings-search-to-chatgpt-heating-up-rivalry-with-google?feed_item_type=news&fr=1&utm_medium=referral&utm_source=msn.com"
    else:
        url = args.url
    
    extractor = WebContentExtractor(max_retries=args.max_retries, timeout=args.timeout)
    
    try:
        content = extractor.extract_content(url)
        
        if content:
            print("\nExtracted Content:")
            print("-" * 80)
            print(content)
            print("-" * 80)
            
            article_meta = fetch_article_text(url)
            if article_meta:
                title, author, pub_date, _ = article_meta
                print("\nMeta Information:")
                print(f"Title: {title}")
                print(f"Author: {author}")
                print(f"Publish Date: {pub_date}")
        else:
            print("Failed to extract content from the URL.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    finally:
        extractor.quit_driver()
