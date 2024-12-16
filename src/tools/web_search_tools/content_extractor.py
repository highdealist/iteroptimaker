import os
import re
import time
from typing import Optional, Tuple
from argparse import ArgumentParser
from bs4 import BeautifulSoup, Comment
from fake_useragent import UserAgent
from msedge.selenium_tools import Edge, Service as EdgeService, EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from newspaper import Article
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_article_text(url: str) -> Tuple[str, str, Optional[str], str]:
    """Fetch article text and metadata using newspaper3k.
    
    Args:
        url (str): The URL of the article to fetch
        
    Returns:
        Tuple[str, str, Optional[str], str]: (title, author, publish_date, article_text)
    """
    article = Article(url)
    article.download()
    article.parse()
    
    title = article.title
    author = ', '.join(article.authors)
    pub_date = article.publish_date
    article_text = article.text
    
    return title, author, pub_date, article_text

class WebContentExtractor:
    """Extracts web content from a given URL with improved error handling and retry logic."""
    
    _driver = None
    
    def __init__(self, max_retries: int = 3, timeout: int = 10):
        self.max_retries = max_retries
        self.timeout = timeout
        self._initialize_driver()
    
    @classmethod
    def get_driver(cls):
        """Returns the shared WebDriver instance."""
        if cls._driver is None:
            cls._initialize_driver()
        return cls._driver
    
    @classmethod
    def _initialize_driver(cls):
        """Initializes Microsoft Edge WebDriver with enhanced anti-detection measures."""
        edge_options = EdgeOptions()
        edge_options.use_chromium = True  # Important for compatibility
        
        service = EdgeService(executable_path="C:\\webdrivers\\edgedriver_win64\\msedgedriver.exe")
        
        cls._driver = webdriver.Edge(service=service, options=edge_options)
        
    @classmethod
    def quit_driver(cls):
        """Quits the WebDriver."""
        if cls._driver:
            cls._driver.quit()
            cls._driver = None
    
    def __del__(self):
        """Ensure proper cleanup of WebDriver."""
        self.quit_driver()
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Checks if a URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def _extract_content_from_soup(soup: BeautifulSoup) -> str:
        """Helper method to extract and clean content from BeautifulSoup object."""
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Remove comments
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_with_requests(self, url: str) -> Optional[str]:
        """Extracts content using requests."""
        try:
            headers = {'User-Agent': UserAgent().random}
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            return self._extract_content_from_soup(soup)
        except Exception as e:
            logger.error(f"Error extracting content with requests: {e}")
            return None
    
    def _extract_with_newspaper(self, url: str) -> Optional[str]:
        """Extracts content using newspaper3k."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text.strip()
        except Exception as e:
            logger.error(f"Error extracting content with newspaper3k: {e}")
            return None
    
    def extract_with_selenium(self, url: str) -> Optional[str]:
        """Extracts content using Selenium with better error handling and wait conditions."""
        try:
            driver = self.get_driver()
            driver.get(url)
            
            # Wait for body to be present
            WebDriverWait(driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page source and parse with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            return self._extract_content_from_soup(soup)
        except Exception as e:
            logger.error(f"Error extracting content with Selenium: {e}")
            return None
    
    def extract_content(self, url: str, retry_count: int = 0) -> Optional[str]:
        """Extract content with automatic retry and fallback mechanisms.
        
        The extraction methods are tried in the following order:
        1. requests with BeautifulSoup (fastest, works for simple pages)
        2. newspaper3k (best for article content)
        3. Selenium (best for dynamic content)
        """
        if not url.startswith(["http://", "https://"]):
            url = "https://" + url
        else:
            url = url
        
        if not self.is_valid_url(url):
            url = input("URL Invalid. Enter URL and press Enter")
            if not self.is_valid_url(url):
                logger.error(f"Invalid URL: {url}")
                return None
        
        # Check content type
        try:
            response = requests.head(url, allow_redirects=True)
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                logger.warning(f"URL may not be HTML content (type: {content_type})")
        except Exception as e:
            logger.warning(f"Could not determine content type: {e}")
        
        # Try different extraction methods in order
        content = None
        
        # Try requests first (fastest)
        content = self._extract_with_requests(url)
        if content and len(content.strip()) >= 200:
            return content
        
        # Try newspaper3k second (good for articles)
        content = self._extract_with_newspaper(url)
        if content and len(content.strip()) >= 200:
            return content
        
        # Finally, try Selenium (best for dynamic content)
        content = self.extract_with_selenium(url)
        if content and len(content.strip()) >= 200:
            return content
        
        # If all methods fail and we haven't exceeded max retries
        if retry_count < self.max_retries:
            logger.info(f"Retrying content extraction for {url} (attempt {retry_count + 1})")
            return self.extract_content(url, retry_count + 1)
        
        logger.error(f"All content extraction methods failed for {url}")
        return None

if __name__ == "__main__":
    # Create argument parser
    parser = ArgumentParser(description='Extract content from a web page.')
    parser.add_argument('--url', type=str, help='URL of the web page to extract content from')
    parser.add_argument('--timeout', type=int, default=10, help='Timeout in seconds for requests (default: 10)')
    parser.add_argument('--max-retries', type=int, default=1, help='Maximum number of retry attempts (default: 1)')
    
    args = parser.parse_args()
    
    # If no URL provided, prompt for it
    if not args.url:
        url = input("Please enter the URL to extract content from: ")
    else:
        url = args.url
    
    # Create extractor instance
    extractor = WebContentExtractor(max_retries=args.max_retries, timeout=args.timeout)
    
    try:
        # Extract content
        content = extractor.extract_content(args.url)
        
        if content:
            print("\nExtracted Content:")
            print("-" * 80)
            print(content)
            print("-" * 80)
            
            # Try to get additional metadata using fetch_article_text
            try:
                title, author, pub_date, _ = fetch_article_text(args.url)
                print("\nMetadata:")
                print(f"Title: {title}")
                print(f"Author: {author}")
                print(f"Publish Date: {pub_date}")
            except Exception as e:
                logger.warning(f"Could not fetch article metadata: {e}")
        else:
            print("Failed to extract content from the URL.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    finally:
        # Cleanup
        extractor.quit_driver()