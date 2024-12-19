import requests
from bs4 import BeautifulSoup, Comment
import time
import re
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional, Union
import logging
from dotenv import load_dotenv
import os
from abc import ABC, abstractmethod
from fake_useragent import UserAgent
import html2text
from duckduckgo_search import DDGS
import random
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium_stealth import stealth
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import gzip
from newspaper import Article
from functools import lru_cache
from datetime import datetime, timedelta
import asyncio
import aiohttp
from aiohttp import ClientSession
from selenium.common.exceptions import TimeoutException as SeleniumTimeoutException
from requests.exceptions import Timeout as RequestsTimeout
from requests.exceptions import ConnectionError as RequestsConnectionError
import json

# Load environment variables
load_dotenv()

# Google API keys
GOOGLE_CUSTOM_SEARCH_ENGINE_ID = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_ID')
GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY')
BRAVE_SEARCH_API_KEY = os.getenv('BRAVE_SEARCH_API_KEY')  # Brave Search API key (if available)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_search_tool.log"),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)  # Get a logger instance

# --- Constants ---
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
EXPONENTIAL_BACKOFF_BASE = 2
TIMEOUT = 10

def fetch_article_text(url):
    """Fetches article title, author, publication date, and text using newspaper3k."""
    article = Article(url)
    article.download()
    article.parse()

    title = article.title
    author = ', '.join(article.authors)
    pub_date = article.publish_date
    article_text = article.text

    return title, author, pub_date, article_text

def initialize_apis() -> List['SearchAPI']:
    """Initializes the APIs with enhanced error handling.

    Returns:
        List[SearchAPI]: A list of initialized SearchAPI objects.

    Raises:
        ValueError: If a required environment variable for an API is not set.
    """
    apis = []
    if GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY and GOOGLE_CUSTOM_SEARCH_ENGINE_ID:
        apis.append(
            SearchAPI(
                "Google",
                GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY,
                "https://www.googleapis.com/customsearch/v1",
                {"cx": GOOGLE_CUSTOM_SEARCH_ENGINE_ID},
                100,
                'items',
                1,
            )
        )
    else:
        logger.warning("Google Custom Search API key or Engine ID not set. Google Search will not be available.")

    if BRAVE_SEARCH_API_KEY:
        apis.append(
            SearchAPI(
                "Brave",
                BRAVE_SEARCH_API_KEY,
                "https://api.search.brave.com/res/v1/web/search",
                {},
                2000,
                'results',
                1,
            )
        )
    else:
        logger.info("Brave Search API key not set. Brave Search will not be available.")

    apis.append(
        SearchAPI(
            "DuckDuckGo",
            "",
            "https://api.duckduckgo.com/",
            {"format": "json"},
            float('inf'),
            'RelatedTopics',
            0,
        )
    )

    return apis

def configure_search_settings() -> Dict[str, Any]:
    """Prompts the user to enable/disable search functionality."""
    while True:
        try:
            user_input = input("Do you want to enable search functionality? (Y/N): ").strip().lower()
            if user_input == 'y':
                return {
                    'search_enabled': True,
                    'all_search_result_data': {},
                    'search_session_counter': 0,
                    'search_session_id': 0,
                    'apis': initialize_apis(),
                }
            elif user_input == 'n':
                return {'search_enabled': False}
            else:
                print("Invalid input. Please enter Y or N.")
        except Exception as e:
            logger.error(f"An error occurred: {e}. Please try again.", exc_info=True)

class SearchProvider(ABC):
    """Abstract base class for search providers."""

    @abstractmethod
    async def search(self, query: str, num_results: int) -> List['SearchResult']:
        """Perform a search and return a list of SearchResult objects."""
        pass

class SearchResult:
    """Represents a single search result."""

    def __init__(self, title: str, url: str, snippet: str, content: str = ""):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.content = content

class SearchAPI(SearchProvider):
    """Represents a search API with rate limiting and quota management."""

    def __init__(self, name: str, api_key: str, base_url: str, params: dict, quota: int, results_path: str,
                 rate_limit: int):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.params = params.copy()
        if api_key:
            self.params['key'] = api_key
        self.quota = quota
        self.used = 0
        self.results_path = results_path
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.user_agent_rotator = UserAgent()
        self.backoff_factor = EXPONENTIAL_BACKOFF_BASE

    def is_within_quota(self) -> bool:
        """Checks if the API is within its usage quota."""
        return self.used < self.quota

    def respect_rate_limit(self):
        """Pauses execution to respect the API's rate limit."""
        time_since_last_request = time.time() - self.last_request_time
        if time_since_last_request < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last_request)

    async def search(self, query: str, num_results: int) -> List[SearchResult]:
        """Performs a search using the API with exponential backoff and retry."""
        self.respect_rate_limit()
        logger.info(f"Searching {self.name} for: {query}")
        params = self.params.copy()
        params['q'] = query

        params['num'] = min(num_results, 10) if self.name == 'Google' else num_results
        headers = {'User-Agent': self.user_agent_rotator.random}

        for attempt in range(MAX_RETRIES):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.base_url, params=params, headers=headers, timeout=TIMEOUT) as response:
                        response.raise_for_status()
                        self.used += 1
                        self.last_request_time = time.time()
                        data = await response.json()

                        results = []
                        for item in data.get(self.results_path, []):
                            url = item.get('link') or item.get('url')
                            title = item.get('title') or "No title"
                            snippet = item.get('snippet') or "No snippet"
                            results.append(SearchResult(title, url, snippet))
                        return results
            except (aiohttp.ClientError, RequestsTimeout, json.JSONDecodeError) as e:
                logger.warning(
                    f"Attempt {attempt + 1} failed for {self.name} search: {e}. Retrying in {self.backoff_factor ** attempt} seconds."
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(self.backoff_factor ** attempt)
                else:
                    logger.error(f"Error during {self.name} search after multiple retries: {e}", exc_info=True)
                    return []
            except Exception as e:
                logger.error(f"An unexpected error occurred during {self.name} search: {e}", exc_info=True)
                return []

class DuckDuckGoSearchProvider(SearchProvider):
    """Provides search functionality using DuckDuckGo."""

    async def search(self, query: str, max_results: int) -> List[SearchResult]:
        """Searches DuckDuckGo and returns a list of SearchResult objects."""
        try:
            sanitized_query = self._sanitize_query(query)
            async with DDGS() as ddgs:
                results = [
                    r async for r in ddgs.text(
                        sanitized_query,
                        region='wt-wt',
                        safesearch='off',
                        timelimit='y',
                        max_results=max_results
                    )
                ]
            return [SearchResult(r['title'], r['href'], r['body']) for r in results]
        except Exception as e:
            logger.error(f"Error searching DuckDuckGo: {e}", exc_info=True)
            return []

    def _sanitize_query(self, query: str) -> str:
        """Sanitizes the search query for DuckDuckGo."""
        query = re.sub(r'[^\w\s]', '', query)
        query = re.sub(r'\s+', ' ', query).strip()
        return query[:5000]

class WebContentExtractor:
    """Extracts web content from a given URL."""
    MAX_RETRIES = 3
    TIMEOUT = 10
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/91.0.4472.80 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (Android 11; Mobile; rv:68.0) Gecko/68.0 Firefox/88.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36 OPR/78.0.4093.147',
    ]
    _driver = None

    @classmethod
    def get_driver(cls):
        """Returns the shared WebDriver instance."""
        if cls._driver is None:
            cls._initialize_driver()
        return cls._driver

    @classmethod
    def _initialize_driver(cls):
        """Initializes the Selenium WebDriver with anti-detection measures."""
        edge_options = Options()
        edge_options.add_argument("--headless=new")
        edge_options.add_argument("--disable-gpu")
        edge_options.add_argument("--no-sandbox")

        user_agent = random.choice(cls.USER_AGENTS)
        edge_options.add_argument(f"user-agent={user_agent}")

        cls._driver = webdriver.Edge(
            service=Service(EdgeChromiumDriverManager().install()),
            options=edge_options
        )

        stealth(
            cls._driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Angle",
            fix_hairline=True,
        )

    @classmethod
    def quit_driver(cls):
        """Quits the WebDriver if it is running."""
        if cls._driver is None:
            return
        cls._driver.quit()
        cls._driver = None

    @classmethod
    async def _extract_with_requests_async(cls, session: ClientSession, url: str) -> str:
        """Asynchronously extracts content using aiohttp with retries."""
        for attempt in range(1, cls.MAX_RETRIES + 1):
            try:
                headers = {
                    'User-Agent': random.choice(cls.USER_AGENTS),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Cache-Control': 'max-age=0',
                    'DNT': '1',
                }
                async with session.get(url, headers=headers, timeout=cls.TIMEOUT) as response:
                    response.raise_for_status()
                    content_type = response.headers.get('Content-Type', '').lower()

                    if 'text/html' not in content_type:
                        logger.warning(f"Non-HTML content returned for {url}: {content_type}")
                        return ""

                    if response.headers.get('content-encoding') == 'gzip':
                        html_content = await response.text(errors='ignore')
                    else:
                        html_content = await response.text(errors='ignore')

                    soup = BeautifulSoup(html_content, 'html.parser')
                    return cls._extract_content_from_soup(soup)

            except (aiohttp.ClientError, RequestsTimeout, aiohttp.ContentTypeError) as e:
                if attempt < cls.MAX_RETRIES:
                    logger.warning(
                        f"Error with requests for {url} (attempt {attempt}): {e}. Retrying in {2 ** attempt} seconds..."
                    )
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(
                        f"Error with requests for {url} after {cls.MAX_RETRIES} attempts: {e}",
                        exc_info=True
                    )
                    return ""

    @classmethod
    async def _extract_with_newspaper_async(cls, url: str) -> str:
        """Asynchronously extracts content using newspaper3k."""
        try:
            loop = asyncio.get_event_loop()
            article = Article(url)
            await loop.run_in_executor(None, article.download)
            await loop.run_in_executor(None, article.parse)
            return article.text
        except Exception as e:
            logger.warning(f"Newspaper error for {url}: {e}", exc_info=True)
            return ""

    @classmethod
    async def extract_with_selenium_async(cls, url: str) -> str:
        """Asynchronously extracts content using Selenium (for dynamic content)."""
        driver = cls.get_driver()
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, driver.get, url)

            await loop.run_in_executor(
                None,
                WebDriverWait(driver, 10).until,
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            )

            html_content = driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
            main_content = soup.find(
                ['div', 'main', 'article'],
                class_=re.compile(
                    r'\b(content|main-content|post-content|entry-content|article-body|'
                    r'product-description|the-content|post-entry|entry|sqs-block-content|'
                    r'content-wrapper|post-body|rich-text-section|postArticle-content|'
                    r'post-full-content|item-description|message-body|thread-content|'
                    r'story-content|news-article-body)\b',
                    re.IGNORECASE
                )
            ) or soup.body
            main_text = main_content.get_text(separator=' ', strip=True) if main_content else ''
            return re.sub(r'\s+', ' ', main_text)
        except SeleniumTimeoutException as e:
            logger.error(f"Selenium timeout for {url}: {e}", exc_info=True)
            return ""
        except Exception as e:
            logger.error(f"Selenium extraction failed for {url}: {e}", exc_info=True)
            return ""

    @classmethod
    async def extract_content_async(cls, url: str) -> str:
        """Asynchronously extracts content with fallbacks and improved error handling."""
        if not cls.is_valid_url(url):
            logger.error(f"Invalid URL: {url}")
            return ""

        async with aiohttp.ClientSession() as session:
            for extractor in [cls._extract_with_requests_async, cls._extract_with_newspaper_async,
                              cls.extract_with_selenium_async]:
                if extractor == cls._extract_with_requests_async:
                    text = await extractor(session, url)
                else:
                    text = await extractor(url)
                if len(text.strip()) >= 200:
                    return text
        return ""

    @staticmethod
    def _extract_content_from_soup(soup: BeautifulSoup) -> str:
        """Helper method to extract and clean content from BeautifulSoup object."""
        for element in soup(['nav', 'header', 'footer', 'aside', 'script', 'style']):
            element.decompose()

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        content = soup.find('main') or soup.find('article') or soup.find(
            'div',
            class_=re.compile(
                r'content|main-content|post-content|body|main-body|body-content|main',
                re.IGNORECASE
            )
        )

        if not content:
            content = soup.body

        if content:
            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            text = h.handle(str(content))

            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            return text
        else:
            return ""

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Checks if a URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

class SearchManager:
    """Manages searches across multiple APIs and providers with enhanced caching."""

    def __init__(self, apis: Optional[List[SearchAPI]] = None,
                 web_search_provider: Optional[SearchProvider] = None,
                 max_content_length: int = 10000,
                 cache_size: int = 100,
                 cache_ttl: int = 3600):
        """Initialize SearchManager with flexible configuration."""
        self.apis = apis or []
        self.web_search_provider = web_search_provider or DuckDuckGoSearchProvider()
        self.content_extractor = WebContentExtractor()
        self.max_content_length = max_content_length
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl

    async def _process_search_results_async(self, search_results: List[SearchResult]) -> List[Dict[str, str]]:
        """Process search results and extract content safely and asynchronously."""
        detailed_results = []
        tasks = []
        for result in search_results:
            task = asyncio.create_task(self._process_single_result(result))
            tasks.append(task)

        detailed_results = await asyncio.gather(*tasks)
        return [res for res in detailed_results if res]

    async def _process_single_result(self, result: SearchResult) -> Optional[Dict[str, str]]:
        """Helper function to process a single search result."""
        try:
            content = await self.content_extractor.extract_content_async(result.url)
            return {
                'title': result.title[:500],
                'url': result.url[:1000],
                'snippet': result.snippet[:1000],
                'content': content[:self.max_content_length] if content else ""
            }
        except Exception as e:
            logger.error(f"Error processing result {result.url}: {e}", exc_info=True)
            return None

    async def search_async(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """
        Performs a cached search using available APIs and the web search provider asynchronously.
        """
        if not query.strip():
            return []

        cache_key = f"{query}:{num_results}"
        cached_results = self._get_cached_results(cache_key)
        if cached_results:
            return cached_results

        detailed_results = await self._try_api_search_async(query, num_results)

        if not detailed_results:
            logger.info(f"Falling back to DuckDuckGo for query: {query}")
            duck_results = await self.web_search_provider.search(query, num_results)
            detailed_results = await self._process_search_results_async(duck_results)

        if detailed_results:
            self._cache_results(cache_key, detailed_results)

        return detailed_results

    async def _try_api_search_async(self, query: str, num_results: int) -> Optional[List[Dict[str, str]]]:
        """Try searching using available APIs in order of preference asynchronously."""
        api_order = ["Google", "Brave", "DuckDuckGo"]

        for api_name in api_order:
            api = next((api for api in self.apis if api.name == api_name), None)
            if api and api.is_within_quota():
                try:
                    logger.info(f"Trying {api_name} for query: {query}")
                    search_results = await api.search(query, num_results)
                    if search_results:
                        return await self._process_search_results_async(search_results)
                except Exception as e:
                    logger.error(f"Error searching {api_name}: {e}", exc_info=True)
                    continue
        return None

    def _get_cached_results(self, cache_key: str) -> Optional[List[Dict[str, str]]]:
        """Get results from cache if valid."""
        if cache_key in self.cache:
            timestamp = self.cache_timestamps.get(cache_key)
            if timestamp and (datetime.now() - timestamp) < timedelta(seconds=self.cache_ttl):
                logger.info(f"Returning cached results for key: {cache_key}")
                return self.cache[cache_key]
        return None

    def _cache_results(self, cache_key: str, results: List[Dict]):
        """Cache search results with timestamp."""
        self.cache[cache_key] = results
        self.cache_timestamps[cache_key] = datetime.now()

        while len(self.cache) > self.cache_size:
            oldest_key = min(self.cache_timestamps.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.cache_timestamps[oldest_key]

    def clear_expired_cache(self):
        """Clear expired cache entries."""
        current_time = datetime.now()
        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items()
            if (current_time - timestamp).total_seconds() > self.cache_ttl
        ]
        for key in expired_keys:
            del self.cache[key]
            del self.cache_timestamps[key]

def initialize_search_manager() -> SearchManager:
    """Initialize SearchManager with default configuration."""
    try:
        apis = initialize_apis()
        web_search_provider = DuckDuckGoSearchProvider()

        return SearchManager(
            apis=apis,
            web_search_provider=web_search_provider,
            max_content_length=10000,
            cache_size=100,
            cache_ttl=3600
        )
    except Exception as e:
        logger.error(f"Error initializing SearchManager: {e}", exc_info=True)
        return SearchManager(
            apis=[],
            web_search_provider=DuckDuckGoSearchProvider()
        )

async def foia_search(query: str) -> List[str]:
    """Searches FOIA.gov for the given query asynchronously and returns a list of relevant content."""
    url = f"https://search.foia.gov/search?utf8=%E2%9C%93&m=true&affiliate=foia.gov&query={query.replace(' ', '+')}"
    web_content_extractor = WebContentExtractor()
    headers = {
        'User-Agent': random.choice(web_content_extractor.USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
        'DNT': '1',
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=web_content_extractor.TIMEOUT) as response:
                response.raise_for_status()
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')

                result_links = [a['href'] for a in soup.select('.result-title a') if a.has_attr('href')]

                content = []
                tasks = []
                for link in result_links:
                    task = asyncio.create_task(
                        WebContentExtractor.extract_content_async(link)
                    )
                    tasks.append(task)

                extracted_contents = await asyncio.gather(*tasks)
                for extracted_content in extracted_contents:
                    if extracted_content:
                        content.append(extracted_content)

                return content
    except (aiohttp.ClientError, RequestsTimeout) as e:
        logger.error(f"Error searching FOIA.gov: {e}", exc_info=True)
        return []

async def main():
    """Main function to demonstrate asynchronous search."""
    search_manager = initialize_search_manager()
    # query = "cyclical agential workflows"
    query = input("Enter your search query: ")
    num_results = 5

    if search_manager:
        results = await search_manager.search_async(query, num_results)
        for result in results:
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Snippet: {result['snippet']}")
            print(f"Content: {result['content'][:15000]}...")
            print("---")
    else:
        print("Search functionality is disabled.")

    # Example usage of foia_search
    # foia_results = await foia_search("FBI")
    # print("\nFOIA Search Results:")
    # for content in foia_results:
    #     print(content[:200])  # Print first 200 characters
    #     print("---")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        WebContentExtractor.quit_driver()