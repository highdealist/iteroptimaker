from certifi import contents
import requests
from bs4 import BeautifulSoup, Comment
import time
import re
from urllib.parse import urlparse
from typing import List, Dict, Any
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
#$end
from newspaper import Article
import threading
from ratelimit import limits
import json
import datetime
import playwright


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

DEFAULT_HEADERS = {
    'User-Agent': None,  # Will be set dynamically
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'max-age=0',
    'DNT': '1',
}

CONTENT_CLASS_PATTERN = re.compile(
    r'\b(content|main-content|post-content|entry-content|article-body|'
    r'product-description|the-content|post-entry|entry|sqs-block-content|'
    r'content-wrapper|post-body|rich-text-section|postArticle-content|'
    r'post-full-content|item-description|message-body|thread-content|'
    r'story-content|news-article-body)\b',
    re.IGNORECASE
)

def get_headers():
    """Get headers with a random user agent."""
    headers = DEFAULT_HEADERS.copy()
    headers['User-Agent'] = random.choice(USER_AGENTS)
    return headers

def fetch_article_text(url):
    article = Article(url)
    article.download()
    article.parse()

    title = article.title
    author = ', '.join(article.authors)
    pub_date = article.publish_date
    article_text = article.text

    return title, author, pub_date, article_text

# Load environment variables
load_dotenv()

# Google API keys
GOOGLE_CUSTOM_SEARCH_ENGINE_ID = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_ID')
GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY= os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY')
BRAVE_SEARCH_API_KEY = os.getenv('BRAVE_SEARCH_API_KEY')  # Brave Search API key (if available)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # Get a logger instance

# --- [Improved] More descriptive error handling ---
def initialize_apis() -> List['SearchAPI']:
    """Initializes the APIs.

    Returns:
        List[SearchAPI]: A list of initialized SearchAPI objects.

    Raises:
        ValueError: If a required environment variable for an API is not set.
    """
    if GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY is None or GOOGLE_CUSTOM_SEARCH_ENGINE_ID is None:
        raise ValueError("GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY and GOOGLE_CUSTOM_SEARCH_ENGINE_ID must be set in .env.")
    apis = [
        SearchAPI(
            "Google",
            GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY,
            "https://www.googleapis.com/customsearch/v1",
            {"cx": GOOGLE_CUSTOM_SEARCH_ENGINE_ID},
            100,
            'items',
            1,
        )
    ]
    if BRAVE_SEARCH_API_KEY:
        apis.append(SearchAPI("Brave", BRAVE_SEARCH_API_KEY, "https://api.search.brave.com/res/v1/web/search",
                            {}, 2000, 'results', 1))

    apis.append(SearchAPI("DuckDuckGo", "", "https://api.duckduckgo.com/",
                        {"format": "json"}, float('inf'), 'RelatedTopics', 0))

    return apis
# --- (end) ---


def configure_search_settings() -> Dict[str, Any]:
    """Prompts the user to enable/disable search functionality.

    Returns:
        Dict[str, Any]: A dictionary containing search settings,
                        including 'search_enabled' (bool) and, if enabled,
                        'all_search_result_data', 'search_session_counter',
                        'search_session_id', and 'apis'.
    """
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
            print(f"An error occurred: {e}. Please try again.")


class SearchProvider(ABC):
    """Abstract base class for search providers."""

    @abstractmethod
    def search(self, query: str, num_results: int) -> List['SearchResult']:
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
        self.params = params.copy()  # Create a copy to avoid modifying the original
        if api_key:
            self.params['key'] = api_key
        self.quota = quota
        self.used = 0
        self.results_path = results_path
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.user_agent_rotator = UserAgent()

    def is_within_quota(self) -> bool:
        """Checks if the API is within its usage quota."""
        return self.used < self.quota

    def search(self, query: str, num_results: int) -> List[SearchResult]:
        """Performs a search using the API."""
        logger.info(f"Searching {self.name} for: {query}")
        params = self.params.copy()
        params['q'] = query

        # Google Custom Search has a max of 10 results per request
        params['num'] = min(num_results, 10) if self.name == 'Google' else num_results
        headers = {'User-Agent': self.user_agent_rotator.random}
        try:
            response = requests.get(self.base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            self.used += 1
            self.last_request_time = time.time()
            data = response.json()

            results = []
            for item in data.get(self.results_path, []):
                url = item.get('link') or item.get('url')
                title = item.get('title') or "No title"
                snippet = item.get('snippet') or "No snippet"
                results.append(SearchResult(title, url, snippet))
            return results
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during {self.name} search: {e}")
            return []


class DuckDuckGoSearchProvider(SearchProvider):
    """Provides search functionality using DuckDuckGo."""

    def search(self, query: str, max_results: int) -> List[SearchResult]:
        """Searches DuckDuckGo and returns a list of SearchResult objects."""
        try:
            sanitized_query = self._sanitize_query(query)
            with DDGS() as ddgs:
                results = list(ddgs.text(sanitized_query, region='wt-wt', safesearch='off', timelimit='y'))[
                          :max_results]
            return [SearchResult(r['title'], r['href'], r['body']) for r in results]
        except Exception as e:
            logging.error(f"Error searching DuckDuckGo: {e}")
            return []

    def _sanitize_query(self, query: str) -> str:
        """Sanitizes the search query for DuckDuckGo."""
        query = re.sub(r'[^\w\s]', '', query)
        query = re.sub(r'\s+', ' ', query).strip()
        return query[:5000]


class WebContentExtractor:
    MAX_RETRIES = 2        # Number of retry attempts for HTTP requests
    TIMEOUT = 10           # Timeout for HTTP requests in seconds

    driver = None  # Class variable to store the driver instance

    @classmethod
    def _initializedriver(cls):
        """Initializes the Selenium WebDriver with anti-detection measures."""
        if cls.driver is not None:  # Only initialize if not already initialized
            return

# Configure Edge browser options
        edge_options = Options()
        edge_options.add_argument("--headless=new")
        edge_options.add_argument("--disable-gpu")
        edge_options.add_argument("--no-sandbox")

        # Set random user agent from predefined list
        user_agent = random.choice(USER_AGENTS)
        edge_options.add_argument(f"user-agent={user_agent}")

        # Set up the WebDriver service
        service = Service(EdgeChromiumDriverManager().install())

        # Initialize the WebDriver
        cls.driver = webdriver.Edge(service=service, options=edge_options)


        stealth(cls.driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Angle",
                fix_hairline=True,
        )

    @classmethod
    def extract_with_selenium(cls, url: str) -> str:
        """Extracts content using Selenium for dynamic content."""
        try:
            cls._initializedriver()  # Ensure driver is initialized
            cls.driver.get(url)
            WebDriverWait(cls.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            )
            html_content = cls.driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
            main_content = soup.find(['div', 'main', 'article'], class_=CONTENT_CLASS_PATTERN)
            main_content = main_content or soup.body
            main_text = main_content.get_text(separator=' ', strip=True) if main_content else ''
            return re.sub(r'\s+', ' ', main_text)
        except Exception as e:
            logging.error(f"Selenium extraction failed for {url}: {e}")
            return ""

    @classmethod
    def extract_content(cls, url: str) -> str:
        """Extracts content, handling dynamic content and fallbacks."""
        if not cls.is_valid_url(url):
            logger.error(f"Invalid URL: {url}")
            return ""
        for extractor in [cls._extract_with_requests, cls._extract_with_newspaper, cls.extract_with_selenium]:
            text = extractor(url)
            if len(text.strip()) >= 200:
                return text
        return ""

    @classmethod
    def _extract_with_requests(cls, url: str) -> str:
        """Extracts content using requests."""
        for attempt in range(1, cls.MAX_RETRIES + 1):
            try:
                headers = get_headers()
                response = requests.get(url, headers=headers, timeout=cls.TIMEOUT)
                response.raise_for_status()
                content_type = response.headers.get('Content-Type', '').lower()

                if 'text/html' not in content_type:
                    logger.warning(f"Non-HTML content returned for {url}: {content_type}")
                    return ""
                if response.headers.get('content-encoding') == 'gzip':
                    try:
                        html_content = gzip.decompress(response.content).decode('utf-8', errors='ignore')
                    except (OSError, gzip.BadGzipFile) as e:
                        logger.warning(f"Error decoding gzip content: {e}. Using raw content.")
                        html_content = response.text
                else:
                    html_content = response.text

                soup = BeautifulSoup(html_content, 'html.parser')
                return cls._extract_content_from_soup(soup)

            except requests.exceptions.RequestException as e:
                if attempt < cls.MAX_RETRIES:
                    logger.warning(f"Error with requests for {url} (attempt {attempt}): {e}. Retrying...")
                    time.sleep(2 ** attempt)
                else:
                    logger.warning(f"Error with requests for {url} after {cls.MAX_RETRIES} attempts: {e}. Giving up.")
                    return ""

    @classmethod
    def _extract_with_newspaper(cls, url: str) -> str:
        """Extracts content using newspaper3k."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            logger.warning(f"Newspaper error for {url}: {e}")
            return ""

    @staticmethod
    def _extract_content_from_soup(soup: BeautifulSoup) -> str:
        """Helper method to extract and clean content from BeautifulSoup object."""
        for element in soup(['nav', 'header', 'footer', 'aside', 'script', 'style']):
            element.decompose()

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        content = soup.find('main') or soup.find('article') or soup.find(
            'div', class_=re.compile(r'content|main-content|post-content|body|main-body|body-content|main', re.IGNORECASE))

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

    @classmethod
    def quitdriver(cls):
        """Quits the WebDriver if it is running."""
        if cls.driver is not None:
            cls.driver.quit()
            cls.driver = None

    def acquire(self):
        with self.lock:
            current = time.time()
            # Remove calls that are outside the current period
            self.calls = [call for call in self.calls if call > current - self.period]

            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (current - self.calls[0])
                time.sleep(sleep_time)
                self.calls = [call for call in self.calls if call > time.time() - self.period]

            self.calls.append(time.time())

class SearchManager:
    """Manages searches across multiple APIs and providers."""

    def __init__(self, apis: List[SearchAPI], web_search_provider: SearchProvider, max_content_length: int = 10000,
                 cache_size: int = 100):
        self.apis = apis
        self.web_search_provider = web_search_provider
        self.content_extractor = WebContentExtractor()
        self.max_content_length = max_content_length
        self.cache = {}
        self.cache_size = cache_size

    def search(self, query: str, num_results: int = 5):
        """
        Performs a search using available APIs and the web search provider.

        Args:
            query (str): The search query.
            num_results (int, optional): The maximum number of results to return.
                                          Defaults to 5.

        Returns:
            List[Dict]: A list of dictionaries, each representing a search result
                        with 'title', 'url', 'snippet', and 'content' keys.
        """
        # Define the order of APIs to try
        api_order = ["Google", "Brave", "DuckDuckGo"]

        # Try each API in order
        for api_name in api_order:
            api = next((api for api in self.apis if api.name == api_name), None)
            if api and api.is_within_quota():
                try:
                    logging.info(f"Trying {api_name} for query: {query}")
                    if search_results := api.search(query, num_results):
                        # Process the results and return
                        detailed_results = []
                        for result in search_results:
                            content = self.content_extractor.extract_content(result.url)
                            result.content = content[:self.max_content_length]
                            detailed_results.append({
                                'title': result.title,
                                'url': result.url,
                                'snippet': result.snippet,
                                'content': result.content
                            })
                        self._cache_results(query, detailed_results)
                        return detailed_results
                except Exception as e:
                    logging.error(f"Error searching {api_name}: {e}")

        # If all APIs fail, try DuckDuckGo as a last resort
        logging.info(f"Trying DuckDuckGo for query: {query}")
        duck_results = self.web_search_provider.search(query, num_results)
        detailed_results = []
        for result in duck_results:
            content = self.content_extractor.extract_content(result.url)
            detailed_results.append({
                'title': result.title,
                'url': result.url,
                'snippet': result.snippet,
                'content': content[:self.max_content_length] if content is not None else ""
            })
        self._cache_results(query, detailed_results)
        return detailed_results

    def _cache_results(self, query: str, results: List[Dict]):
        """Caches the search results."""
        self.cache[query] = results
        if len(self.cache) > self.cache_size:
            self.cache.pop(next(iter(self.cache)))


def initialize_search_manager():
    """Initializes the SearchManager with configured APIs and settings."""
    search_settings = configure_search_settings()
    if search_settings['search_enabled']:
        apis = search_settings['apis']
        return SearchManager(apis, web_search_provider=DuckDuckGoSearchProvider())
    return None


# Example tool function (from your description)
def foia_search(query: str) -> List[str]:
    """Searches FOIA.gov for the given query and returns a list of relevant content.

    Args:
        query (str): The search query.

    Returns:
        List[str]: A list of text content extracted from relevant FOIA.gov search results.
    """
    url = f"https://search.foia.gov/search?utf8=%E2%9C%93&m=true&affiliate=foia.gov&query={query.replace(' ', '+')}"
    web_content_extractor = WebContentExtractor()
    try:
        response = requests.get(url, headers=get_headers(), timeout=web_content_extractor.TIMEOUT)
        response.raise_for_status()
        html_content = response.content.decode('utf-8')
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract links to actual FOIA results (not navigation links)
        result_links = [a['href'] for a in soup.select('.result-title a') if a.has_attr('href')]

        content = []
        for link in result_links:
            try:
                if extracted_content := WebContentExtractor.extract_content(
                    link
                ):
                    content.append(extracted_content)
            except Exception as e:
                logger.error(f"Error extracting content from {link}: {e}")

        return content
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching FOIA.gov: {e}")
        return []
def save_results_to_file(results, query):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"search_results_{query.replace(' ', '_')}_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {filename}")

# Example usage
if __name__ == "__main__":
    search_manager = initialize_search_manager()
    query = "secret service corrupt"
    num_results = 15

    if search_manager:
        results = search_manager.search(query, num_results)
        for result in results:
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Snippet: {result['snippet']}")
            print(f"Content: {result['content'][:15000]}...")
            print("---")
    else:
        print("Search functionality is disabled.")

