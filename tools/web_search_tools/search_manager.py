from certifi import contents
import requests
from bs4 import BeautifulSoup, Comment
import time
import re
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv
import os
from abc import ABC, abstractmethod
from fake_useragent import UserAgent
import html2text
from duckduckgo_search import DDGS
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import gzip
from utils import log_and_handle_error
from newspaper import Article
from functools import lru_cache
from datetime import datetime, timedelta
from selenium_stealth import stealth
import asyncio
from .web_search import SearchAPI, SearchResult, DuckDuckGoSearchProvider, initialize_apis
from .content_extractor import WebContentExtractor


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

class SearchManager:
    """Manages searches across multiple APIs and providers with enhanced caching."""
    
    def __init__(self, apis: Optional[List[SearchAPI]] = None,
                 web_search_provider: Optional[DuckDuckGoSearchProvider] = None,
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
    
    def _get_cached_results(self, cache_key: str) -> Optional[List[Dict]]:
        """Get results from cache if valid."""
        if cache_key in self.cache:
            timestamp = self.cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp <= self.cache_ttl:
                return self.cache[cache_key]
        return None
    
    async def _try_api_search(self, query: str, num_results: int) -> List[SearchResult]:
        """Try searching using available APIs in order of preference."""
        for api in self.apis:
            if await api.is_within_quota():
                results = await api.search(query, num_results)
                if results:
                    return results
        
        # Fallback to DuckDuckGo if all APIs fail or are over quota
        return await self.web_search_provider.search(query, num_results)
    
    def _process_search_results(self, search_results: List[SearchResult]) -> List[Dict]:
        """Process search results and extract content safely."""
        processed_results = []
        for result in search_results:
            try:
                content = self.content_extractor.extract_content(result.url)
                if content:
                    # Truncate content if it's too long
                    content = content[:self.max_content_length]
                    result.content = content
                processed_results.append(result.to_dict())
            except Exception as e:
                logger.error(f"Error processing result {result.url}: {e}")
        return processed_results
    
    def _cache_results(self, cache_key: str, results: List[Dict]):
        """Cache search results with timestamp."""
        self.cache[cache_key] = results
        self.cache_timestamps[cache_key] = time.time()
        
        # Remove oldest entries if cache is full
        if len(self.cache) > self.cache_size:
            oldest_key = min(self.cache_timestamps.keys(),
                           key=lambda k: self.cache_timestamps[k])
            del self.cache[oldest_key]
            del self.cache_timestamps[oldest_key]
    
    def clear_expired_cache(self):
        """Clear expired cache entries."""
        current_time = time.time()
        expired_keys = [k for k, v in self.cache_timestamps.items()
                       if current_time - v > self.cache_ttl]
        for key in expired_keys:
            del self.cache[key]
            del self.cache_timestamps[key]
    
    async def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Performs a cached search using available APIs and the web search provider."""
        cache_key = f"{query}:{num_results}"
        
        # Try to get results from cache
        cached_results = self._get_cached_results(cache_key)
        if cached_results is not None:
            return cached_results
        
        # Perform new search
        search_results = await self._try_api_search(query, num_results)
        processed_results = self._process_search_results(search_results)
        
        # Cache the results
        self._cache_results(cache_key, processed_results)
        
        return processed_results

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
            print(f"An error occurred: {e}. Please try again.")

async def initialize_search_manager() -> SearchManager:
    """Initialize SearchManager with default configuration."""
    try:
        apis = []
        if GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY and GOOGLE_CUSTOM_SEARCH_ENGINE_ID:
            apis.append(SearchAPI(
                "Google",
                GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY,
                "https://www.googleapis.com/customsearch/v1",
                {"cx": GOOGLE_CUSTOM_SEARCH_ENGINE_ID},
                100,
                'items',
                1
            ))
        
        if BRAVE_SEARCH_API_KEY:
            apis.append(SearchAPI(
                "Brave",
                BRAVE_SEARCH_API_KEY,
                "https://api.search.brave.com/res/v1/web/search",
                {},
                2000,
                'results',
                1
            ))

        # DuckDuckGo is always available as fallback
        web_search_provider = DuckDuckGoSearchProvider()
        
        return SearchManager(
            apis=apis,
            web_search_provider=web_search_provider,
            max_content_length=10000,
            cache_size=100,
            cache_ttl=3600
        )
    except Exception as e:
        logger.error(f"Error initializing SearchManager: {e}")
        # Return a SearchManager with just DuckDuckGo as fallback
        return SearchManager(
            apis=[],
            web_search_provider=DuckDuckGoSearchProvider()
        )


# Example tool function (from your description)
async def foia_search(query: str) -> List[str]:
    """Searches FOIA.gov for the given query and returns a list of relevant content.

    Args:
        query (str): The search query.

    Returns:
        List[str]: A list of text content extracted from relevant FOIA.gov search results.
    """
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
        response = requests.get(url, headers=headers, timeout=web_content_extractor.TIMEOUT)
        response.raise_for_status()
        html_content = response.content.decode('utf-8')
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract links to actual FOIA results (not navigation links)
        result_links = [a['href'] for a in soup.select('.result-title a') if a.has_attr('href')]

        content = []
        for link in result_links:
            try:
                if extracted_content := await WebContentExtractor.extract_content(
                    link
                ):
                    content.append(extracted_content)
            except Exception as e:
                logger.error(f"Error extracting content from {link}: {e}")

        return content
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching FOIA.gov: {e}")
        return []

num_results = 10

# Example usage
async def main():
    search_manager = await initialize_search_manager()
    query = "test"
    num_results = 15

    if search_manager:
        results = await search_manager.search(query, num_results)
        for result in results:
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Snippet: {result['snippet']}")
            print(f"Content: {result['content'][:15000]}...")  
            print("---")
    else:
        print("Search functionality is disabled.")

asyncio.run(main())
