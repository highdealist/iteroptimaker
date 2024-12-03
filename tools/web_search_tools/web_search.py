from typing import List, Dict, Any, Optional, Union
import requests
import time
import os
import logging
from abc import ABC, abstractmethod
from fake_useragent import UserAgent
from duckduckgo_search import DDGS
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google API keys
GOOGLE_CUSTOM_SEARCH_ENGINE_ID = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_ID')
GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY')
BRAVE_SEARCH_API_KEY = os.getenv('BRAVE_SEARCH_API_KEY')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]):
        return cls(data['title'], data['url'], data['snippet'], data['content'])
    
    def to_dict(self):
        return {
            'title': self.title,
            'url': self.url,
            'snippet': self.snippet,
            'content': self.content
        }

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
    
    async def is_within_quota(self) -> bool:
        """Checks if the API is within its usage quota."""
        return self.used < self.quota
    
    async def respect_rate_limit(self):
        """Pauses execution to respect the API's rate limit."""
        time_since_last_request = time.time() - self.last_request_time
        if time_since_last_request < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last_request)
    
    async def search(self, query: str, num_results: int) -> List[SearchResult]:
        """Performs a search using the API."""
        await self.respect_rate_limit()
        logger.info(f"Searching {self.name} for: {query}")
        params = self.params.copy()
        params['q'] = query
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
            logger.error(f"Error searching {self.name}: {e}")
            return []

class DuckDuckGoSearchProvider(SearchProvider):
    """Provides search functionality using DuckDuckGo."""
    
    async def search(self, query: str, max_results: int) -> List[SearchResult]:
        """Searches DuckDuckGo and returns a list of SearchResult objects."""
        try:
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=max_results):
                    result = SearchResult(
                        title=r.get('title', ''),
                        url=r.get('link', ''),
                        snippet=r.get('body', '')
                    )
                    results.append(result)
                return results
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitizes the search query for DuckDuckGo."""
        return query.strip()

def initialize_apis() -> List[SearchAPI]:
    """Initializes the APIs."""
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
