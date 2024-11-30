import unittest
from search_manager import SearchManager, SearchAPI, DuckDuckGoSearchProvider, SearchResult
import os
from dotenv import load_dotenv
import time # Added import for time module

load_dotenv()

GOOGLE_CUSTOM_SEARCH_ENGINE_ID = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_ID')
GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY')

class TestSearchManager(unittest.TestCase):

    def test_search_manager_initialization(self):
        """Tests that SearchManager initializes correctly."""
        search_manager = SearchManager()
        self.assertIsNotNone(search_manager)
        self.assertIsInstance(search_manager.web_search_provider, DuckDuckGoSearchProvider)
        self.assertEqual(search_manager.max_content_length, 10000)
        self.assertEqual(search_manager.cache_size, 100)
        self.assertEqual(search_manager.cache_ttl, 3600)

    def test_search_manager_with_apis(self):
        """Tests that SearchManager initializes with APIs correctly."""
        if GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY and GOOGLE_CUSTOM_SEARCH_ENGINE_ID:
            apis = [
                SearchAPI(
                    "Google",
                    GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY,
                    "https://www.googleapis.com/customsearch/v1",
                    {"cx": GOOGLE_CUSTOM_SEARCH_ENGINE_ID},
                    100,
                    'items',
                    1
                )
            ]
            search_manager = SearchManager(apis=apis)
            self.assertEqual(len(search_manager.apis), 1)
            self.assertIsInstance(search_manager.apis[0], SearchAPI)
        else:
            print("Skipping test_search_manager_with_apis as Google API keys are not available.")

    def test_search_manager_search_empty_query(self):
        """Tests that search returns an empty list for an empty query."""
        search_manager = SearchManager()
        results = search_manager.search("")
        self.assertEqual(results, [])

    def test_search_manager_search_duckduckgo(self):
        """Tests that search returns results from DuckDuckGo."""
        search_manager = SearchManager()
        results = search_manager.search("Python programming", num_results=3)
        self.assertGreater(len(results), 0)
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('title', result)
            self.assertIn('url', result)
            self.assertIn('snippet', result)
            self.assertIn('content', result)

    def test_search_manager_search_with_apis(self):
        """Tests that search returns results from APIs if available."""
        if GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY and GOOGLE_CUSTOM_SEARCH_ENGINE_ID:
            apis = [
                SearchAPI(
                    "Google",
                    GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY,
                    "https://www.googleapis.com/customsearch/v1",
                    {"cx": GOOGLE_CUSTOM_SEARCH_ENGINE_ID},
                    100,
                    'items',
                    1
                )
            ]
            search_manager = SearchManager(apis=apis)
            results = search_manager.search("Python programming", num_results=3)
            self.assertGreater(len(results), 0)
            for result in results:
                self.assertIsInstance(result, dict)
                self.assertIn('title', result)
                self.assertIn('url', result)
                self.assertIn('snippet', result)
                self.assertIn('content', result)
        else:
            print("Skipping test_search_manager_search_with_apis as Google API keys are not available.")

    def test_search_manager_cache(self):
        """Tests that search results are cached."""
        search_manager = SearchManager(cache_size=5, cache_ttl=1)
        query = "Python programming"
        results1 = search_manager.search(query, num_results=3)
        results2 = search_manager.search(query, num_results=3)
        self.assertEqual(results1, results2)

    def test_search_manager_cache_expiration(self):
        """Tests that cached results expire."""
        search_manager = SearchManager(cache_size=5, cache_ttl=1)
        query = "Python programming"
        results1 = search_manager.search(query, num_results=3)
        time.sleep(2) # Using time.sleep here
        results2 = search_manager.search(query, num_results=3)
        self.assertNotEqual(results1, results2)

    def test_search_api_is_within_quota(self):
        """Tests that SearchAPI correctly checks quota."""
        api = SearchAPI("TestAPI", "test_key", "https://test.com", {}, 10, 'results', 1)
        self.assertTrue(api.is_within_quota())
        for _ in range(10):
            api.search("test query", 1)
        self.assertTrue(api.is_within_quota())
        api.search("test query", 1)
        self.assertFalse(api.is_within_quota())

    def test_search_api_respect_rate_limit(self):
        """Tests that SearchAPI respects rate limit."""
        api = SearchAPI("TestAPI", "test_key", "https://test.com", {}, 10, 'results', 1)
        start_time = time.time() # Using time.time here
        api.respect_rate_limit()
        end_time = time.time() # Using time.time here
        self.assertGreaterEqual(end_time - start_time, 1)

    def test_search_result_initialization(self):
        """Tests that SearchResult initializes correctly."""
        search_result = SearchResult("Test Title", "https://test.com", "Test Snippet", "Test Content")
        self.assertEqual(search_result.title, "Test Title")
        self.assertEqual(search_result.url, "https://test.com")
        self.assertEqual(search_result.snippet, "Test Snippet")
        self.assertEqual(search_result.content, "Test Content")

if __name__ == '__main__':
    unittest.main()
