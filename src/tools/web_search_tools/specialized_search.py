import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Optional
import arxiv
from datetime import datetime, timedelta
from content_extractor import WebContentExtractor
from web_search_tool_v2 import SearchResult
import asyncio
from aiohttp import ClientSession
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FOIASearchProvider:
    """Provider for searching FOIA.gov records."""
    
    BASE_URL = "https://www.foia.gov/api/search"
    
    def __init__(self):
        self.content_extractor = WebContentExtractor()
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search FOIA.gov for records matching the query.
        
        Args:
            query (str): The search query
            max_results (int): Maximum number of results to return
            
        Returns:
            List[SearchResult]: List of search results
        """
        try:
            # FOIA.gov API parameters
            params = {
                "q": query,
                "size": max_results,
                "from": 0,
                "sort": "relevance",
            }
            
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("results", [])[:max_results]:
                title = item.get("title", "No Title")
                url = item.get("url") or f"https://www.foia.gov/request/{item.get('id')}"
                snippet = item.get("description", "No description available")
                
                # Try to extract content if URL is available
                content = ""
                if "url" in item:
                    content = self.content_extractor.extract_content(item["url"]) or ""
                
                results.append(SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=content
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching FOIA.gov: {e}")
            return []

class ArXivSearchProvider:
    """Provider for searching arXiv papers."""
    
    def __init__(self):
        self.client = arxiv.Client()
    
    def get_latest_papers(self, category: str = None, max_results: int = 10,
                              days: int = 7) -> List[SearchResult]:
        """Get the latest arXiv papers, optionally filtered by category.
        
        Args:
            category (str, optional): arXiv category (e.g., 'cs.AI', 'physics')
            max_results (int): Maximum number of results to return
            days (int): Number of past days to search
            
        Returns:
            List[SearchResult]: List of search results
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Build search query
            search_query = f"submittedDate:[{start_date.strftime('%Y%m%d')}* TO {end_date.strftime('%Y%m%d')}*]"
            if category:
                search_query += f" AND cat:{category}"
            
            # Create search
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            results = []
            for paper in self.client.results(search):
                # Extract authors
                authors = ", ".join([author.name for author in paper.authors])
                
                # Create content combining abstract and metadata
                content = f"Title: {paper.title}\n\n"
                content += f"Authors: {authors}\n\n"
                content += f"Published: {paper.published}\n"
                content += f"Updated: {paper.updated}\n"
                content += f"DOI: {paper.doi}\n" if paper.doi else ""
                content += f"Primary Category: {paper.primary_category}\n"
                content += f"Categories: {', '.join(paper.categories)}\n\n"
                content += f"Abstract:\n{paper.summary}\n\n"
                content += f"PDF URL: {paper.pdf_url}\n"
                
                results.append(SearchResult(
                    title=paper.title,
                    url=paper.entry_id,
                    snippet=paper.summary[:200] + "...",
                    content=content
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching arXiv papers: {e}")
            return []
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search arXiv papers by query.
        
        Args:
            query (str): The search query
            max_results (int): Maximum number of results to return
            
        Returns:
            List[SearchResult]: List of search results
        """
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in self.client.results(search):
                # Extract authors
                authors = ", ".join([author.name for author in paper.authors])
                
                # Create content combining abstract and metadata
                content = f"Title: {paper.title}\n\n"
                content += f"Authors: {authors}\n\n"
                content += f"Published: {paper.published}\n"
                content += f"Updated: {paper.updated}\n"
                content += f"DOI: {paper.doi}\n" if paper.doi else ""
                content += f"Primary Category: {paper.primary_category}\n"
                content += f"Categories: {', '.join(paper.categories)}\n\n"
                content += f"Abstract:\n{paper.summary}\n\n"
                content += f"PDF URL: {paper.pdf_url}\n"
                
                results.append(SearchResult(
                    title=paper.title,
                    url=paper.entry_id,
                    snippet=paper.summary[:200] + "...",
                    content=content
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []

#Example usage
# Initialize search providers
foia_search_provider = FOIASearchProvider()
arxiv_search_provider = ArXivSearchProvider()

# Search FOIA.gov for records
foia_results = foia_search_provider.search("vaccine")
logger.info(f"FOIA.gov search results: {foia_results}")

# Get latest arXiv papers
latest_papers = arxiv_search_provider.get_latest_papers(category="cs.AI", max_results=5)
logger.info(f"Latest arXiv papers: {latest_papers}")
