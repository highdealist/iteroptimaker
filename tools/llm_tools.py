"""LLM-compatible tools using the @tool decorator pattern."""
from typing import List, Dict, Optional, Any
from langchain.tools import tool
from .specialized_search import FOIASearchProvider, ArXivSearchProvider
import asyncio
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SearchResult:
    """Standardized search result format for LLM consumption."""
    title: str
    url: str
    snippet: str
    content: str
    metadata: Dict[str, Any]

class AsyncRunner:
    """Helper class to run async functions in sync context."""
    @staticmethod
    def run(coro):
        return asyncio.get_event_loop().run_until_complete(coro)

@tool
def search_foia(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Search FOIA.gov for government records and documents.
    
    Args:
        query: The search query to find FOIA records
        max_results: Maximum number of results to return (default: 10)
    
    Returns:
        List of dictionaries containing:
        - title: Document title
        - url: Document URL
        - snippet: Brief description
        - content: Full document content if available
    """
    provider = FOIASearchProvider()
    results = AsyncRunner.run(provider.search(query, max_results))
    
    return [
        {
            "title": r.title,
            "url": r.url,
            "snippet": r.snippet,
            "content": r.content
        }
        for r in results
    ]

@tool
def search_arxiv(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Search arXiv for scientific papers.
    
    Args:
        query: The search query to find papers
        max_results: Maximum number of results to return (default: 10)
    
    Returns:
        List of dictionaries containing:
        - title: Paper title
        - url: arXiv URL
        - snippet: Abstract preview
        - content: Full paper details including abstract and metadata
    """
    provider = ArXivSearchProvider()
    results = AsyncRunner.run(provider.search(query, max_results))
    
    return [
        {
            "title": r.title,
            "url": r.url,
            "snippet": r.snippet,
            "content": r.content
        }
        for r in results
    ]

@tool
def get_latest_arxiv_papers(
    category: Optional[str] = None,
    max_results: int = 10,
    days: int = 7
) -> List[Dict[str, Any]]:
    """
    Get the latest papers from arXiv, optionally filtered by category.
    
    Args:
        category: arXiv category (e.g., 'cs.AI', 'physics') (optional)
        max_results: Maximum number of results to return (default: 10)
        days: Number of past days to search (default: 7)
    
    Returns:
        List of dictionaries containing:
        - title: Paper title
        - url: arXiv URL
        - authors: List of author names
        - published_date: Publication date
        - updated_date: Last update date
        - categories: List of arXiv categories
        - abstract: Full paper abstract
        - pdf_url: Direct link to PDF
    """
    provider = ArXivSearchProvider()
    results = AsyncRunner.run(provider.get_latest_papers(
        category=category,
        max_results=max_results,
        days=days
    ))
    
    formatted_results = []
    for r in results:
        # Parse the content string into structured data
        content_lines = r.content.split('\n')
        metadata = {}
        current_field = None
        
        for line in content_lines:
            if line.startswith('Authors:'):
                metadata['authors'] = line.replace('Authors:', '').strip().split(', ')
            elif line.startswith('Published:'):
                metadata['published_date'] = line.replace('Published:', '').strip()
            elif line.startswith('Updated:'):
                metadata['updated_date'] = line.replace('Updated:', '').strip()
            elif line.startswith('Categories:'):
                metadata['categories'] = line.replace('Categories:', '').strip().split(', ')
            elif line.startswith('Abstract:'):
                current_field = 'abstract'
                metadata['abstract'] = ''
            elif line.startswith('PDF URL:'):
                metadata['pdf_url'] = line.replace('PDF URL:', '').strip()
            elif current_field == 'abstract' and line.strip():
                metadata['abstract'] = metadata.get('abstract', '') + line + '\n'
        
        formatted_results.append({
            "title": r.title,
            "url": r.url,
            **metadata
        })
    
    return formatted_results

def get_llm_tools():
    """Get all LLM-compatible tools."""
    return [
        search_foia,
        search_arxiv,
        get_latest_arxiv_papers
    ]
