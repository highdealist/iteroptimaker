"""Tools for specialized search functionality."""
from typing import Dict, Any, List, Optional
from ..base_tool import BaseTool, ToolResult
from .specialized_search import FOIASearchProvider, ArXivSearchProvider
import asyncio

class FOIASearchTool(BaseTool):
    """Tool for searching FOIA.gov records."""
    
    def __init__(self):
        super().__init__(
            name="foia_search",
            description="Search FOIA.gov for government records",
            use_case="Use this tool to search for Freedom of Information Act (FOIA) records and documents.",
            operation="Searches FOIA.gov's database and returns relevant documents with their content."
        )
        self.provider = FOIASearchProvider()
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "query": {
                "type": str,
                "description": "The search query",
                "required": True
            },
            "max_results": {
                "type": int,
                "description": "Maximum number of results to return",
                "required": False,
                "default": 10
            }
        }
    
    def execute(self, **kwargs) -> ToolResult:
        try:
            query = kwargs["query"]
            max_results = kwargs.get("max_results", 10)
            
            # Run the async search
            results = asyncio.run(self.provider.search(query, max_results))
            
            # Format results for output
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "content": result.content
                })
            
            return ToolResult(
                success=True,
                result=formatted_results
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"FOIA search failed: {str(e)}"
            )

class ArXivSearchTool(BaseTool):
    """Tool for searching arXiv papers."""
    
    def __init__(self):
        super().__init__(
            name="arxiv_search",
            description="Search arXiv for scientific papers",
            use_case="Use this tool to search for scientific papers on arXiv.",
            operation="Searches arXiv's database and returns papers with their abstracts and metadata."
        )
        self.provider = ArXivSearchProvider()
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "query": {
                "type": str,
                "description": "The search query",
                "required": True
            },
            "max_results": {
                "type": int,
                "description": "Maximum number of results to return",
                "required": False,
                "default": 10
            }
        }
    
    def execute(self, **kwargs) -> ToolResult:
        try:
            query = kwargs["query"]
            max_results = kwargs.get("max_results", 10)
            
            # Run the async search
            results = asyncio.run(self.provider.search(query, max_results))
            
            # Format results for output
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "content": result.content
                })
            
            return ToolResult(
                success=True,
                result=formatted_results
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"arXiv search failed: {str(e)}"
            )

class ArXivLatestTool(BaseTool):
    """Tool for fetching latest arXiv papers."""
    
    def __init__(self):
        super().__init__(
            name="arxiv_latest",
            description="Get latest papers from arXiv",
            use_case="Use this tool to get the most recent papers from arXiv, optionally filtered by category.",
            operation="Fetches the latest papers from arXiv within a specified timeframe and category."
        )
        self.provider = ArXivSearchProvider()
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "category": {
                "type": str,
                "description": "arXiv category (e.g., 'cs.AI', 'physics')",
                "required": False,
                "default": None
            },
            "max_results": {
                "type": int,
                "description": "Maximum number of results to return",
                "required": False,
                "default": 10
            },
            "days": {
                "type": int,
                "description": "Number of past days to search",
                "required": False,
                "default": 7
            }
        }
    
    def execute(self, **kwargs) -> ToolResult:
        try:
            category = kwargs.get("category")
            max_results = kwargs.get("max_results", 10)
            days = kwargs.get("days", 7)
            
            # Run the async search
            results = asyncio.run(self.provider.get_latest_papers(
                category=category,
                max_results=max_results,
                days=days
            ))
            
            # Format results for output
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "content": result.content
                })
            
            return ToolResult(
                success=True,
                result=formatted_results
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"arXiv latest papers fetch failed: {str(e)}"
            )
