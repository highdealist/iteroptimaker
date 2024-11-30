"""
Search tool implementation.
"""
from typing import Dict, Any, List
import requests
from ..base.tool import BaseTool
from ...config import TAVILY_API_KEY

class SearchTool(BaseTool):
    """Tool for performing web searches."""
    
    def __init__(self, tool_config: Dict[str, Any] = None):
        super().__init__(tool_config)
        self._api_key = TAVILY_API_KEY
        
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a web search.
        
        Args:
            params: Must contain 'query' key with search query
            
        Returns:
            Dictionary containing search results
        """
        if not self.validate_params(params):
            return {"error": "Missing required parameters"}
            
        query = params["query"]
        max_results = params.get("max_results", 5)
        
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": max_results
                }
            )
            response.raise_for_status()
            return {"results": response.json()["results"]}
            
        except requests.exceptions.RequestException as e:
            return {"error": f"Search failed: {str(e)}"}
            
    @property
    def tool_name(self) -> str:
        return "web_search"
        
    @property
    def description(self) -> str:
        return "Perform web searches to find relevant information"
        
    @property
    def required_params(self) -> List[str]:
        return ["query"]
