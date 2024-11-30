"""Web search tool implementation."""
from typing import Dict, Any
from .base_tool import BaseTool, ToolResult
from ..config import TIMEOUT

class SearchTool(BaseTool):
    """Tool for performing web searches."""
    
    def __init__(self, search_manager):
        super().__init__(
            name="web_search",
            description="Performs web searches and returns relevant results"
        )
        self.search_manager = search_manager

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "query": {
                "type": "string",
                "description": "The search query",
                "required": True
            },
            "num_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 10,
                "minimum": 1
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Maximum time to wait for results",
                "default": TIMEOUT
            }
        }

    def execute(self, **kwargs) -> ToolResult:
        try:
            query = kwargs.get("query")
            if not query:
                return ToolResult(
                    success=False,
                    result=None,
                    error="Query parameter is required"
                )

            num_results = kwargs.get("num_results", 10)
            timeout_seconds = kwargs.get("timeout_seconds", TIMEOUT)

            results = self.search_manager.search(
                query,
                num_results=num_results,
                timeout=timeout_seconds
            )

            return ToolResult(
                success=True,
                result=results
            )

        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )
