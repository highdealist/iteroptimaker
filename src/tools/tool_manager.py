"""
Tool manager for creating and managing different tools.
"""
from typing import Dict, Any, List, Optional, Type, Union
from .base_tool import BaseTool
import logging

logger = logging.getLogger(__name__)

class ToolManager:
    """Manages the creation and lifecycle of tools."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        self._initialize_tools()
        
    def _initialize_tools(self):
        """Initialize tool classes lazily to avoid circular imports."""
        try:
            # Import tools lazily
            from .web_search_tools.search_tools import SearchTool
            from .code_analysis_tool import CodeAnalysisTool
            from .read_document import ReadDocumentTool
            
            # Register core tools
            self._register_tool("search", SearchTool)
            self._register_tool("code_analysis", CodeAnalysisTool)
            self._register_tool("read_document", ReadDocumentTool)
            
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            
    def _register_tool(self, name: str, tool_class: Type[BaseTool]) -> None:
        """Register a tool class."""
        if not issubclass(tool_class, BaseTool):
            raise ValueError(f"Tool class {tool_class.__name__} must inherit from BaseTool")
        self._tool_classes[name] = tool_class
        
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool instance by name."""
        if name not in self.tools:
            tool_class = self._tool_classes.get(name)
            if tool_class is None:
                logger.warning(f"Tool '{name}' not found")
                return None
            try:
                self.tools[name] = tool_class()
            except Exception as e:
                logger.error(f"Failed to create tool '{name}': {e}")
                return None
        return self.tools[name]
        
    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self._tool_classes.keys())
        
    def cleanup(self):
        """Clean up all tool resources."""
        for tool in self.tools.values():
            if hasattr(tool, 'cleanup'):
                try:
                    tool.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up tool {tool.name}: {e}")
        self.tools.clear()
