"""
Tool manager for creating and managing different tools.
"""
from typing import Dict, Any, List, Optional, Type, Union
from langchain_core.tools import BaseTool
import logging
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)

class ToolManager:
    """Manages the creation and lifecycle of tools."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        self._initialize_tools()
        self.tool_node = None
        
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
            
            # Initialize ToolNode
            self.tool_node = ToolNode(list(self.tools.values()))
            
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            
    def _register_tool(self, name: str, tool_class: Type[BaseTool]) -> None:
        """Register a tool class."""
        if not issubclass(tool_class, BaseTool):
            raise ValueError(f"Tool class {tool_class.__name__} must inherit from BaseTool")
        
        try:
            tool_instance = tool_class()
            self.tools[name] = tool_instance
            self._tool_classes[name] = tool_class
        except Exception as e:
            logger.error(f"Failed to create tool instance for '{name}': {e}")
            
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool instance by name."""
        return self.tools.get(name)
        
    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self._tool_classes.keys())
    
    def get_tool_node(self) -> Optional[ToolNode]:
        """Get the ToolNode instance."""
        return self.tool_node
        
    def cleanup(self):
        """Clean up all tool resources."""
        for tool in self.tools.values():
            if hasattr(tool, 'cleanup'):
                try:
                    tool.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up tool {tool.name}: {e}")
        self.tools.clear()
