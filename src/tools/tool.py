"""
Base tool interface and implementation.
"""
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, tool_config: Optional[Dict[str, Any]] = None):
        self.tool_config = tool_config or {}
        
    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with the given parameters.  
        
        Args:
            params: Tool parameters
            
        Returns:
            Dictionary containing tool execution results
        """
        pass
        
    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Get the name of this tool."""
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        """Get the description of this tool."""
        pass
        
    @property
    def required_params(self) -> list:
        """Get list of required parameters for this tool."""
        return []
        
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate that all required parameters are present.
        
        Args:
            params: Parameters to validate
            
        Returns:
            True if all required parameters are present, False otherwise
        """
        return all(param in params for param in self.required_params)
