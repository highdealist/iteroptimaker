"""Base tool class defining the interface for all tools."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class ToolResult:
    """Container for tool execution results."""
    success: bool
    result: Any
    error: Optional[str] = None

class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, name: str, description: str, use_case: str = None, operation: str = None):
        """Initialize a tool.
        
        Args:
            name: Tool name
            description: Brief description of the tool's purpose
            use_case: Detailed description of when to use this tool
            operation: Technical description of how the tool works
        """
        self.name = name
        self.description = description
        self.use_case = use_case
        self.operation = operation
        
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult containing execution status and result
        """
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the tool's parameter specifications.
        
        Returns:
            Dictionary mapping parameter names to their specifications:
            {
                "param_name": {
                    "type": str,  # Parameter type
                    "description": str,  # Parameter description
                    "required": bool,  # Whether parameter is required
                    "default": Any  # Default value if any
                }
            }
        """
        pass
