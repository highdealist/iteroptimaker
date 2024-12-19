"""Base tool class defining the interface for all tools."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field

@dataclass
class ToolParameter:
    """Defines a parameter for a tool."""
    name: str
    type: str
    description: str
    required: bool = False
    default: Optional[Any] = None

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
    def parameters(self) -> List[ToolParameter]:
        """Get the tool's parameter specifications.
        
        Returns:
            List of ToolParameter objects
        """
        pass
