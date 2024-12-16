"""Base workflow class defining the interface for all workflows."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from ..models.model_manager import ModelManager
from ..tools.tool_manager import ToolManager

@dataclass
class WorkflowState:
    """Base state class for all workflows."""
    input: str = ""
    output: str = ""
    context: str = ""
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseWorkflow(ABC):
    """Abstract base class for all workflows."""
    
    def __init__(
        self,
        model_manager: ModelManager,
        tool_manager: ToolManager,
        max_iterations: int = 3
    ):
        self.model_manager = model_manager
        self.tool_manager = tool_manager
        self.max_iterations = max_iterations
        self.execution_history = []
        
    @abstractmethod
    def run(self, task: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute the workflow.
        
        Args:
            task: The task to execute
            **kwargs: Additional workflow-specific parameters
            
        Returns:
            Dictionary containing workflow results
        """
        pass
        
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate that a task contains all required information.
        
        Args:
            task: Task dictionary to validate
            
        Returns:
            True if task is valid, False otherwise
        """
        return True  # Base implementation accepts all tasks
        
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the workflow's execution history.
        
        Returns:
            List of execution records
        """
        return self.execution_history
        
    def _record_execution(
        self,
        task: Dict[str, Any],
        result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an execution in the history.
        
        Args:
            task: The executed task
            result: The execution result
            metadata: Optional execution metadata
        """
        self.execution_history.append({
            "task": task,
            "result": result,
            "metadata": metadata or {}
        })
        
    def _get_tool(self, tool_name: str):
        """Get a tool by name with error handling.
        
        Args:
            tool_name: Name of the tool to get
            
        Returns:
            The tool instance or None if not found
        """
        try:
            return self.tool_manager.get_tool(tool_name)
        except Exception as e:
            self._record_execution(
                {"tool_request": tool_name},
                {"error": str(e)},
                {"type": "tool_error"}
            )
            return None
