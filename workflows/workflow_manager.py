"""Workflow manager for registering and executing workflows."""
from typing import Dict, Any, Type, Optional
from .base_workflow import BaseWorkflow
from ..models.model_manager import ModelManager
from ..tools.tool_manager import ToolManager

class WorkflowManager:
    """Manager class for workflow registration and execution."""
    
    def __init__(
        self,
        model_manager: ModelManager,
        tool_manager: ToolManager
    ):
        self.model_manager = model_manager
        self.tool_manager = tool_manager
        self._workflows: Dict[str, Type[BaseWorkflow]] = {}
        self._instances: Dict[str, BaseWorkflow] = {}
        
    def register_workflow(
        self,
        name: str,
        workflow_class: Type[BaseWorkflow],
        **kwargs
    ) -> None:
        """Register a workflow class.
        
        Args:
            name: Name to register the workflow under
            workflow_class: The workflow class to register
            **kwargs: Additional arguments for workflow instantiation
        """
        if not issubclass(workflow_class, BaseWorkflow):
            raise ValueError(
                f"Workflow class {workflow_class.__name__} must inherit from BaseWorkflow"
            )
            
        self._workflows[name] = workflow_class
        # Pre-instantiate the workflow if kwargs are provided
        if kwargs:
            self.get_workflow(name, **kwargs)
            
    def get_workflow(
        self,
        name: str,
        **kwargs
    ) -> BaseWorkflow:
        """Get or create a workflow instance.
        
        Args:
            name: Name of the workflow to get
            **kwargs: Additional arguments for workflow instantiation
            
        Returns:
            Workflow instance
            
        Raises:
            KeyError: If workflow is not registered
        """
        if name not in self._workflows:
            raise KeyError(f"Workflow '{name}' not registered")
            
        # Create new instance if it doesn't exist or kwargs are provided
        if name not in self._instances or kwargs:
            workflow_class = self._workflows[name]
            self._instances[name] = workflow_class(
                model_manager=self.model_manager,
                tool_manager=self.tool_manager,
                **kwargs
            )
            
        return self._instances[name]
        
    def execute_workflow(
        self,
        name: str,
        task: Dict[str, Any],
        workflow_kwargs: Optional[Dict[str, Any]] = None,
        **execution_kwargs
    ) -> Dict[str, Any]:
        """Execute a workflow.
        
        Args:
            name: Name of the workflow to execute
            task: Task to execute
            workflow_kwargs: Optional kwargs for workflow instantiation
            **execution_kwargs: Additional kwargs for workflow execution
            
        Returns:
            Workflow execution results
            
        Raises:
            KeyError: If workflow is not registered
        """
        workflow = self.get_workflow(name, **(workflow_kwargs or {}))
        
        if not workflow.validate_task(task):
            raise ValueError(f"Invalid task for workflow '{name}'")
            
        return workflow.run(task, **execution_kwargs)
        
    def list_workflows(self) -> Dict[str, str]:
        """Get a list of registered workflows.
        
        Returns:
            Dictionary mapping workflow names to their descriptions
        """
        return {
            name: workflow_class.__doc__ or "No description available"
            for name, workflow_class in self._workflows.items()
        }
        
    def get_workflow_history(
        self,
        name: str
    ) -> Dict[str, Any]:
        """Get execution history for a workflow.
        
        Args:
            name: Name of the workflow
            
        Returns:
            Dictionary containing workflow execution history
            
        Raises:
            KeyError: If workflow is not registered or instantiated
        """
        if name not in self._instances:
            raise KeyError(
                f"No execution history available for workflow '{name}'"
            )
            
        workflow = self._instances[name]
        return {
            "workflow": name,
            "history": workflow.get_execution_history()
        }
