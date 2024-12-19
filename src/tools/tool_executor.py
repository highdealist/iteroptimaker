"""
Tool executor for handling tool calls with validation and error handling.
"""
from typing import Dict, Any
from .base_tool import BaseTool, ToolResult
from .tool_manager import ToolManager

class ToolExecutor:
    """Executes tools with validation and error handling."""
    
    def __init__(self, tool_manager: ToolManager):
        """
        Initialize the ToolExecutor.
        
        Args:
            tool_manager: ToolManager instance
        """
        self.tool_manager = tool_manager
        
    def execute(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Dictionary of parameters for the tool
            
        Returns:
            ToolResult object containing the result or error
        """
        tool = self.tool_manager.get_tool(tool_name)
        if not tool:
            return ToolResult(success=False, error=f"Tool '{tool_name}' not found")
            
        # Validate parameters
        validation_result = self._validate_params(tool, params)
        if not validation_result.success:
            return validation_result
            
        try:
            result = tool.execute(**params)
            return ToolResult(success=True, result=result)
        except Exception as e:
            return ToolResult(success=False, error=f"Error executing tool '{tool_name}': {e}")
            
    def _validate_params(self, tool: BaseTool, params: Dict[str, Any]) -> ToolResult:
        """
        Validate the parameters against the tool's parameter specifications.
        
        Args:
            tool: The tool to validate parameters for
            params: Dictionary of parameters to validate
            
        Returns:
            ToolResult object indicating success or failure
        """
        for param_name, param_spec in tool.parameters.items():
            if param_spec.get("required", False) and param_name not in params:
                return ToolResult(
                    success=False,
                    error=f"Missing required parameter '{param_name}' for tool '{tool.name}'"
                )
            if param_name in params:
                param_value = params[param_name]
                param_type = param_spec.get("type", "Any")
                
                if param_type == "str" and not isinstance(param_value, str):
                    return ToolResult(
                        success=False,
                        error=f"Parameter '{param_name}' for tool '{tool.name}' must be a string"
                    )
                if param_type == "int" and not isinstance(param_value, int):
                    try:
                        int(param_value)
                    except ValueError:
                        return ToolResult(
                            success=False,
                            error=f"Parameter '{param_name}' for tool '{tool.name}' must be an integer"
                        )
                if param_type == "float" and not isinstance(param_value, (int, float)):
                    try:
                        float(param_value)
                    except ValueError:
                        return ToolResult(
                            success=False,
                            error=f"Parameter '{param_name}' for tool '{tool.name}' must be a float"
                        )
                if param_type == "bool" and not isinstance(param_value, bool):
                    if not isinstance(param_value, str) or param_value.lower() not in ["true", "false"]:
                        return ToolResult(
                            success=False,
                            error=f"Parameter '{param_name}' for tool '{tool.name}' must be a boolean"
                        )
                if param_type == "list" and not isinstance(param_value, list):
                    return ToolResult(
                        success=False,
                        error=f"Parameter '{param_name}' for tool '{tool.name}' must be a list"
                    )
                if param_type == "dict" and not isinstance(param_value, dict):
                    return ToolResult(
                        success=False,
                        error=f"Parameter '{param_name}' for tool '{tool.name}' must be a dictionary"
                    )
        return ToolResult(success=True)
