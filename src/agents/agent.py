"""
Base agent implementation providing core functionality for all agent types.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import re
from ..models.model_manager import ModelManager
from ..tools.tool_manager import ToolManager
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.prebuilt import ToolNode
model_manager = ModelManager()
tool_manager = ToolManager()


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(
        self,
        model_manager: ModelManager,
        tool_manager: ToolManager,
        agent_type: str,
        instruction: str,
        tools: List[str],
        model_config: Dict[str, Any],
        name: Optional[str] = None
    ):
        self.model_manager = model_manager
        self.tool_manager = tool_manager
        self.agent_type = agent_type
        self.instruction = instruction
        self.tools = tools
        self.model_config = model_config
        self.name = name or agent_type.capitalize()
        self.chat_log = []
        self.tool_node = ToolNode(self._get_langchain_tools())
        
    @abstractmethod
    def analyze(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task and return results.
        
        Args:
            task: Dictionary containing task details and context
            
        Returns:
            Dictionary containing analysis results
        """
        pass
        
    def get_available_tools(self) -> List[str]:
        """Get list of tools available to this agent."""
        return self.tool_manager.get_tools_for_agent(self.agent_type)
        
    @abstractmethod
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """
        Validate that a task contains all required information.
        
        Args:
            task: Task dictionary to validate
            
        Returns:
            True if task is valid, False otherwise
        """
        pass

    def generate_response(self, user_input: str, context: str = "") -> str:
        """Generate a response based on user input and context.
        
        Args:
            user_input: The user's input message
            context: Additional context for the response
            
        Returns:
            The agent's response
        """
        model = self.model_manager.get_model(self.agent_type)
        full_instruction = self._construct_instructions(self.instruction)
        
        # Add the input to chat log
        messages = self.chat_log + [{"role": "user", "content": user_input}]
        
        # Generate initial response
        response = model.chat(
            messages,
            context=context,
            tools=self._get_langchain_tools(),
            **self.model_config
        )
        
        # Process any tool calls in the response
        response_text = self._process_tool_calls(response)
        
        # Update chat log
        self.chat_log.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response_text}
        ])
        
        return response_text

    def _construct_instructions(self, base_instruction: str) -> str:
        """Construct full instructions including available tools and usage format.
        
        Args:
            base_instruction: The base instruction for the agent
            
        Returns:
            Complete instruction string including tool information and usage format
        """
        # Tool documentation
        tool_descriptions = []
        for tool_name in self.tools:
            tool = self.tool_manager.get_tool(tool_name)
            if tool:
                # Get complete tool documentation
                params = tool.parameters
                param_docs = []
                for param_name, param_spec in params.items():
                    param_type = param_spec.type
                    param_desc = param_spec.description
                    param_required = param_spec.required
                    param_docs.append(f"    - {param_name} ({param_type}{'*' if param_required else ''}): {param_desc}")
                
                # Format tool documentation
                tool_doc = f"""- {tool_name}:
    Purpose: {tool.description}
    Use Case: {tool.use_case if hasattr(tool, 'use_case') else 'Not specified'}
    Operation: {tool.operation if hasattr(tool, 'operation') else 'Not specified'}
    Arguments:
{chr(10).join(param_docs)}"""
                tool_descriptions.append(tool_doc)
        
        # Tool usage format
        usage_format = """
Tool Usage Format:
To call a tool, use the following format:
<tool>
{tool_name}(
    param1="value1",
    param2=123,
    param3=true
)
</tool>

Important:
1. Always wrap the tool call in <tool> tags
2. Put each parameter on a new line with proper indentation
3. Use proper Python literal syntax for values:
   - Strings in double quotes: "text"
   - Numbers without quotes: 123, 3.14
   - Booleans: true, false
   - Lists in square brackets: ["item1", "item2"]
   - Dictionaries in curly braces: {"key": "value"}
4. Always provide required parameters marked with *
5. Optional parameters can be omitted"""

        tools_text = "\n\nAvailable Tools:\n" + "\n\n".join(tool_descriptions) if tool_descriptions else "No tools available."
        return f"{base_instruction}\n{tools_text}\n{usage_format}"

    def _process_tool_calls(self, response: BaseMessage) -> str:
        """Process any tool calls in the response text.
        
        Args:
            response_text: The text to process for tool calls
            
        Returns:
            Updated text with tool calls replaced by their results
        """
        if isinstance(response, AIMessage) and response.tool_calls:
            tool_result = self.tool_node.invoke({"messages": [response]})
            return tool_result["messages"][-1].content
        else:
            return response.content
    
    def _get_langchain_tools(self) -> List[Any]:
        """Get a list of langchain tools from the tool manager."""
        langchain_tools = []
        for tool_name in self.tools:
            tool = self.tool_manager.get_tool(tool_name)
            if tool:
                langchain_tools.append(self._convert_to_langchain_tool(tool))
        return langchain_tools
    
    def _convert_to_langchain_tool(self, tool: Any) -> Any:
        """Convert a custom tool to a langchain tool."""
        @tool(tool.name, args_schema=self._create_pydantic_model(tool.parameters))
        def _tool(**kwargs):
            result = tool.execute(**kwargs)
            if result.success:
                return result.result
            else:
                return result.error
        return _tool
    
    def _create_pydantic_model(self, parameters: List[Any]) -> Any:
        """Create a pydantic model from a list of parameters."""
        from pydantic import BaseModel, create_model
        fields = {}
        for param in parameters:
            param_name = param.name
            param_type = param.type
            if param_type == "str":
                annotation = str
            elif param_type == "int":
                annotation = int
            elif param_type == "float":
                annotation = float
            elif param_type == "bool":
                annotation = bool
            elif param_type == "list":
                annotation = List
            elif param_type == "dict":
                annotation = Dict
            else:
                annotation = Any
            
            if param.required:
                fields[param_name] = (annotation, ...)
            else:
                fields[param_name] = (Optional[annotation], param.default)
        
        return create_model("ToolArgs", **fields)
