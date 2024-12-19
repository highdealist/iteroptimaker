"""
Base agent implementation providing core functionality for all agent types.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import re
from ..models.model_manager import ModelManager
from ..tools.tool_manager import ToolManager
from ..tools.tool_executor import ToolExecutor
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
        self.tool_executor = ToolExecutor(tool_manager)
        self.agent_type = agent_type
        self.instruction = instruction
        self.tools = tools
        self.model_config = model_config
        self.name = name or agent_type.capitalize()
        self.chat_log = []
        
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
        response = model.generate_message(
            messages,
            context=context,
            **self.model_config
        )
        
        response_text = response['content']
        
        # Process any tool calls in the response
        response_text = self._process_tool_calls(response_text)
        
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
                    param_type = param_spec.get("type", "Any")
                    param_desc = param_spec.get("description", "")
                    param_required = param_spec.get("required", False)
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

    def _process_tool_calls(self, response_text: str) -> str:
        """Process any tool calls in the response text.
        
        Args:
            response_text: The text to process for tool calls
            
        Returns:
            Updated text with tool calls replaced by their results
        """
        # Find tool calls between <tool> tags
        tool_pattern = r"<tool>\s*([\w_]+)\s*\((.*?)\)\s*</tool>"
        matches = re.finditer(tool_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                tool_name = match.group(1).strip()
                args_str = match.group(2).strip()
                
                if tool_name not in self.tools:
                    continue
                    
                tool = self.tool_manager.get_tool(tool_name)
                if not tool:
                    continue
                
                # Parse arguments safely
                args_dict = self._parse_tool_args(args_str, tool.parameters)
                
                # Execute tool
                result = self.tool_executor.execute(tool_name, args_dict)
                
                if result.success:
                    response_text = response_text.replace(
                        match.group(0),
                        str(result.result)
                    )
                else:
                    response_text = response_text.replace(
                        match.group(0),
                        f"Error using {tool_name}: {result.error}"
                    )
            except Exception as e:
                response_text = response_text.replace(
                    match.group(0),
                    f"Error processing {tool_name}: {str(e)}"
                )
                    
        return response_text
        
    def _parse_tool_args(self, args_str: str, param_specs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Safely parse tool arguments from string.
        
        Args:
            args_str: String containing tool arguments
            param_specs: Parameter specifications from the tool
            
        Returns:
            Dictionary of parsed arguments
            
        Raises:
            ValueError: If argument parsing fails
        """
        args_dict = {}
        
        # Split into individual parameter assignments
        param_pattern = r'(\w+)\s*=\s*(.+?)(?=\s*(?:\w+\s*=|$))'
        param_matches = re.finditer(param_pattern, args_str)
        
        for param_match in param_matches:
            param_name = param_match.group(1).strip()
            param_value_str = param_match.group(2).strip()
            
            if param_name not in param_specs:
                continue
                
            # Get expected type
            param_type = param_specs[param_name].get("type", "Any")
            
            try:
                # Parse value based on type
                if param_type == "str":
                    # Remove quotes and unescape
                    if param_value_str.startswith('"') and param_value_str.endswith('"'):
                        param_value = param_value_str[1:-1].encode().decode('unicode_escape')
                    else:
                        raise ValueError(f"String parameter {param_name} must be quoted")
                elif param_type == "int":
                    param_value = int(param_value_str)
                elif param_type == "float":
                    param_value = float(param_value_str)
                elif param_type == "bool":
                    param_value = param_value_str.lower() == "true"
                elif param_type == "list":
                    param_value = self._parse_list_safely(param_value_str, param_name)
                elif param_type == "dict":
                    param_value = self._parse_dict_safely(param_value_str, param_name)
                else:
                    raise ValueError(f"Unsupported parameter type: {param_type}")
                    
                args_dict[param_name] = param_value
                
            except Exception as e:
                raise ValueError(f"Failed to parse parameter {param_name}: {str(e)}")
        
        return args_dict
        
    def _parse_list_safely(self, list_str: str, param_name: str) -> List[Any]:
        """Safely parse a list string without using eval.
        
        Args:
            list_str: String containing list (e.g., '["a", 1, true]')
            param_name: Parameter name for error messages
            
        Returns:
            Parsed list
            
        Raises:
            ValueError: If list format is invalid
        """
        if not (list_str.startswith("[") and list_str.endswith("]")):
            raise ValueError(f"List parameter {param_name} must be in square brackets")
            
        # Empty list
        if list_str == "[]":
            return []
            
        # Remove brackets and split by commas, handling nested structures
        content = list_str[1:-1].strip()
        if not content:
            return []
            
        items = []
        current_item = ""
        bracket_count = 0
        brace_count = 0
        in_string = False
        escape_next = False
        
        for char in content:
            if escape_next:
                current_item += char
                escape_next = False
                continue
                
            if char == "\\":
                current_item += char
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                current_item += char
            elif char == "[":
                bracket_count += 1
                current_item += char
            elif char == "]":
                bracket_count -= 1
                current_item += char
            elif char == "{":
                brace_count += 1
                current_item += char
            elif char == "}":
                brace_count -= 1
                current_item += char
            elif char == "," and not in_string and bracket_count == 0 and brace_count == 0:
                items.append(current_item.strip())
                current_item = ""
            else:
                current_item += char
                
        if current_item:
            items.append(current_item.strip())
            
        # Parse each item
        parsed_items = []
        for item in items:
            if item.startswith('"') and item.endswith('"'):
                # String
                parsed_items.append(item[1:-1].encode().decode('unicode_escape'))
            elif item.lower() == "true":
                # Boolean true
                parsed_items.append(True)
            elif item.lower() == "false":
                # Boolean false
                parsed_items.append(False)
            elif item.startswith("["):
                # Nested list
                parsed_items.append(self._parse_list_safely(item, f"{param_name} (nested)"))
            elif item.startswith("{"):
                # Nested dict
                parsed_items.append(self._parse_dict_safely(item, f"{param_name} (nested)"))
            else:
                # Try number
                try:
                    if "." in item:
                        parsed_items.append(float(item))
                    else:
                        parsed_items.append(int(item))
                except ValueError:
                    raise ValueError(f"Invalid list item in {param_name}: {item}")
                    
        return parsed_items
        
    def _parse_dict_safely(self, dict_str: str, param_name: str) -> Dict[str, Any]:
        """Safely parse a dictionary string without using eval.
        
        Args:
            dict_str: String containing dictionary (e.g., '{"key": "value"}')
            param_name: Parameter name for error messages
            
        Returns:
            Parsed dictionary
            
        Raises:
            ValueError: If dictionary format is invalid
        """
        if not (dict_str.startswith("{") and dict_str.endswith("}")):
            raise ValueError(f"Dictionary parameter {param_name} must be in curly braces")
            
        # Empty dict
        if dict_str == "{}":
            return {}
            
        # Remove braces
        content = dict_str[1:-1].strip()
        if not content:
            return {}
            
        # Split into key-value pairs
        pairs = []
        current_pair = ""
        bracket_count = 0
        brace_count = 0
        in_string = False
        escape_next = False
        
        for char in content:
            if escape_next:
                current_pair += char
                escape_next = False
                continue
                
            if char == "\\":
                current_pair += char
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                current_pair += char
            elif char == "[":
                bracket_count += 1
                current_pair += char
            elif char == "]":
                bracket_count -= 1
                current_pair += char
            elif char == "{":
                brace_count += 1
                current_pair += char
            elif char == "}":
                brace_count -= 1
                current_pair += char
            elif char == "," and not in_string and bracket_count == 0 and brace_count == 0:
                pairs.append(current_pair.strip())
                current_pair = ""
            else:
                current_pair += char
                
        if current_pair:
            pairs.append(current_pair.strip())
            
        # Parse each key-value pair
        parsed_dict = {}
        for pair in pairs:
            # Split key and value
            key_value = pair.split(":", 1)
            if len(key_value) != 2:
                raise ValueError(f"Invalid dictionary pair in {param_name}: {pair}")
                
            key_str = key_value[0].strip()
            value_str = key_value[1].strip()
            
            # Parse key (must be a string)
            if not (key_str.startswith('"') and key_str.endswith('"')):
                raise ValueError(f"Dictionary key must be a quoted string in {param_name}: {key_str}")
            key = key_str[1:-1].encode().decode('unicode_escape')
            
            # Parse value
            if value_str.startswith('"') and value_str.endswith('"'):
                # String
                value = value_str[1:-1].encode().decode('unicode_escape')
            elif value_str.lower() == "true":
                # Boolean true
                value = True
            elif value_str.lower() == "false":
                # Boolean false
                value = False
            elif value_str.startswith("["):
                # Nested list
                value = self._parse_list_safely(value_str, f"{param_name} (nested)")
            elif value_str.startswith("{"):
                # Nested dict
                value = self._parse_dict_safely(value_str, f"{param_name} (nested)")
            else:
                # Try number
                try:
                    if "." in value_str:
                        value = float(value_str)
                    else:
                        value = int(value_str)
                except ValueError:
                    raise ValueError(f"Invalid dictionary value in {param_name}: {value_str}")
                    
            parsed_dict[key] = value
            
        return parsed_dict


#Example of how to use this class
agent = BaseAgent(model_manager, tool_manager, "assistant", "Your instructions here", ["tool1", "tool2"], {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "max_output_tokens": 32000})
