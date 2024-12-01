"""Researcher agent implementation."""
import re
from typing import Dict, Any, List
from agent import BaseAgent

class ResearcherAgent(BaseAgent):
    """Agent specialized in research tasks using search and analysis tools."""
    
    def __init__(
        self,
        model_manager,
        tool_manager,
        instruction: str,
        model_config: Dict[str, Any],
        name: str = "researcher"
    ):
        super().__init__(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="researcher",
            instruction=instruction,
            tools=["web_search", "arxiv_search"],  # Default tools for researcher
            model_config=model_config,
            name=name
        )

    def generate_response(self, user_input: str, context: str = "") -> str:
        """Generate a researched response based on user input.
        
        Args:
            user_input: The user's research query
            context: Additional context for the research
            
        Returns:
            Researched response incorporating tool results
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
        
        # Check for tool usage in response
        for tool_name in self.tools:
            tool_pattern = rf"{tool_name}\((.*?)\)"
            matches = re.finditer(tool_pattern, response_text, re.DOTALL)
            
            for match in matches:
                try:
                    # Extract and evaluate tool arguments
                    args_str = match.group(1).strip()
                    tool = self.tool_manager.get_tool(tool_name)
                    
                    if tool:
                        # Parse arguments (simplified for example)
                        args_dict = eval(f"dict({args_str})")
                        result = tool.execute(**args_dict)
                        
                        if result.success:
                            # Replace tool call with result in response
                            tool_call = match.group(0)
                            response_text = response_text.replace(
                                tool_call,
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
        
        # Update chat log
        self.chat_log.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response_text}
        ])
        
        return response_text
