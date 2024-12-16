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
        return super().generate_response(user_input, context)
