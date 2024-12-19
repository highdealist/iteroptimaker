"""
Agent manager for creating and managing different types of agents.
"""
from typing import Dict, Any, Optional
import json
import os
from agent import BaseAgent
from coding_crew_agents import (
    DeveloperAgent,
    CodeReviewerAgent,
    SecurityReviewerAgent,
    TechLeadAgent
)
from ..models.model_manager import ModelManager
from ..tools.tool_manager import ToolManager

class AgentManager:
    """Manages the creation and lifecycle of agents."""
    
    def __init__(self, config_path: str, model_manager: ModelManager, tool_manager: ToolManager):
        self.config_path = config_path
        self.model_manager = model_manager
        self.tool_manager = tool_manager
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_configs = self._load_agent_configs()
        
    def _load_agent_configs(self) -> Dict[str, Any]:
        """Load agent configurations from JSON files."""
        configs = {}
        config_dir = os.path.join(os.path.dirname(__file__), "../configs")
        
        for filename in os.listdir(config_dir):
            if filename.endswith(".json"):
                with open(os.path.join(config_dir, filename), "r") as f:
                    agent_type = filename.replace(".json", "")
                    configs[agent_type] = json.load(f)
                    
        return configs
        
    def create_agent(
        self,
        agent_type: str,
        name: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> BaseAgent:
        """
        Create a new agent of the specified type.
        
        Args:
            agent_type: Type of agent to create
            name: Optional name for the agent
            custom_config: Optional custom configuration
            
        Returns:
            Created agent instance
        """
        config = custom_config or self.agent_configs.get(agent_type)
        if not config:
            raise ValueError(f"No configuration found for agent type: {agent_type}")
            
        agent_classes = {
            "developer": DeveloperAgent,
            "code_reviewer": CodeReviewerAgent,
            "security_reviewer": SecurityReviewerAgent,
            "tech_lead": TechLeadAgent
        }
        
        agent_class = agent_classes.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        agent = agent_class(
            model_manager=self.model_manager,
            tool_manager=self.tool_manager,
            agent_type=agent_type,
            instruction=config["instruction"],
            tools=config["tools"],
            model_config=config.get("model_config", {}),
            name=name
        )
        
        self.agents[agent.name] = agent
        return agent
        
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self.agents.get(name)
        
    def list_agents(self) -> Dict[str, str]:
        """List all active agents and their types."""
        return {name: agent.agent_type for name, agent in self.agents.items()}
