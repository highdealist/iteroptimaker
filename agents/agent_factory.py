"""Factory class for creating agents with specific configurations."""

from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import importlib.util
import sys

from models.model_manager import ModelManager
from tools.tool_manager import ToolManager
from agent import BaseAgent

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_type: str
    system_prompt: str
    tools: List[str] = field(default_factory=list)
    generation_config: Dict[str, Any] = field(default_factory=dict)
    model_type: str = "default"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.agent_type:
            raise ValueError("agent_type cannot be empty")
        if not self.system_prompt:
            raise ValueError("system_prompt cannot be empty")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Create AgentConfig from dictionary with validation."""
        required_fields = {"agent_type", "system_prompt"}
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
            
        return cls(
            agent_type=data["agent_type"],
            system_prompt=data["system_prompt"],
            tools=data.get("tools", []),
            generation_config=data.get("generation_config", {}),
            model_type=data.get("model_type", "default")
        )


class AgentFactory:
    """Factory class for creating agents with specific configurations."""
    
    _agent_cache: Dict[str, Type[BaseAgent]] = {}

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the agent factory.
        
        Args:
            config_dir: Optional directory containing agent configurations.
                     If None, uses the 'config' directory in the same directory as this file.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, AgentConfig] = {}
        self._load_configs()

    def _validate_config(self, name: str, config: Dict[str, Any]) -> None:
        """
        Validate agent configuration.
        
        Args:
            name: Name of the agent
            config: Configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["agent_type", "system_prompt"]
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"Missing required fields in agent config '{name}': {missing_fields}")
        
        if not isinstance(config.get("tools", []), list):
            raise ValueError(f"'tools' must be a list in agent config '{name}'")
            
        if not isinstance(config.get("generation_config", {}), dict):
            raise ValueError(f"'generation_config' must be a dictionary in agent config '{name}'")

    def _load_configs(self) -> None:
        """Load agent configurations from the config directory."""
        try:
            config_file = self.config_dir / "agents.json"
            if not config_file.exists():
                logger.warning(f"Agent config file not found at {config_file}")
                return
                
            with open(config_file, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse agent configurations: {e}")
                    return
                    
                for name, config in data.items():
                    try:
                        self._validate_config(name, config)
                        self.configs[name] = AgentConfig.from_dict(config)
                    except (ValueError, KeyError) as e:
                        logger.error(f"Invalid configuration for agent '{name}': {e}")
                        
        except Exception as e:
            logger.error(f"Error loading agent configs: {e}")
            self.configs = {}

    @classmethod
    def _import_agent_class(cls, agent_type: str) -> Optional[Type[BaseAgent]]:
        """
        Import agent class using importlib for better error handling.
        
        Args:
            agent_type: Type of agent to import
            
        Returns:
            Agent class or None if import fails
        """
        if agent_type in cls._agent_cache:
            return cls._agent_cache[agent_type]
            
        try:
            module_name = f"{agent_type.lower()}_agent"
            module_path = Path(__file__).parent / f"{module_name}.py"
            
            if not module_path.exists():
                raise ImportError(f"Agent module not found: {module_path}")
                
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load module spec for {module_name}")
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            class_name = f"{agent_type.capitalize()}Agent"
            agent_class = getattr(module, class_name)
            
            # Verify the class is a subclass of BaseAgent
            if not issubclass(agent_class, BaseAgent):
                raise TypeError(f"Agent class {class_name} must inherit from BaseAgent")
                
            cls._agent_cache[agent_type] = agent_class
            return agent_class
            
        except Exception as e:
            logger.error(f"Failed to import agent class {agent_type}: {e}")
            return None

    @classmethod
    def create_agent(
        cls,
        agent_type: str,
        model_manager: ModelManager,
        tool_manager: ToolManager,
        config: AgentConfig
    ) -> Optional[BaseAgent]:
        """
        Creates an agent based on configuration.
        
        Args:
            agent_type: Type of agent to create
            model_manager: ModelManager instance
            tool_manager: ToolManager instance
            config: Agent configuration
            
        Returns:
            Agent instance or None if creation fails
        """
        try:
            agent_class = cls._import_agent_class(agent_type)
            if agent_class is None:
                return None

            # Create agent instance with validated configuration
            agent = agent_class(
                model_manager=model_manager,
                tool_manager=tool_manager,
                agent_type=config.agent_type,
                instruction=config.system_prompt,
                tools=config.tools,
                model_config=config.generation_config
            )
            return agent

        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            return None

    def get_agent_config(self, agent_type: str) -> Optional[AgentConfig]:
        """Get configuration for an agent type."""
        return self.configs.get(agent_type)

    def list_agent_types(self) -> List[str]:
        """Get list of available agent types."""
        return list(self.configs.keys())
