"""Main application entry point."""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
import json
from tkinter import messagebox
import os

from agents.agent_factory import AgentFactory, AgentConfig
from tools.tool_manager import ToolManager
from models.model_manager import ModelManager
from agents.agent import Agent
from tools.search_manager import SearchManager, initialize_search_manager
from gui.app import App
from config import (
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    LLM_PROVIDER,
    GEMINI_PRO_MODEL
)
from models import OpenAIProvider, GeminiProvider
import tools.read_document

logger = logging.getLogger(__name__)

class AIAssistantApp:
    """Main application class with improved dependency management."""
    
    def __init__(self):
        # Initialize core services
        self.search_manager = initialize_search_manager()
        self.tool_manager = self._initialize_tool_manager()
        self.model_manager = ModelManager(search_enabled=True, tool_manager=self.tool_manager)
        self.llm_provider = self._initialize_llm_provider()
        
        # Initialize agent factory
        self.agent_factory = AgentFactory()
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize GUI
        self.app = App(
            search_manager=self.search_manager,
            tool_manager=self.tool_manager,
            model_manager=self.model_manager,
            llm_provider=self.llm_provider,
            researcher_agent=self.agents.get('researcher'),
            writer_agent=self.agents.get('writer')
        )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper cleanup of resources."""
        try:
            self.model_manager.cleanup_all()
            if hasattr(self.tool_manager, 'cleanup'):
                self.tool_manager.cleanup()
            if hasattr(self.search_manager, 'cleanup'):
                self.search_manager.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _initialize_llm_provider(self):
        """Initialize LLM provider based on configuration."""
        try:
            if not LLM_PROVIDER or LLM_PROVIDER.strip() == "":  # Better empty check
                raise ValueError("LLM_PROVIDER not configured in environment")
                
            providers = {
                "openai": (OPENAI_API_KEY, OpenAIProvider),
                "gemini": (GEMINI_API_KEY, GeminiProvider)
            }
            
            if LLM_PROVIDER not in providers:
                raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
                
            api_key, provider_class = providers[LLM_PROVIDER]
            if not api_key:
                raise ValueError(f"{LLM_PROVIDER} API key not configured in environment")
                
            return provider_class()
        except ValueError as e:  # More specific error handling
            logger.error(f"Configuration error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing LLM provider: {str(e)}")
            raise
    
    def _initialize_tool_manager(self):
        """Initialize tool manager and register tools."""
        tool_manager = ToolManager()
        
        # Register core tools
        tool_manager.register_tool("read_document", tools.read_document.ReadDocumentTool())
        
        return tool_manager
    
    def _initialize_agents(self) -> Dict[str, Agent]:
        """Initialize all configured agents."""
        agents = {}
        
        # Load agent configurations from an absolute path
        config_path = Path(__file__).parent.absolute() / "config" / "agents.json"
        try:
            if not config_path.exists():
                raise FileNotFoundError(f"Agent configuration file not found at {config_path}")
                
            with open(config_path) as f:
                agent_configs = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load agent configurations: {e}")
            raise
        
        # Create agents from configurations
        for name, config in agent_configs.items():
            try:
                agent_config = AgentConfig(**config)
                agent = self.agent_factory.create_agent(
                    name=name,
                    config=agent_config,
                    model_manager=self.model_manager,
                    tool_manager=self.tool_manager
                )
                agents[name] = agent
            except Exception as e:
                logger.error(f"Failed to create agent '{name}': {e}")
        
        return agents
    
    def run(self):
        """Start the application with error handling."""
        try:
            self.app.run()
        except Exception as e:
            logger.error(f"Application error: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")
            raise


def main():
    """Application entry point with error handling."""
    try:
        app = AIAssistantApp()
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        messagebox.showerror("Fatal Error", f"A fatal error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
