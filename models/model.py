"""
Base model interface and implementation.
"""
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all model implementations."""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on the prompt."""
        pass
        
    @abstractmethod
    def chat(self, messages: list, **kwargs) -> str:
        """Generate response in a chat context."""
        pass
        
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Get the type of this model."""
        pass
        
    def validate_config(self) -> bool:
        """Validate the model configuration."""
        required_fields = ["temperature", "max_tokens"]
        return all(field in self.model_config for field in required_fields)
        
    def get_config(self) -> Dict[str, Any]:
        """Get the current model configuration."""
        return self.model_config.copy()
        
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update the model configuration."""
        self.model_config.update(new_config)
