"""
OpenAI model implementation.
"""
from typing import Dict, Any, List
import openai
from ..base.model import BaseModel
from ...config import OPENAI_API_KEY

class OpenAIModel(BaseModel):
    """Implementation of the OpenAI model interface."""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self._setup_model()
        
    def _setup_model(self):
        """Initialize the OpenAI model."""
        openai.api_key = OPENAI_API_KEY
        self._model_name = self.model_config.get("model_name", "gpt-4")
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the OpenAI model."""
        response = openai.ChatCompletion.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.model_config.get("temperature", 0.7),
            max_tokens=self.model_config.get("max_tokens", 2048),
            **kwargs
        )
        return response.choices[0].message.content
        
    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Generate response in a chat context using OpenAI."""
        response = openai.ChatCompletion.create(
            model=self._model_name,
            messages=messages,
            temperature=self.model_config.get("temperature", 0.7),
            max_tokens=self.model_config.get("max_tokens", 2048),
            **kwargs
        )
        return response.choices[0].message.content
        
    @property
    def model_type(self) -> str:
        return "openai"
