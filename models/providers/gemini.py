"""
Gemini model implementation.
"""
from typing import Dict, Any, List
import google.generativeai as genai
import logging
from .base.model import BaseModel
from config.model_config import (
    GEMINI_API_KEY,
    SAFETY_SETTINGS,
    GEMINI_PRO_MODEL,
    GEMINI_FLASH_MODEL
)

logger = logging.getLogger(__name__)

class GeminiModel(BaseModel):
    """Implementation of the Gemini model interface."""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self._setup_model()
        
    def _setup_model(self):
        """Initialize the Gemini model."""
        genai.configure(api_key=GEMINI_API_KEY)
        model_name = self.model_config.get("model_name", GEMINI_PRO_MODEL)
        generation_config = {
            "temperature": self.model_config.get("temperature", 0.7),
            "top_p": self.model_config.get("top_p", 0.95),
            "top_k": self.model_config.get("top_k", 40),
            "max_output_tokens": self.model_config.get("max_tokens", 8096),
        }
        
        self._model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=SAFETY_SETTINGS
        )
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the Gemini model."""
        try:
            response = self._model.generate_content(prompt, **kwargs)
            if not response.text:
                raise ValueError("Empty response received from Gemini API")
            return response.text
        except Exception as e:
            logger.error(f"Error generating content with Gemini: {str(e)}")
            raise
        
    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Generate response in a chat context using Gemini."""
        try:
            chat = self._model.start_chat()
            for message in messages:
                if message["role"] == "user":
                    chat.send_message(message["content"])
                elif message["role"] == "system":
                    # Convert system messages to user messages with a special prefix
                    chat.send_message(f"[System Instruction]: {message['content']}")
            
            response = chat.send_message(messages[-1]["content"])
            if not response.text:
                raise ValueError("Empty response received from Gemini chat API")
            return response.text
        except Exception as e:
            logger.error(f"Error in Gemini chat: {str(e)}")
            raise
        
    @property
    def model_type(self) -> str:
        return "gemini"
