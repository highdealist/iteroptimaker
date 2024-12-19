"""
Gemini model implementation with function calling, env var initialization,
improved error handling, logging, and model selection.
"""
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from google import genai
from google.generativeai.types import GenerateContentResponse
import logging as logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import os
from model import BaseModel
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from google.api_core import exceptions

#Get system_instructions from the corresponding agent type
def get_system_instructions(Agent()) -> str:
    """Get system instructions based on the agent type."""
    if agent_type == AgentType.GEMINI:
        return "You are a helpful AI assistant."




system_instructions = "You are a helpful AI assistant."

MODEL_FALLBACKS = {
    "gemini": ["gemini-pro-1.5-latest", "gemini-flash-1.5-latest", "claude-3-opus"],
    "gpt-4": ["gemini-pro", "gpt-3.5-turbo", "claude-3-opus"],
    "claude-3-opus": ["gpt-4", "gemini-pro", "gpt-3.5-turbo"]
}

MODEL_SETTINGS = {
    "gemini": {
        "system_instructions": system_instructions,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 16000,
        "safety_settings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    },
}

# Retry settings
RETRY_SETTINGS = {
    "max_retries": 3,
    "initial_delay": 1,
    "backoff_factor": 2,
    "max_delay": 10
}

@retry(
    stop=stop_after_attempt(RETRY_SETTINGS["max_retries"]),
    wait=wait_exponential(
        multiplier=RETRY_SETTINGS["initial_delay"],
        max=RETRY_SETTINGS["max_delay"]
    ),
    retry=retry_if_exception_type((
        ConnectionError,
        TimeoutError,
        exceptions.GoogleAPIError,
        Exception
    ))
)
def retry_generate(model, prompt: str, tools: Optional[List[Dict]] = None, **kwargs) -> GenerateContentResponse:
    """Retry wrapper for generate_content"""
    try:
        response = model.generate_content(prompt, tools=tools, **kwargs)
        if not response.text:
            raise ValueError("Empty response received from Gemini API")
        return response
    except exceptions.GoogleAPIError as e:
        logger.error(f"Google API Error during generation: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during generation: {e}")
        raise

class GeminiModel(BaseModel):
    """Implementation of the Gemini model interface."""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self._setup_model()
        self._history = []
        
    def _setup_model(self):
        """Initialize the Gemini model."""
        if not GEMINI_API_KEY and not (GEMINI_PROJECT_ID and GEMINI_LOCATION):
            raise ValueError("Either GEMINI_API_KEY or both GEMINI_PROJECT_ID and GEMINI_LOCATION environment variables must be set")
        
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
        else:
            genai.configure(project=GEMINI_PROJECT_ID, location=GEMINI_LOCATION)
            
        model_name = self.model_config.get("model_name")
        if not model_name:
            model_name = GEMINI_FLASH_MODEL if self.model_config.get("use_flash", False) else GEMINI_PRO_MODEL
        system_instruction = self.model_config.get("system_instruction")
        if system_instruction:
            genai.configure(system_instruction=system_instruction)
        
        generation_config = {
            "temperature": self.model_config.get("temperature", 0.7),
            "top_p": self.model_config.get("top_p", 0.95),
            "top_k": self.model_config.get("top_k", 40),
            "max_output_tokens": self.model_config.get("max_output_tokens", 8096),
        }
        
        try:
            self._model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                safety_settings=SAFETY_SETTINGS
            )
            logger.info(f"Initialized Gemini model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}")
            raise
        
    def generate(self, prompt: str, tools: Optional[List[Dict]] = None, **kwargs) -> str:
        """Generate text using the Gemini model with retry logic and function calling."""
        try:
            response = retry_generate(self._model, prompt, tools=tools, **kwargs)
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError("No text content in the response")
        except Exception as e:
            logger.error(f"Error generating content with Gemini: {str(e)}")
            raise
        
    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None, **kwargs) -> str:
        """Generate response in a chat context using Gemini with function calling."""
        try:
            chat = self._model.start_chat(history=self._history)
            
            for message in messages:
                if message["role"] == "user":
                    response = chat.send_message(message["content"], tools=tools)
                    self._history.append({"role": "user", "content": message["content"]})
                    if response.text:
                        self._history.append({"role": "assistant", "content": response.text})
                    elif response.candidates and response.candidates[0].content.parts:
                        self._history.append({"role": "assistant", "content": response.candidates[0].content.parts[0].text})
                    else:
                        raise ValueError("Empty response received from Gemini chat API")
                elif message["role"] == "system":
                    # Handle system messages by prepending to user messages
                    next_user_msg = next(
                        (m for m in messages[messages.index(message):] if m["role"] == "user"),
                        None
                    )
                    if next_user_msg:
                        modified_content = f"{message['content']}\n\nUser: {next_user_msg['content']}"
                        response = chat.send_message(modified_content, tools=tools)
                        self._history.append({"role": "user", "content": modified_content})
                        if response.text:
                            self._history.append({"role": "assistant", "content": response.text})
                        elif response.candidates and response.candidates[0].content.parts:
                            self._history.append({"role": "assistant", "content": response.candidates[0].content.parts[0].text})
                        else:
                            raise ValueError("Empty response received from Gemini chat API")
            
            if response.text:
                return response.text
            elif response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError("Empty response received from Gemini chat API")
            
        except Exception as e:
            logger.error(f"Error in Gemini chat: {str(e)}")
            raise
        
    def reset_chat(self):
        """Reset the chat history."""
        self._history = []
        
    @property
    def model_type(self) -> str:
        return "gemini"