"""
Model manager for creating and managing different language models.
"""
from typing import Dict, Any, Optional
from models.model import BaseModel
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages the creation and lifecycle of language models."""
    
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self._model_classes = {}
        self._model_settings = {}
        self._model_fallbacks = {}
        self._initialize_model_classes()
        
    def _initialize_model_classes(self):
        """Initialize model classes lazily to avoid circular imports."""
        from gemini import GeminiModel
        from openai import OpenAIModel
        from .model_config import MODEL_FALLBACKS, MODEL_SETTINGS
        
        self._model_classes = {
            "gemini": GeminiModel,
            "openai": OpenAIModel
        }
        self._model_settings = MODEL_SETTINGS
        self._model_fallbacks = MODEL_FALLBACKS
        
    def create_model(
        self,
        model_type: str,
        model_config: Optional[Dict[str, Any]] = None
    ) -> BaseModel:
        """
        Create a new model instance.
        
        Args:
            model_type: Type of model to create ("gemini" or "openai")
            model_config: Optional model configuration
            
        Returns:
            Created model instance
            
        Raises:
            ValueError: If model type is unknown or initialization fails
        """
        if model_type not in self._model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Clean up existing model if it exists
        if model_type in self.models:
            self.cleanup_model(model_type)
            
        config = model_config or self._model_settings.get(model_type, {})
        try:
            model_class = self._model_classes[model_type]
            model = model_class(config)
            self.models[model_type] = model
            return model
        except Exception as e:
            logger.error(f"Failed to initialize {model_type} model: {e}")
            raise ValueError(f"Model initialization failed: {e}")
            
    def cleanup_model(self, model_type: str) -> None:
        """Clean up resources for a specific model with proper state checking."""
        model = self.models.get(model_type)
        if model is None:
            return
            
        try:
            if hasattr(model, 'cleanup') and not getattr(model, '_cleaned_up', False):
                model.cleanup()
                setattr(model, '_cleaned_up', True)
        except Exception as e:
            logger.error(f"Error cleaning up {model_type} model: {e}")
        finally:
            self.models.pop(model_type, None)
                
    def cleanup_all(self) -> None:
        """Clean up all model resources."""
        for model_type in list(self.models.keys()):
            self.cleanup_model(model_type)
            
    def get_model(self, model_type: str) -> Optional[BaseModel]:
        """Get an existing model instance by type."""
        return self.models.get(model_type)
        
    def list_models(self) -> Dict[str, str]:
        """List all available model types."""
        return {name: model.model_type for name, model in self.models.items()}
        
    def get_fallback_model(self, model_type: str) -> Optional[BaseModel]:
        """Get a fallback model if the requested model is unavailable."""
        fallback_type = self._model_fallbacks.get(model_type)
        if fallback_type:
            return self.get_model(fallback_type)
        return None
