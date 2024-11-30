"""Test configuration and fixtures."""

import sys
from unittest.mock import MagicMock

# Mock configuration module
mock_config = MagicMock()
mock_config.GEMINI_API_KEY = "test_api_key"
mock_config.GEMINI_MODEL = "gemini-pro"
mock_config.GEMINI_TEMPERATURE = 0.7
mock_config.GEMINI_TOP_P = 0.9
mock_config.GEMINI_TOP_K = 40
mock_config.GEMINI_MAX_OUTPUT_TOKENS = 2048

# Mock model configuration
mock_model_config = MagicMock()
mock_model_config.MODEL_FALLBACKS = {
    "default": "gemini-pro",
    "gemini-pro": ["gpt-4", "gpt-3.5-turbo"]
}
mock_model_config.MODEL_SETTINGS = {
    "gemini-pro": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 2048
    }
}

sys.modules['id8r.config'] = mock_config
sys.modules['id8r.config.model_config'] = mock_model_config
