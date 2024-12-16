"""Configuration for model fallbacks and settings."""

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
    "gpt-4": {
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": 4000
    },
    "claude-3-opus": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4000
    }
}

# Retry settings
RETRY_SETTINGS = {
    "max_retries": 3,
    "initial_delay": 1,
    "backoff_factor": 2,
    "max_delay": 10
}
