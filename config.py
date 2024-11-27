"""
Configuration settings for the application.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()


class Config:
    """Configuration class for the application."""

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Please set OPENAI_API_KEY in .env file")

    # Model Configuration
    OPENAI_CHAT_MODEL = "gpt-4"
    OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
    OLLAMA_BASE_URL = "http://localhost:11434/v1"  # Changed: Removed /v1
    OLLAMA_MODEL = "llama3.2"  # Changed: Using exact model name

    # ChromaDB Configuration
    CHROMA_COLLECTION_NAME = "financial_data"
    CHROMA_PERSIST_DIR = "chroma_db"

    # Cache Configuration
    CACHE_DIR = "cache"
    CACHE_EXPIRATION_HOURS = 24

    # Query Configuration
    DEFAULT_N_RESULTS = 5

    # Model Settings
    AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
        "OpenAI GPT-4": {
            "type": "openai",
            "model": OPENAI_CHAT_MODEL,
            "api_key": OPENAI_API_KEY,
            "base_url": None,
        },
        "Local Llama3.2": {  # Changed: Updated model name
            "type": "ollama",
            "model": OLLAMA_MODEL,
            "api_key": "ollama",
            "base_url": OLLAMA_BASE_URL,  # Changed: Using base URL without /v1
        },
    }

    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_DIR = "logs"

    # Data Validation
    MAX_FILE_SIZE_MB = 100
    ALLOWED_EXTENSIONS = [".csv"]
    MIN_ROWS = 1
    MAX_COLUMNS = 100

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings."""
        try:
            # Validate API keys
            if cls.OPENAI_API_KEY is None:
                raise ValueError("OpenAI API key is missing")

            # Validate model configurations
            for model_name, model_config in cls.AVAILABLE_MODELS.items():
                if not all(
                    k in model_config for k in ["type", "model", "api_key", "base_url"]
                ):
                    raise ValueError(f"Invalid configuration for model {model_name}")

            # Validate directories exist
            for directory in [cls.CHROMA_PERSIST_DIR, cls.CACHE_DIR, cls.LOG_DIR]:
                os.makedirs(directory, exist_ok=True)

            return True
        except Exception as e:
            print(f"Configuration validation failed: {str(e)}")
            return False
