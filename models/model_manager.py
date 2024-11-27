"""
AI model management system supporting multiple model providers.

This module handles:
- Model initialization and configuration
- Switching between different AI models
- Error handling and fallback mechanisms
- Model availability testing
"""

from typing import Dict, Optional, Tuple, List
from openai import OpenAI
from config import Config
from utils.logger import Logger
import sys
import traceback


class ModelManager:
    """
    Manages multiple AI models and their configurations.

    Key responsibilities:
    - Initialize and manage model clients
    - Handle model switching
    - Provide fallback mechanisms
    - Test model availability
    """

    def __init__(self):
        """Initialize ModelManager with default settings and clients."""
        self.logger = Logger("model_manager")
        print("\n=== Initializing ModelManager ===")

        self.current_model = "OpenAI GPT-4"
        self.fallback_model = "OpenAI GPT-4"
        self.clients: Dict[str, Optional[OpenAI]] = {}

        # Initialize clients
        success = self._initialize_clients()
        if not success:
            print("WARNING: Failed to initialize all model clients")
            print("Available models:", self.get_available_models())

    def _initialize_clients(self) -> bool:
        """
        Initialize clients for all configured models.

        Returns:
            bool: True if at least one client was initialized successfully
        """
        success = False
        print("Available models config:", Config.AVAILABLE_MODELS)

        for model_name, model_config in Config.AVAILABLE_MODELS.items():
            try:
                print(f"\nInitializing client for {model_name}")
                print(f"Model config: {model_config}")

                client = OpenAI(
                    api_key=model_config["api_key"],
                    base_url=(
                        model_config["base_url"] if model_config["base_url"] else None
                    ),
                )

                # Test the client with a simple query
                print(f"Testing {model_name} with model: {model_config['model']}")
                test_response = client.chat.completions.create(
                    model=model_config["model"],
                    messages=[
                        {"role": "system", "content": "Test message"},
                        {"role": "user", "content": "Test"},
                    ],
                )

                if test_response and test_response.choices:
                    self.clients[model_name] = client
                    success = True
                    print(f"✓ Successfully initialized {model_name}")
                else:
                    print(f"✗ Failed to get valid response from {model_name}")

            except Exception as e:
                print(f"✗ Error initializing {model_name}:")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {str(e)}")
                print(f"  Traceback:\n{traceback.format_exc()}")
                self.logger.error(
                    f"Failed to initialize client for {model_name}: {str(e)}"
                )
                self.clients[model_name] = None

        return success

    def get_available_models(self) -> List[str]:
        """
        Get list of currently available models.

        Returns:
            List[str]: Names of available models
        """
        available = [
            name for name, client in self.clients.items() if client is not None
        ]
        print(f"Available models: {available}")
        return available

    def get_current_model(self) -> str:
        """
        Get name of currently active model.

        Returns:
            str: Name of current model
        """
        return self.current_model

    def set_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Switch to a different model.

        Args:
            model_name: Name of the model to switch to

        Returns:
            Tuple[bool, str]: Success status and message
        """
        print(f"\nAttempting to set model to: {model_name}")
        print(f"Available models: {self.get_available_models()}")

        if model_name not in self.clients:
            msg = f"Model {model_name} not available"
            print(f"✗ {msg}")
            return False, msg

        if self.clients[model_name] is None:
            msg = f"Model {model_name} failed to initialize"
            print(f"✗ {msg}")
            return False, msg

        try:
            # Test model availability
            print(f"Testing {model_name}...")
            self._test_model(model_name)
            self.current_model = model_name
            msg = f"Successfully switched to {model_name}"
            print(f"✓ {msg}")
            return True, msg
        except Exception as e:
            msg = f"Failed to switch to {model_name}: {str(e)}"
            print(f"✗ {msg}")
            print(f"Traceback:\n{traceback.format_exc()}")
            self.logger.error(msg)
            return False, msg

    def _test_model(self, model_name: str) -> None:
        """
        Test if a model is available and responding.

        Args:
            model_name: Name of the model to test

        Raises:
            Exception: If model test fails
        """
        try:
            print(f"Testing model: {model_name}")
            client = self.clients[model_name]
            if client is None:
                raise Exception("Model client is not initialized")

            model_config = Config.AVAILABLE_MODELS[model_name]
            print(f"Model config: {model_config}")

            response = client.chat.completions.create(
                model=model_config["model"],
                messages=[
                    {"role": "system", "content": "Test message"},
                    {"role": "user", "content": "Test"},
                ],
            )

            if not response or not response.choices:
                raise Exception("No response from model")

            print(f"✓ Model test successful for {model_name}")

        except Exception as e:
            print(f"✗ Model test failed: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            raise Exception(f"Model test failed: {str(e)}")

    def get_client(self, model_name: Optional[str] = None) -> Optional[OpenAI]:
        """
        Get client for specified model or current model.

        Args:
            model_name: Optional name of model to get client for

        Returns:
            Optional[OpenAI]: Client instance for the requested model or None if not available
        """
        model = model_name or self.current_model
        print(f"\nGetting client for model: {model}")

        if model not in self.clients:
            print(f"Model {model} not found, trying fallback: {self.fallback_model}")
            model = self.fallback_model

        client = self.clients.get(model)
        if client is None:
            print(f"✗ No client available for {model}")
            # Try fallback if different from requested model
            if model != self.fallback_model:
                print(f"Trying fallback model: {self.fallback_model}")
                client = self.clients.get(self.fallback_model)
        else:
            print(f"✓ Found client for {model}")

        return client
