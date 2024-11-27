import pandas as pd
from typing import Dict, List, Optional, Tuple
import re


class DataValidator:
    @staticmethod
    def validate_csv(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate CSV data structure and content."""
        try:
            # Check if DataFrame is empty
            if df.empty:
                return False, "DataFrame is empty"

            # Check for minimum number of rows
            if len(df) < 1:
                return False, "DataFrame must contain at least one row"

            # Check for duplicate column names
            if len(df.columns) != len(set(df.columns)):
                return False, "DataFrame contains duplicate column names"

            # Check for missing values
            missing_counts = df.isnull().sum()
            if missing_counts.any():
                columns_with_missing = missing_counts[missing_counts > 0].index.tolist()
                return (
                    False,
                    f"Missing values found in columns: {', '.join(columns_with_missing)}",
                )

            return True, "Validation successful"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize column names to ensure compatibility."""
        df = df.copy()

        # Convert to lowercase and replace spaces/special chars with underscore
        df.columns = [
            re.sub(r"[^a-zA-Z0-9]", "_", col.lower().strip()) for col in df.columns
        ]

        # Remove consecutive underscores and trailing/leading underscores
        df.columns = [re.sub(r"_+", "_", col).strip("_") for col in df.columns]

        return df


class ConfigValidator:
    @staticmethod
    def validate_model_config(config: Dict) -> Tuple[bool, str]:
        """Validate model configuration."""
        required_fields = ["type", "model", "api_key", "base_url"]

        for model_name, model_config in config.items():
            # Check if all required fields are present
            missing_fields = [
                field for field in required_fields if field not in model_config
            ]
            if missing_fields:
                return (
                    False,
                    f"Missing required fields for {model_name}: {', '.join(missing_fields)}",
                )

            # Validate model type
            if model_config["type"] not in ["openai", "ollama"]:
                return (
                    False,
                    f"Invalid model type for {model_name}: {model_config['type']}",
                )

            # Validate API key for OpenAI models
            if model_config["type"] == "openai" and not model_config["api_key"]:
                return False, f"OpenAI API key is required for {model_name}"

        return True, "Configuration validation successful"
