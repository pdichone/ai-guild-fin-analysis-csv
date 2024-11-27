"""
AI-powered conversation handler for natural language data queries.

This module manages:
- Natural language query processing
- Model switching between OpenAI and Llama3.2
- Context management for conversations
- Data retrieval and response generation
"""

from openai import OpenAI
import json
from datetime import datetime
from config import Config
from utils.logger import Logger
import pandas as pd
from io import StringIO
import streamlit as st
import sys
import traceback


class ConversationHandler:
    """
    Handles natural language conversations about data.

    Key responsibilities:
    - Process natural language queries
    - Manage AI model selection
    - Maintain conversation context
    - Generate context-aware responses
    """

    def __init__(self, data_embedder, model_manager):
        """
        Initialize ConversationHandler with required components.

        Args:
            data_embedder: Instance of DataEmbedder for data retrieval
            model_manager: Instance of ModelManager for AI model handling
        """
        print(f"\n{'='*50}")
        print("INITIALIZING CONVERSATION HANDLER")
        print(f"{'='*50}")

        self.logger = Logger("conversation_handler")

        if not data_embedder:
            print("ERROR: data_embedder is None")
            raise ValueError("data_embedder cannot be None")

        if not model_manager:
            print("ERROR: model_manager is None")
            raise ValueError("model_manager cannot be None")

        self.data_embedder = data_embedder
        self.model_manager = model_manager
        self.context = []
        self.current_model = "OpenAI GPT-4"  # Default model

        print("✓ ConversationHandler initialized successfully")
        print(f"{'='*50}\n")

    def get_current_model(self) -> str:
        """Get the name of the currently active model."""
        return self.current_model

    def set_model(self, model_name: str) -> tuple[bool, str]:
        """Set the active model."""
        print(f"\nAttempting to set model to: {model_name}")
        if model_name not in self.model_manager.get_available_models():
            print(f"ERROR: Model {model_name} not available")
            return False, f"Model {model_name} not available"
        self.current_model = model_name
        print(f"✓ Successfully set model to: {model_name}")
        return True, f"Switched to {model_name}"

    def _extract_numeric_values(self, text: str) -> dict:
        """Extract numeric values from text for validation."""
        values = {}
        try:
            # Look for patterns like "$X" or "X dollars" or just numbers
            import re

            # Find all monetary values
            monetary_values = re.findall(r"\$?([\d,]+\.?\d*)", text)
            if monetary_values:
                values["monetary"] = [
                    float(v.replace(",", "")) for v in monetary_values
                ]

            # Find all percentage values
            percentage_values = re.findall(r"([\d.]+)%", text)
            if percentage_values:
                values["percentages"] = [float(v) for v in percentage_values]

            print(f"Extracted numeric values: {values}")
            return values
        except Exception as e:
            print(f"Error extracting numeric values: {str(e)}")
            return {}

    def _validate_response(self, response: str, context_data: dict) -> bool:
        """Validate response against context data."""
        try:
            # Extract numeric values from response
            response_values = self._extract_numeric_values(response)
            if not response_values:
                print("No numeric values found in response")
                return True  # No values to validate

            # Extract numeric values from context
            context_values = {}
            for metadata in context_data.get("metadatas", [[]])[0]:
                if metadata:
                    for key, value in metadata.items():
                        try:
                            if isinstance(value, str) and any(
                                c.isdigit() for c in value
                            ):
                                context_values[key] = float(
                                    value.replace(",", "").replace("$", "")
                                )
                        except ValueError:
                            continue

            print(f"Context values: {context_values}")

            # Validate that response values are within reasonable range of context values
            for values in response_values.values():
                for value in values:
                    if value > max(context_values.values()) * 1.1:  # Allow 10% margin
                        print(f"Value {value} exceeds maximum context value")
                        return False

            return True
        except Exception as e:
            print(f"Error validating response: {str(e)}")
            return True  # Default to accepting response if validation fails

    def process_query(self, query: str) -> str:
        """Process a natural language query about the data."""
        print(f"\n{'='*50}")
        print(f"PROCESSING QUERY: {query}")
        print(f"{'='*50}")

        try:
            # Get relevant context from ChromaDB
            print("\nStep 1: Fetching relevant data from ChromaDB...")
            relevant_data = self.data_embedder.query_data(query)

            print(f"\nRelevant data structure: {json.dumps(relevant_data, indent=2)}")

            # Validate the response structure
            if not relevant_data:
                print("ERROR: No response from data_embedder.query_data")
                return "I'm having trouble accessing the data. Please try again."

            if not isinstance(relevant_data, dict):
                print(f"ERROR: Invalid response type: {type(relevant_data)}")
                return "The data structure seems incorrect. Please try again."

            # Check for required fields
            required_fields = ["documents", "metadatas", "distances"]
            missing_fields = [
                field for field in required_fields if field not in relevant_data
            ]
            if missing_fields:
                print(f"ERROR: Missing required fields: {missing_fields}")
                return "The data structure seems incomplete. Please try a different question."

            # Check for empty results
            if not relevant_data["documents"] or not relevant_data["documents"][0]:
                print("WARNING: No relevant data found for query")
                return "I couldn't find any relevant data to answer your question. Please try rephrasing or ask about different aspects of the data."

            # Prepare context entries
            print("\nStep 2: Preparing context entries...")
            context_entries = []

            # Validate documents and metadata arrays match
            if len(relevant_data["documents"][0]) != len(relevant_data["metadatas"][0]):
                print("ERROR: Mismatch between documents and metadata length")
                return "There seems to be a data inconsistency. Please try again."

            for doc, metadata in zip(
                relevant_data["documents"][0], relevant_data["metadatas"][0]
            ):
                if doc and metadata:
                    source = metadata.get("source_name", "unknown source")
                    context_entry = f"From {source}: {doc}"
                    context_entries.append(context_entry)
                    print(
                        f"Added context entry: {context_entry[:100]}..."
                    )  # Print first 100 chars

            if not context_entries:
                print("ERROR: No valid context entries generated")
                return "I couldn't process the data properly. Please try a different question."

            # Display context data for debugging
            print("\nStep 3: Context Data:")
            for i, entry in enumerate(context_entries):
                print(f"Context {i+1}: {entry[:100]}...")

            # Prepare the prompt
            print("\nStep 4: Preparing prompt for AI model...")
            system_prompt = """You are a data analysis assistant. Analyze the provided context and answer questions about the data. 
            Important guidelines:
            - Be specific and include numerical values
            - Format monetary values with $ and commas
            - Cite the source dataset when providing information
            - If aggregating across multiple records, explain the calculation
            - If the answer requires assumptions, state them clearly
            - For calculations involving monetary values, ensure you're using the correct aggregation level
            - When dealing with project data, aggregate at the project level first
            """

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Based on this context:\n{' '.join(context_entries)}\n\nQuestion: {query}",
                },
            ]

            print("\nStep 5: Getting AI model client...")
            client = self.model_manager.get_client()
            if not client:
                print("ERROR: Failed to get model client")
                return "Error: AI model not available. Please try again later."

            model_config = Config.AVAILABLE_MODELS.get(self.current_model)
            if not model_config:
                print(f"ERROR: Model configuration not found for {self.current_model}")
                return "Error: Model configuration not found."

            print(f"\nStep 6: Sending request to {model_config['model']}...")
            if model_config["type"] == "ollama":
                # For Ollama models - use exact model name "llama3.2"
                client = OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",  # required, but unused
                )
                print("Using Ollama model with exact name: llama3.2")
                response = client.chat.completions.create(
                    model="llama3.2",  # Use exact model name
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500,
                )
            else:
                # For OpenAI models
                response = client.chat.completions.create(
                    model=model_config["model"],
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500,
                )

            print("\nStep 7: Processing AI response...")
            if not response or not response.choices:
                print("ERROR: Invalid response from model")
                return "I received an invalid response. Please try again."

            response_content = response.choices[0].message.content
            print(f"Raw response: {response_content}")

            # Validate response
            if not self._validate_response(response_content, relevant_data):
                print("ERROR: Response validation failed")
                return "I apologize, but I may have generated incorrect calculations. Please try rephrasing your question."

            print("\n✓ Query processing completed successfully")
            print(f"{'='*50}\n")

            return response_content

        except Exception as e:
            print(f"\nERROR processing query: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            self.logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return "I encountered an error while processing your question. Please try again or rephrase your question."

    def update_context(self, query: str, response: str) -> None:
        """Update conversation context with new query and response."""
        self.context.append(
            {
                "query": query,
                "response": response,
                "timestamp": datetime.now().isoformat(),
            }
        )
