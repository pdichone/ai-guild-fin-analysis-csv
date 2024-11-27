"""
ChromaDB-based data embedding and retrieval system for CSV data analysis.

This module handles:
- Vector database management using ChromaDB
- Data embedding and storage
- Query processing and retrieval
- Consistent data aggregation across multiple files
"""

import chromadb
import pandas as pd
import numpy as np
from chromadb.utils import embedding_functions
import os
import hashlib
from config import Config
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from utils.logger import Logger


class DataEmbedder:
    """
    Manages data embedding and retrieval using ChromaDB.

    Key responsibilities:
    - Initialize and manage ChromaDB connection
    - Create and store embeddings for CSV data
    - Handle data retrieval for natural language queries
    - Maintain data consistency across multiple files
    - Track file hashes to prevent duplicate embeddings
    """

    def __init__(self):
        """Initialize DataEmbedder with ChromaDB and OpenAI embedding function."""
        self.logger = Logger("data_embedder")

        if not Config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found")

        # Initialize directory for persistent storage
        self.persist_dir = os.path.join(os.getcwd(), Config.CHROMA_PERSIST_DIR)
        os.makedirs(self.persist_dir, exist_ok=True)

        # Set up OpenAI embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=Config.OPENAI_API_KEY, model_name=Config.OPENAI_EMBEDDING_MODEL
        )

        # Initialize ChromaDB client and collection
        self._initialize_chroma()

        # Track metadata and file hashes
        self.file_metadata: Dict[str, Dict] = {}
        self.file_hashes: Dict[str, str] = {}

    def _initialize_chroma(self) -> None:
        """Initialize ChromaDB client and collection with error handling."""
        try:
            # First, try to create a new client
            self.client = chromadb.PersistentClient(path=self.persist_dir)

            try:
                # Try to get existing collection
                self.collection = self.client.get_collection(
                    name=Config.CHROMA_COLLECTION_NAME,
                    embedding_function=self.embedding_function,
                )
            except Exception:
                # If collection doesn't exist, create a new one
                self.collection = self.client.create_collection(
                    name=Config.CHROMA_COLLECTION_NAME,
                    embedding_function=self.embedding_function,
                )
        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise

    def query_data(
        self,
        query_text: str,
        n_results: int = Config.DEFAULT_N_RESULTS,
        filter_dict: Optional[dict] = None,
    ) -> Dict:
        """
        Query the vector database with error handling and default response.

        Args:
            query_text: The query string
            n_results: Number of results to return
            filter_dict: Optional filter criteria

        Returns:
            Dict containing query results with default structure if error occurs
        """
        try:
            # Ensure collection exists
            if not hasattr(self, "collection"):
                self.logger.error("ChromaDB collection not initialized")
                return self._get_default_response()

            # Execute query
            results = self.collection.query(
                query_texts=[query_text], n_results=n_results, where=filter_dict
            )

            # Validate results
            if not results or not isinstance(results, dict):
                self.logger.error("Invalid query results structure")
                return self._get_default_response()

            # Ensure all required fields are present
            required_fields = ["documents", "metadatas", "distances"]
            if not all(field in results for field in required_fields):
                self.logger.error("Missing required fields in query results")
                return self._get_default_response()

            return results

        except Exception as e:
            self.logger.error(f"Error querying data: {str(e)}")
            return self._get_default_response()

    def _get_default_response(self) -> Dict:
        """
        Get default response structure for error cases.

        Returns:
            Dict with empty results in expected format
        """
        return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

    def embed_data(
        self, df: pd.DataFrame, column_info: dict, source_name: Optional[str] = None
    ) -> bool:
        """
        Embed data from DataFrame into vector database.

        Args:
            df: DataFrame to embed
            column_info: Dictionary containing column type information
            source_name: Optional name of the data source

        Returns:
            bool: True if embedding successful, False otherwise
        """
        try:
            # Generate unique file ID and source name
            file_id = str(len(self.file_metadata))
            source_name = source_name or f"dataset_{file_id}"

            # Check if file is already embedded
            if self._is_file_embedded(df, source_name):
                self.logger.info(f"File {source_name} already embedded")
                return True

            # Store file metadata
            self.file_metadata[file_id] = {
                "source_name": source_name,
                "num_rows": len(df),
                "columns": list(df.columns),
                "column_info": column_info,
            }

            # Calculate and store file hash
            self.file_hashes[source_name] = self._calculate_file_hash(df)

            # Prepare and add data in batches
            batch_size = 100
            for start_idx in range(0, len(df), batch_size):
                end_idx = min(start_idx + batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx]

                documents, metadatas, ids = self._prepare_batch_data(
                    batch_df, column_info, file_id, source_name
                )

                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

            return True

        except Exception as e:
            self.logger.error(f"Error embedding data: {str(e)}")
            return False

    def _calculate_file_hash(self, df: pd.DataFrame) -> str:
        """Calculate unique hash for DataFrame content."""
        return hashlib.sha256(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()

    def _is_file_embedded(self, df: pd.DataFrame, source_name: str) -> bool:
        """Check if file is already embedded by comparing content hashes."""
        file_hash = self._calculate_file_hash(df)
        return (
            source_name in self.file_hashes
            and self.file_hashes[source_name] == file_hash
        )

    def _prepare_batch_data(
        self, df: pd.DataFrame, column_info: dict, file_id: str, source_name: str
    ) -> Tuple[List[str], List[dict], List[str]]:
        """Prepare data batch for embedding."""
        documents = []
        metadatas = []
        ids = []

        for idx, row in df.iterrows():
            # Create document description
            doc = self._prepare_row_description(row, df)

            # Create metadata
            metadata = {"file_id": file_id, "source_name": source_name}

            # Add all column values to metadata
            for col in df.columns:
                val = row[col]
                # Convert numpy/pandas types to Python native types
                if isinstance(val, (np.integer, np.floating)):
                    val = float(val)
                elif isinstance(val, pd.Timestamp):
                    val = val.isoformat()
                metadata[col] = str(val)

            documents.append(doc)
            metadatas.append(metadata)
            ids.append(f"file_{file_id}_row_{idx}")

        return documents, metadatas, ids

    def _prepare_row_description(self, row: pd.Series, df: pd.DataFrame) -> str:
        """Create natural language description of a row."""
        parts = []

        for col in df.columns:
            # Format date columns
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                parts.append(f"on {row[col].strftime('%Y-%m-%d')}")
            # Format numeric values
            elif pd.api.types.is_numeric_dtype(df[col]):
                if any(
                    term in col.lower()
                    for term in ["amount", "budget", "price", "cost", "revenue"]
                ):
                    parts.append(f"the {col.replace('_', ' ')} was ${row[col]:,.2f}")
                else:
                    parts.append(f"the {col.replace('_', ' ')} was {row[col]:,}")
            # Format categorical values
            else:
                parts.append(f"the {col.replace('_', ' ')} was {row[col]}")

        return ", ".join(parts)

    def clear_data(self) -> bool:
        """Clear all data from the collection."""
        try:
            self.collection.delete(where={})
            self.file_metadata.clear()
            self.file_hashes.clear()
            return True
        except Exception as e:
            self.logger.error(f"Error clearing data: {str(e)}")
            return False

    def get_source_names(self) -> List[str]:
        """Get list of all source names in the collection."""
        return list(
            set(metadata["source_name"] for metadata in self.file_metadata.values())
        )
