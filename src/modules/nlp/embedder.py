"""
Text embedding functionality using LangChain
"""

import logging
from typing import List, Optional
import numpy as np

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings


class Embedder:
    """Text embedding using LangChain embeddings"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_type: str = "huggingface", api_key: Optional[str] = None):
        self.model_name = model_name
        self.embedding_type = embedding_type
        self.logger = logging.getLogger(__name__)

        try:
            if embedding_type == "openai":
                if not api_key:
                    raise ValueError("OpenAI API key is required for OpenAI embeddings")
                self.embeddings = OpenAIEmbeddings(
                    model=model_name,
                    openai_api_key=api_key
                )
            else:  # huggingface
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )

            self.logger.info(f"Loaded {embedding_type} embedding model: {model_name}")
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            self.logger.error(f"Error embedding text: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            self.logger.error(f"Error embedding texts: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        try:
            # Test embedding to get dimension
            test_embedding = self.embed_text("test")
            return len(test_embedding)
        except Exception as e:
            self.logger.error(f"Error getting embedding dimension: {e}")
            return 384  # Default for all-MiniLM-L6-v2

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def get_langchain_embeddings(self) -> Embeddings:
        """Get the underlying LangChain embeddings object"""
        return self.embeddings
