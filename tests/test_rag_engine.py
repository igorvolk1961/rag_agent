"""
Tests for RAG Engine
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.core.rag_engine import RAGEngine
from src.config.settings import Settings
from src.models.schemas import Document, Chunk, Query, Response


class TestRAGEngine:
    """Test cases for RAGEngine"""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing"""
        settings = Mock(spec=Settings)
        settings.embedding_model = "test-model"
        settings.embedding_type = "huggingface"
        settings.llm_model = "deepseek-chat"
        settings.openai_api_key = "test-openai-key"
        settings.deepseek_api_key = "sk-163567b8e043481bb97dd4cbbc3a965b"
        settings.huggingface_api_key = None
        settings.vector_store_type = "chroma"
        settings.vector_store_path = Path("test_store")
        settings.chunk_size = 1000
        settings.chunk_overlap = 200
        settings.temperature = 0.7
        settings.max_tokens = None
        settings.collection_name = "rag_documents"
        return settings

    @patch('src.core.rag_engine.Embedder')
    @patch('src.core.rag_engine.LLMClient')
    @patch('src.core.rag_engine.VectorStoreManager')
    @patch('src.core.rag_engine.DocumentProcessor')
    def test_rag_engine_initialization(self, mock_document_processor, mock_vector_store, mock_llm_client, mock_embedder, mock_settings):
        """Test RAGEngine initialization"""
        engine = RAGEngine(mock_settings)

        assert engine.settings == mock_settings
        mock_embedder.assert_called_once_with(
            model_name="test-model",
            embedding_type="huggingface",
            api_key=None
        )
        mock_llm_client.assert_called_once_with(
            model="deepseek-chat",
            api_key="sk-163567b8e043481bb97dd4cbbc3a965b",
            temperature=0.7,
            max_tokens=None
        )
        mock_vector_store.assert_called_once()

    @patch('src.core.rag_engine.Embedder')
    @patch('src.core.rag_engine.LLMClient')
    @patch('src.core.rag_engine.VectorStoreManager')
    @patch('src.core.rag_engine.DocumentProcessor')
    def test_add_documents(self, mock_document_processor, mock_vector_store, mock_llm_client, mock_embedder, mock_settings):
        """Test adding documents to the knowledge base"""
        # Setup mocks
        mock_chunks = [Mock(spec=Chunk), Mock(spec=Chunk)]
        mock_document_processor.return_value.process_multiple_documents.return_value = mock_chunks

        # Create engine and test
        engine = RAGEngine(mock_settings)
        file_paths = [Path("test1.txt"), Path("test2.txt")]

        engine.add_documents(file_paths)

        # Verify calls
        mock_document_processor.return_value.process_multiple_documents.assert_called_once_with(file_paths)
        mock_vector_store.return_value.add_chunks.assert_called_once_with(mock_chunks)
        mock_vector_store.return_value.save.assert_called_once()

    @patch('src.core.rag_engine.Embedder')
    @patch('src.core.rag_engine.LLMClient')
    @patch('src.core.rag_engine.VectorStoreManager')
    @patch('src.core.rag_engine.DocumentProcessor')
    def test_query(self, mock_document_processor, mock_vector_store, mock_llm_client, mock_embedder, mock_settings):
        """Test querying the RAG system"""
        # Setup mocks
        mock_chunks = [Mock(spec=Chunk), Mock(spec=Chunk)]
        mock_chunks[0].content = "First chunk content"
        mock_chunks[0].metadata = {"source": "doc1"}
        mock_chunks[1].content = "Second chunk content"
        mock_chunks[1].metadata = {"source": "doc2"}

        mock_vector_store.return_value.search.return_value = mock_chunks

        # Create engine first
        engine = RAGEngine(mock_settings)

        # Mock the RAG chain invoke method
        mock_rag_chain = Mock()
        mock_rag_chain.invoke.return_value = "Generated answer"

        # Replace the engine's rag_chain property
        engine.rag_chain = mock_rag_chain

        query_text = "What is the main topic?"
        response = engine.query(query_text)

        # Verify response
        assert isinstance(response, Response)
        assert response.query.text == query_text
        assert response.answer == "Generated answer"
        assert len(response.sources) == 2
        assert response.metadata["chunks_found"] == 2

        # Verify calls
        mock_vector_store.return_value.search.assert_called_once_with(query_text, top_k=5)
        mock_rag_chain.invoke.assert_called_once_with(query_text)

    @patch('src.core.rag_engine.Embedder')
    @patch('src.core.rag_engine.LLMClient')
    @patch('src.core.rag_engine.VectorStoreManager')
    @patch('src.core.rag_engine.DocumentProcessor')
    def test_get_stats(self, mock_document_processor, mock_vector_store, mock_llm_client, mock_embedder, mock_settings):
        """Test getting statistics"""
        # Setup mocks
        mock_vector_store.return_value.get_document_count.return_value = 10
        mock_vector_store.return_value.get_chunk_count.return_value = 50

        engine = RAGEngine(mock_settings)
        stats = engine.get_stats()

        assert stats["total_documents"] == 10
        assert stats["total_chunks"] == 50
        assert stats["embedding_model"] == "test-model"
        assert stats["llm_model"] == "deepseek-chat"
        assert stats["vector_store_type"] == "chroma"
        assert stats["chunk_size"] == 1000
        assert stats["chunk_overlap"] == 200
