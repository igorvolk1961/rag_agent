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
        settings.llm_model = "gpt-3.5-turbo"
        settings.openai_api_key = "test-key"
        settings.vector_store_type = "chroma"
        settings.vector_store_path = Path("test_store")
        return settings
    
    @patch('src.core.rag_engine.Embedder')
    @patch('src.core.rag_engine.LLMClient')
    @patch('src.core.rag_engine.VectorStore')
    def test_rag_engine_initialization(self, mock_vector_store, mock_llm_client, mock_embedder, mock_settings):
        """Test RAGEngine initialization"""
        engine = RAGEngine(mock_settings)
        
        assert engine.settings == mock_settings
        mock_embedder.assert_called_once_with("test-model")
        mock_llm_client.assert_called_once_with("gpt-3.5-turbo", "test-key")
        mock_vector_store.assert_called_once_with("chroma", Path("test_store"))
    
    @patch('src.core.rag_engine.DocumentProcessorFactory')
    @patch('src.core.rag_engine.RAGEngine.embedder')
    @patch('src.core.rag_engine.RAGEngine.vector_store')
    def test_add_documents(self, mock_vector_store, mock_embedder, mock_factory, mock_settings):
        """Test adding documents to the knowledge base"""
        # Setup mocks
        mock_processor = Mock()
        mock_document = Mock(spec=Document)
        mock_chunks = [Mock(spec=Chunk), Mock(spec=Chunk)]
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        
        mock_factory.get_processor.return_value = mock_processor
        mock_processor.process_document.return_value = mock_document
        mock_processor.chunk_document.return_value = mock_chunks
        mock_embedder.embed_texts.return_value = mock_embeddings
        
        # Create engine and test
        engine = RAGEngine(mock_settings)
        file_paths = [Path("test1.txt"), Path("test2.txt")]
        
        engine.add_documents(file_paths)
        
        # Verify calls
        assert mock_factory.get_processor.call_count == 2
        assert mock_processor.process_document.call_count == 2
        assert mock_processor.chunk_document.call_count == 2
        assert mock_embedder.embed_texts.call_count == 2
        assert mock_vector_store.add_chunks.call_count == 2
    
    @patch('src.core.rag_engine.RAGEngine.embedder')
    @patch('src.core.rag_engine.RAGEngine.vector_store')
    @patch('src.core.rag_engine.RAGEngine.llm_client')
    def test_query(self, mock_llm_client, mock_vector_store, mock_embedder, mock_settings):
        """Test querying the RAG system"""
        # Setup mocks
        mock_query_embedding = [0.1, 0.2, 0.3]
        mock_chunks = [Mock(spec=Chunk), Mock(spec=Chunk)]
        mock_chunks[0].content = "First chunk content"
        mock_chunks[0].metadata = {"source": "doc1"}
        mock_chunks[1].content = "Second chunk content"
        mock_chunks[1].metadata = {"source": "doc2"}
        
        mock_embedder.embed_texts.return_value = [mock_query_embedding]
        mock_vector_store.search.return_value = mock_chunks
        mock_llm_client.generate_response.return_value = "Generated answer"
        
        # Create engine and test
        engine = RAGEngine(mock_settings)
        query_text = "What is the main topic?"
        
        response = engine.query(query_text)
        
        # Verify response
        assert isinstance(response, Response)
        assert response.query.text == query_text
        assert response.answer == "Generated answer"
        assert len(response.sources) == 2
        assert response.metadata["chunks_found"] == 2
        
        # Verify calls
        mock_embedder.embed_texts.assert_called_once_with([query_text])
        mock_vector_store.search.assert_called_once_with(mock_query_embedding, top_k=5)
        mock_llm_client.generate_response.assert_called_once()
    
    @patch('src.core.rag_engine.RAGEngine.vector_store')
    def test_get_stats(self, mock_vector_store, mock_settings):
        """Test getting statistics"""
        mock_vector_store.get_document_count.return_value = 10
        mock_vector_store.get_chunk_count.return_value = 50
        
        engine = RAGEngine(mock_settings)
        stats = engine.get_stats()
        
        assert stats["total_documents"] == 10
        assert stats["total_chunks"] == 50
        assert stats["embedding_model"] == "test-model"
        assert stats["llm_model"] == "gpt-3.5-turbo"
        assert stats["vector_store_type"] == "chroma"
    
    def test_prepare_context(self, mock_settings):
        """Test context preparation"""
        engine = RAGEngine(mock_settings)
        
        mock_chunks = [
            Mock(spec=Chunk),
            Mock(spec=Chunk)
        ]
        mock_chunks[0].content = "First chunk"
        mock_chunks[1].content = "Second chunk"
        
        context = engine._prepare_context(mock_chunks)
        
        assert "Document 1:" in context
        assert "First chunk" in context
        assert "Document 2:" in context
        assert "Second chunk" in context
