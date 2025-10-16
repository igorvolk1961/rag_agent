"""
Tests for document processor
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.document_processor import TextProcessor, PDFProcessor, DocumentProcessorFactory
from src.models.schemas import Document, Chunk


class TestTextProcessor:
    """Test cases for TextProcessor"""
    
    def test_text_processor_initialization(self):
        """Test TextProcessor initialization"""
        processor = TextProcessor(chunk_size=500, chunk_overlap=100)
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 100
    
    @patch('src.utils.file_utils.FileUtils.read_text_file')
    def test_process_document(self, mock_read_file):
        """Test document processing"""
        mock_read_file.return_value = "This is a test document content."
        
        processor = TextProcessor()
        file_path = Path("test.txt")
        
        document = processor.process_document(file_path)
        
        assert isinstance(document, Document)
        assert document.id == "test"
        assert document.title == "test"
        assert document.content == "This is a test document content."
        assert document.metadata["type"] == "text"
    
    def test_chunk_document(self):
        """Test document chunking"""
        processor = TextProcessor(chunk_size=10, chunk_overlap=2)
        
        document = Document(
            id="test",
            title="Test Document",
            content="This is a test document with some content.",
            metadata={"type": "text"}
        )
        
        chunks = processor.chunk_document(document)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.document_id == "test" for chunk in chunks)


class TestPDFProcessor:
    """Test cases for PDFProcessor"""
    
    def test_pdf_processor_initialization(self):
        """Test PDFProcessor initialization"""
        processor = PDFProcessor()
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 200
    
    @patch('src.utils.file_utils.FileUtils.read_pdf_file')
    def test_process_pdf_document(self, mock_read_pdf):
        """Test PDF document processing"""
        mock_read_pdf.return_value = "PDF content here."
        
        processor = PDFProcessor()
        file_path = Path("test.pdf")
        
        document = processor.process_document(file_path)
        
        assert isinstance(document, Document)
        assert document.id == "test"
        assert document.metadata["type"] == "pdf"


class TestDocumentProcessorFactory:
    """Test cases for DocumentProcessorFactory"""
    
    def test_get_text_processor(self):
        """Test getting text processor"""
        file_path = Path("test.txt")
        processor = DocumentProcessorFactory.get_processor(file_path)
        assert isinstance(processor, TextProcessor)
    
    def test_get_pdf_processor(self):
        """Test getting PDF processor"""
        file_path = Path("test.pdf")
        processor = DocumentProcessorFactory.get_processor(file_path)
        assert isinstance(processor, PDFProcessor)
    
    def test_unsupported_file_type(self):
        """Test unsupported file type"""
        file_path = Path("test.xyz")
        with pytest.raises(ValueError, match="Unsupported file type"):
            DocumentProcessorFactory.get_processor(file_path)
