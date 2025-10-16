"""
Tests for document processor using LangChain
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.document_processor import DocumentProcessor
from src.models.schemas import Document, Chunk


class TestDocumentProcessor:
    """Test cases for DocumentProcessor using LangChain"""

    def test_document_processor_initialization(self):
        """Test DocumentProcessor initialization"""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 100

    @patch('src.core.document_processor.TextLoader')
    @patch('os.stat')
    def test_load_text_document(self, mock_stat, mock_text_loader):
        """Test loading text document"""
        # Mock file stats
        mock_stat.return_value.st_size = 100

        mock_doc = Mock()
        mock_doc.page_content = "This is a test document content."
        mock_doc.metadata = {"source": "test.txt"}
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_text_loader.return_value = mock_loader_instance

        processor = DocumentProcessor()
        file_path = Path("test.txt")

        document = processor.load_document(file_path)

        assert document.page_content == "This is a test document content."
        assert document.metadata["source"] == "test.txt"

    @patch('src.core.document_processor.TextLoader')
    @patch('os.stat')
    def test_process_document(self, mock_stat, mock_text_loader):
        """Test document processing"""
        # Mock file stats
        mock_stat.return_value.st_size = 100

        mock_doc = Mock()
        mock_doc.page_content = "This is a test document content."
        mock_doc.metadata = {"source": "test.txt"}
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_text_loader.return_value = mock_loader_instance

        processor = DocumentProcessor()
        file_path = Path("test.txt")

        document = processor.process_document(file_path)

        assert isinstance(document, Document)
        assert document.id == "test"
        assert document.title == "test"
        assert document.content == "This is a test document content."
        assert document.metadata["type"] == "txt"

    def test_chunk_document(self):
        """Test document chunking"""
        processor = DocumentProcessor(chunk_size=10, chunk_overlap=2)

        document = Document(
            id="test",
            title="Test Document",
            content="This is a test document with some content.",
            metadata={"type": "txt"}
        )

        chunks = processor.chunk_document(document)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.document_id == "test" for chunk in chunks)


    @patch('src.core.document_processor.PyPDFLoader')
    @patch('os.stat')
    def test_load_pdf_document(self, mock_stat, mock_pdf_loader):
        """Test loading PDF document"""
        # Mock file stats
        mock_stat.return_value.st_size = 100

        mock_doc = Mock()
        mock_doc.page_content = "PDF content here."
        mock_doc.metadata = {"source": "test.pdf", "page": 1}
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_pdf_loader.return_value = mock_loader_instance

        processor = DocumentProcessor()
        file_path = Path("test.pdf")

        document = processor.load_document(file_path)

        assert document.page_content == "PDF content here."
        assert document.metadata["source"] == "test.pdf"

    def test_get_supported_extensions(self):
        """Test getting supported file extensions"""
        processor = DocumentProcessor()
        extensions = processor.get_supported_extensions()

        assert '.txt' in extensions
        assert '.pdf' in extensions
        assert '.docx' in extensions

    def test_is_supported_file(self):
        """Test checking if file type is supported"""
        processor = DocumentProcessor()

        assert processor.is_supported_file(Path("test.txt")) == True
        assert processor.is_supported_file(Path("test.pdf")) == True
        assert processor.is_supported_file(Path("test.docx")) == True
        assert processor.is_supported_file(Path("test.xyz")) == False

    @patch('os.stat')
    def test_unsupported_file_type(self, mock_stat):
        """Test unsupported file type"""
        # Mock file stats
        mock_stat.return_value.st_size = 100

        processor = DocumentProcessor()
        file_path = Path("test.xyz")

        with pytest.raises(ValueError, match="Unsupported file type"):
            processor.load_document(file_path)
