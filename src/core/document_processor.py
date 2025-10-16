"""
Document processing functionality for RAG Agent
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from ..models.schemas import Document, Chunk
from ..utils.file_utils import FileUtils


class DocumentProcessor(ABC):
    """Abstract base class for document processors"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def process_document(self, file_path: Path) -> Document:
        """Process a single document and return Document object"""
        pass
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split document into chunks"""
        chunks = []
        text = document.content
        
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                if last_period > self.chunk_size * 0.7:  # If period is in last 30%
                    end = start + last_period + 1
                    chunk_text = text[start:end]
            
            chunk = Chunk(
                id=f"{document.id}_chunk_{chunk_id}",
                document_id=document.id,
                content=chunk_text,
                start_pos=start,
                end_pos=end,
                metadata={
                    "chunk_id": chunk_id,
                    "document_title": document.title,
                    "document_type": document.metadata.get("type", "unknown")
                }
            )
            
            chunks.append(chunk)
            chunk_id += 1
            start = end - self.chunk_overlap
            
            if start >= len(text):
                break
        
        self.logger.info(f"Created {len(chunks)} chunks for document {document.id}")
        return chunks


class TextProcessor(DocumentProcessor):
    """Processor for plain text documents"""
    
    def process_document(self, file_path: Path) -> Document:
        """Process a plain text document"""
        try:
            content = FileUtils.read_text_file(file_path)
            
            document = Document(
                id=file_path.stem,
                title=file_path.stem,
                content=content,
                metadata={
                    "type": "text",
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size
                }
            )
            
            self.logger.info(f"Processed text document: {file_path}")
            return document
            
        except Exception as e:
            self.logger.error(f"Error processing text document {file_path}: {e}")
            raise


class PDFProcessor(DocumentProcessor):
    """Processor for PDF documents"""
    
    def process_document(self, file_path: Path) -> Document:
        """Process a PDF document"""
        try:
            content = FileUtils.read_pdf_file(file_path)
            
            document = Document(
                id=file_path.stem,
                title=file_path.stem,
                content=content,
                metadata={
                    "type": "pdf",
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size
                }
            )
            
            self.logger.info(f"Processed PDF document: {file_path}")
            return document
            
        except Exception as e:
            self.logger.error(f"Error processing PDF document {file_path}: {e}")
            raise


class DocumentProcessorFactory:
    """Factory for creating document processors"""
    
    @staticmethod
    def get_processor(file_path: Path) -> DocumentProcessor:
        """Get appropriate processor for file type"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            return TextProcessor()
        elif suffix == '.pdf':
            return PDFProcessor()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
