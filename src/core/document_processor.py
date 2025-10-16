"""
Document processing functionality using LangChain
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from ..models.schemas import Document, Chunk


class DocumentProcessor:
    """Document processor using LangChain loaders and splitters"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_document(self, file_path: Path) -> LangChainDocument:
        """Load document using LangChain loaders"""
        try:
            suffix = file_path.suffix.lower()

            if suffix == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif suffix == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif suffix == '.docx':
                loader = Docx2txtLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

            documents = loader.load()

            if not documents:
                raise ValueError(f"No content found in {file_path}")

            # Combine multiple pages into single document
            if len(documents) > 1:
                combined_content = "\n\n".join([doc.page_content for doc in documents])
                combined_metadata = documents[0].metadata.copy()
                combined_metadata.update({
                    "total_pages": len(documents),
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size
                })
                document = LangChainDocument(
                    page_content=combined_content,
                    metadata=combined_metadata
                )
            else:
                document = documents[0]
                document.metadata.update({
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size
                })

            self.logger.info(f"Loaded document: {file_path}")
            return document

        except Exception as e:
            self.logger.error(f"Error loading document {file_path}: {e}")
            raise

    def process_document(self, file_path: Path) -> Document:
        """Process a document and return Document object"""
        try:
            # Load document using LangChain
            langchain_doc = self.load_document(file_path)

            # Convert to our Document format
            document = Document(
                id=file_path.stem,
                title=file_path.stem,
                content=langchain_doc.page_content,
                metadata={
                    "type": file_path.suffix.lower().lstrip('.'),
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    **langchain_doc.metadata
                }
            )

            self.logger.info(f"Processed document: {file_path}")
            return document

        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {e}")
            raise

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split document into chunks using LangChain text splitter"""
        try:
            # Create LangChain document for splitting
            langchain_doc = LangChainDocument(
                page_content=document.content,
                metadata=document.metadata
            )

            # Split document
            chunks = self.text_splitter.split_documents([langchain_doc])

            # Convert to our Chunk format
            result_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_obj = Chunk(
                    id=f"{document.id}_chunk_{i}",
                    document_id=document.id,
                    content=chunk.page_content,
                    start_pos=0,  # LangChain doesn't track positions
                    end_pos=len(chunk.page_content),
                    metadata={
                        "chunk_id": i,
                        "document_title": document.title,
                        "document_type": document.metadata.get("type", "unknown"),
                        **chunk.metadata
                    }
                )
                result_chunks.append(chunk_obj)

            self.logger.info(f"Created {len(result_chunks)} chunks for document {document.id}")
            return result_chunks

        except Exception as e:
            self.logger.error(f"Error chunking document {document.id}: {e}")
            raise

    def process_and_chunk(self, file_path: Path) -> List[Chunk]:
        """Process document and return chunks in one step"""
        document = self.process_document(file_path)
        return self.chunk_document(document)

    def process_multiple_documents(self, file_paths: List[Path]) -> List[Chunk]:
        """Process multiple documents and return all chunks"""
        all_chunks = []

        for file_path in file_paths:
            try:
                chunks = self.process_and_chunk(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                continue

        self.logger.info(f"Processed {len(file_paths)} documents, created {len(all_chunks)} chunks")
        return all_chunks

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return ['.txt', '.pdf', '.docx']

    def is_supported_file(self, file_path: Path) -> bool:
        """Check if file type is supported"""
        return file_path.suffix.lower() in self.get_supported_extensions()

    def set_text_splitter(self, text_splitter: TextSplitter) -> None:
        """Set custom text splitter"""
        self.text_splitter = text_splitter
        self.logger.info("Updated text splitter")

    def get_text_splitter(self) -> TextSplitter:
        """Get current text splitter"""
        return self.text_splitter
