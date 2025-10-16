"""
RAG Engine - Main orchestrator for Retrieval-Augmented Generation
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..config.settings import Settings
from ..models.schemas import Document, Chunk, Query, Response
from .document_processor import DocumentProcessorFactory
from ..modules.nlp.embedder import Embedder
from ..modules.llm.llm_client import LLMClient
from ..utils.vector_store import VectorStore


class RAGEngine:
    """Main RAG Engine orchestrating all components"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.embedder = Embedder(settings.embedding_model)
        self.llm_client = LLMClient(settings.llm_model, settings.openai_api_key)
        self.vector_store = VectorStore(
            store_type=settings.vector_store_type,
            store_path=settings.vector_store_path
        )
        
        self.logger.info("RAG Engine initialized successfully")
    
    def add_documents(self, file_paths: List[Path]) -> None:
        """Add documents to the knowledge base"""
        self.logger.info(f"Adding {len(file_paths)} documents to knowledge base")
        
        for file_path in file_paths:
            try:
                # Process document
                processor = DocumentProcessorFactory.get_processor(file_path)
                document = processor.process_document(file_path)
                
                # Create chunks
                chunks = processor.chunk_document(document)
                
                # Generate embeddings for chunks
                chunk_texts = [chunk.content for chunk in chunks]
                embeddings = self.embedder.embed_texts(chunk_texts)
                
                # Store in vector database
                self.vector_store.add_chunks(chunks, embeddings)
                
                self.logger.info(f"Successfully added document: {file_path}")
                
            except Exception as e:
                self.logger.error(f"Error adding document {file_path}: {e}")
                continue
    
    def query(self, query_text: str, top_k: int = 5) -> Response:
        """Query the RAG system"""
        try:
            # Create query object
            query = Query(text=query_text)
            
            # Generate query embedding
            query_embedding = self.embedder.embed_texts([query_text])[0]
            
            # Retrieve relevant chunks
            relevant_chunks = self.vector_store.search(query_embedding, top_k=top_k)
            
            if not relevant_chunks:
                return Response(
                    query=query,
                    answer="No relevant documents found.",
                    sources=[],
                    metadata={"chunks_found": 0}
                )
            
            # Prepare context for LLM
            context = self._prepare_context(relevant_chunks)
            
            # Generate response using LLM
            answer = self.llm_client.generate_response(query_text, context)
            
            # Create response
            response = Response(
                query=query,
                answer=answer,
                sources=[chunk.metadata for chunk in relevant_chunks],
                metadata={
                    "chunks_found": len(relevant_chunks),
                    "context_length": len(context)
                }
            )
            
            self.logger.info(f"Generated response for query: {query_text[:50]}...")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return Response(
                query=Query(text=query_text),
                answer=f"Error processing query: {str(e)}",
                sources=[],
                metadata={"error": str(e)}
            )
    
    def _prepare_context(self, chunks: List[Chunk]) -> str:
        """Prepare context from relevant chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"Document {i}:\n{chunk.content}\n")
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        return {
            "total_documents": self.vector_store.get_document_count(),
            "total_chunks": self.vector_store.get_chunk_count(),
            "embedding_model": self.settings.embedding_model,
            "llm_model": self.settings.llm_model,
            "vector_store_type": self.settings.vector_store_type
        }
