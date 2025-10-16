"""
RAG Engine using LangChain with LCEL (LangChain Expression Language)
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever

from ..config.settings import Settings
from ..models.schemas import Document, Chunk, Query, Response
from .document_processor import DocumentProcessor
from ..modules.nlp.embedder import Embedder
from ..modules.llm.llm_client import LLMClient
from ..utils.vector_store import VectorStoreManager


class RAGEngine:
    """Main RAG Engine using LangChain with LCEL"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.embedder = Embedder(
            model_name=settings.embedding_model,
            embedding_type=getattr(settings, 'embedding_type', 'huggingface'),
            api_key=getattr(settings, 'huggingface_api_key', None)
        )

        self.llm_client = LLMClient(
            model=settings.llm_model,
            api_key=settings.deepseek_api_key,
            temperature=getattr(settings, 'temperature', 0.7),
            max_tokens=getattr(settings, 'max_tokens', None)
        )

        self.vector_store_manager = VectorStoreManager(
            store_type=settings.vector_store_type,
            store_path=settings.vector_store_path,
            embeddings=self.embedder.get_langchain_embeddings(),
            collection_name=getattr(settings, 'collection_name', 'rag_documents')
        )

        self.document_processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        # Create RAG chain using LCEL
        self.rag_chain = self._create_rag_chain()

        self.logger.info("RAG Engine initialized successfully with LangChain")

    def _create_rag_chain(self):
        """Create RAG chain using LangChain Expression Language"""
        try:
            # Get LangChain components
            retriever = self.vector_store_manager.get_langchain_vector_store().as_retriever(
                search_kwargs={"k": 5}
            )
            llm = self.llm_client.get_langchain_llm()

            # Create prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that answers questions based on the provided context.
                Use the following pieces of context to answer the question. If you don't know the answer based on the context,
                just say that you don't know, don't try to make up an answer."""),
                ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
            ])

            # Create the RAG chain using LCEL
            rag_chain = (
                {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )

            self.logger.info("Created RAG chain with LCEL")
            return rag_chain

        except Exception as e:
            self.logger.error(f"Error creating RAG chain: {e}")
            raise

    def _format_docs(self, docs):
        """Format retrieved documents for the prompt"""
        return "\n\n".join(doc.page_content for doc in docs)

    def add_documents(self, file_paths: List[Path]) -> None:
        """Add documents to the knowledge base"""
        self.logger.info(f"Adding {len(file_paths)} documents to knowledge base")

        try:
            # Process all documents
            all_chunks = self.document_processor.process_multiple_documents(file_paths)

            if not all_chunks:
                self.logger.warning("No chunks created from documents")
                return

            # Add chunks to vector store
            self.vector_store_manager.add_chunks(all_chunks)

            # Save vector store
            self.vector_store_manager.save()

            self.logger.info(f"Successfully added {len(file_paths)} documents, created {len(all_chunks)} chunks")

        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            raise

    def query(self, query_text: str, top_k: int = 5) -> Response:
        """Query the RAG system using LangChain chain"""
        try:
            # Create query object
            query = Query(text=query_text)

            # Update retriever k parameter if different
            if top_k != 5:
                retriever = self.vector_store_manager.get_langchain_vector_store().as_retriever(
                    search_kwargs={"k": top_k}
                )
                # Recreate chain with new retriever
                self.rag_chain = self._create_rag_chain()

            # Get relevant chunks for metadata
            relevant_chunks = self.vector_store_manager.search(query_text, top_k=top_k)

            # Generate response using RAG chain
            answer = self.rag_chain.invoke(query_text)

            # Create response
            response = Response(
                query=query,
                answer=answer,
                sources=[chunk.metadata for chunk in relevant_chunks],
                metadata={
                    "chunks_found": len(relevant_chunks),
                    "query_length": len(query_text),
                    "answer_length": len(answer)
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

    def stream_query(self, query_text: str, top_k: int = 5):
        """Stream response for real-time output"""
        try:
            # Update retriever if needed
            if top_k != 5:
                retriever = self.vector_store_manager.get_langchain_vector_store().as_retriever(
                    search_kwargs={"k": top_k}
                )
                self.rag_chain = self._create_rag_chain()

            # Stream response
            for chunk in self.rag_chain.stream(query_text):
                yield chunk

        except Exception as e:
            self.logger.error(f"Error streaming query: {e}")
            yield f"Error streaming query: {str(e)}"

    def batch_query(self, queries: List[str], top_k: int = 5) -> List[Response]:
        """Process multiple queries in batch"""
        try:
            responses = []

            for query_text in queries:
                response = self.query(query_text, top_k)
                responses.append(response)

            self.logger.info(f"Processed {len(queries)} queries in batch")
            return responses

        except Exception as e:
            self.logger.error(f"Error in batch query: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        return {
            "total_documents": self.vector_store_manager.get_document_count(),
            "total_chunks": self.vector_store_manager.get_chunk_count(),
            "embedding_model": self.settings.embedding_model,
            "llm_model": self.settings.llm_model,
            "vector_store_type": self.settings.vector_store_type,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap
        }

    def clear_knowledge_base(self) -> None:
        """Clear all data from the knowledge base"""
        try:
            self.vector_store_manager.clear()
            self.logger.info("Knowledge base cleared")
        except Exception as e:
            self.logger.error(f"Error clearing knowledge base: {e}")
            raise

    def get_retriever(self) -> VectorStoreRetriever:
        """Get the retriever for advanced usage"""
        return self.vector_store_manager.get_langchain_vector_store().as_retriever()

    def update_chain_parameters(self, **kwargs) -> None:
        """Update RAG chain parameters"""
        try:
            # Update retriever parameters
            if 'search_kwargs' in kwargs:
                retriever = self.vector_store_manager.get_langchain_vector_store().as_retriever(
                    search_kwargs=kwargs['search_kwargs']
                )
                self.rag_chain = self._create_rag_chain()

            # Update LLM parameters
            if 'temperature' in kwargs or 'max_tokens' in kwargs:
                self.llm_client = LLMClient(
                    model=self.settings.llm_model,
                    api_key=self.settings.openai_api_key,
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=kwargs.get('max_tokens', None)
                )
                self.rag_chain = self._create_rag_chain()

            self.logger.info("Updated RAG chain parameters")

        except Exception as e:
            self.logger.error(f"Error updating chain parameters: {e}")
            raise
