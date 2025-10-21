"""
RAG Engine using LangChain with LCEL (LangChain Expression Language)
"""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import Settings
from models.schemas import Document, Chunk, Query, Response
from .document_processor import DocumentProcessor
from modules.nlp.embedder import Embedder
from modules.llm.llm_client import LLMClient
from utils.vector_store import VectorStoreManager


class RAGEngine:
    """Main RAG Engine using LangChain with LCEL"""

    def __init__(self, settings: Settings):
        init_start = time.time()
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        # Initialize components
        embedder_start = time.time()
        self.logger.info(f"🔄 Инициализация эмбеддера {settings.embedding_model}...")
        self.embedder = Embedder(
            model_name=settings.embedding_model,
            embedding_type=getattr(settings, 'embedding_type', 'huggingface'),
            api_key=getattr(settings, 'huggingface_api_key', None)
        )
        embedder_time = time.time() - embedder_start
        self.logger.info(f"✅ Эмбеддер инициализирован за {embedder_time:.2f}с")

        llm_start = time.time()
        self.logger.info(f"🔄 Инициализация LLM {settings.llm_model}...")
        self.llm_client = LLMClient(
            model=settings.llm_model,
            api_key=settings.deepseek_api_key,
            temperature=getattr(settings, 'temperature', 0.7),
            max_tokens=getattr(settings, 'max_tokens', None)
        )
        llm_time = time.time() - llm_start
        self.logger.info(f"✅ LLM инициализирован за {llm_time:.2f}с")

        vector_start = time.time()
        self.logger.info(f"🔄 Инициализация векторного хранилища {settings.vector_store_type}...")
        self.vector_store_manager = VectorStoreManager(
            store_type=settings.vector_store_type,
            store_path=settings.vector_store_path,
            embeddings=self.embedder.get_langchain_embeddings(),
            collection_name=getattr(settings, 'collection_name', 'rag_documents')
        )
        vector_time = time.time() - vector_start
        self.logger.info(f"✅ Векторное хранилище инициализировано за {vector_time:.2f}с")

        processor_start = time.time()
        self.logger.info("🔄 Инициализация процессора документов...")
        self.document_processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            use_smart_chunker=getattr(settings, 'use_smart_chunker', False),
            smart_chunker_config=getattr(settings, 'smart_chunker_config_path', None)
        )
        processor_time = time.time() - processor_start
        self.logger.info(f"✅ Процессор документов инициализирован за {processor_time:.2f}с")

        # Create RAG chain using LCEL
        chain_start = time.time()
        self.logger.info("🔄 Создание RAG цепочки...")
        self.rag_chain = self._create_rag_chain()
        chain_time = time.time() - chain_start
        self.logger.info(f"✅ RAG цепочка создана за {chain_time:.2f}с")

        total_init_time = time.time() - init_start
        self.logger.info(f"🚀 RAG Engine полностью инициализирован за {total_init_time:.2f}с")
        self.logger.info(f"⏱️  Детализация: Эмбеддер={embedder_time:.2f}с | LLM={llm_time:.2f}с | Векторное хранилище={vector_time:.2f}с | Процессор={processor_time:.2f}с | RAG цепочка={chain_time:.2f}с")

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

    def add_documents(self, file_paths: List[Path]) -> int:
        """Add documents to the knowledge base. Returns number of chunks created."""
        start_time = time.time()
        self.logger.info(f"Adding {len(file_paths)} documents to knowledge base")

        try:
            # Process all documents
            process_start = time.time()
            all_chunks = self.document_processor.process_multiple_documents(file_paths)
            process_time = time.time() - process_start

            if not all_chunks:
                self.logger.warning("No chunks created from documents")
                raise ValueError("Не удалось создать фрагменты из документов")

            # Add chunks to vector store
            vector_start = time.time()
            self.vector_store_manager.add_chunks(all_chunks)
            vector_time = time.time() - vector_start

            # Save vector store
            save_start = time.time()
            self.vector_store_manager.save()
            save_time = time.time() - save_start

            total_time = time.time() - start_time

            self.logger.info(f"Successfully added {len(file_paths)} documents, created {len(all_chunks)} chunks")
            self.logger.info(f"⏱️  Время обработки: {process_time:.2f}с | Векторизация: {vector_time:.2f}с | Сохранение: {save_time:.2f}с | Общее: {total_time:.2f}с")
            return len(all_chunks)

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Error adding documents after {total_time:.2f}с: {e}")
            raise

    def add_documents_with_smart_chunker(self, file_paths: List[Path]) -> int:
        """Add documents using SmartChunker for hierarchical processing. Returns number of chunks created."""
        if not self.document_processor.is_smart_chunker_available():
            raise ValueError("SmartChunker не доступен. Установите зависимости и включите в настройках.")

        start_time = time.time()
        self.logger.info(f"Adding {len(file_paths)} documents with SmartChunker to knowledge base")

        try:
            # Process documents with SmartChunker
            process_start = time.time()
            documents = self.document_processor.process_multiple_documents_with_smart_chunker(file_paths)
            process_time = time.time() - process_start

            # Extract all chunks from documents
            extract_start = time.time()
            all_chunks = []
            for doc in documents:
                all_chunks.extend(doc.chunks)
            extract_time = time.time() - extract_start

            if not all_chunks:
                self.logger.warning("No chunks created from documents with SmartChunker")
                raise ValueError("Не удалось создать фрагменты из документов с SmartChunker")

            # Add chunks to vector store
            vector_start = time.time()
            self.vector_store_manager.add_chunks(all_chunks)
            vector_time = time.time() - vector_start

            # Save vector store
            save_start = time.time()
            self.vector_store_manager.save()
            save_time = time.time() - save_start

            total_time = time.time() - start_time

            self.logger.info(f"Successfully added {len(all_chunks)} hierarchical chunks to knowledge base")
            self.logger.info(f"⏱️  SmartChunker обработка: {process_time:.2f}с | Извлечение чанков: {extract_time:.2f}с | Векторизация: {vector_time:.2f}с | Сохранение: {save_time:.2f}с | Общее: {total_time:.2f}с")
            return len(all_chunks)

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Error adding documents with SmartChunker after {total_time:.2f}с: {e}")
            raise

    def query(self, query_text: str, top_k: int = 5) -> Response:
        """Query the RAG system using LangChain chain"""
        start_time = time.time()

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
            search_start = time.time()
            relevant_chunks = self.vector_store_manager.search(query_text, top_k=top_k)
            search_time = time.time() - search_start

            # Generate response using RAG chain
            llm_start = time.time()
            answer = self.rag_chain.invoke(query_text)
            llm_time = time.time() - llm_start

            total_time = time.time() - start_time

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
            self.logger.info(f"⏱️  Поиск: {search_time:.2f}с | LLM: {llm_time:.2f}с | Общее: {total_time:.2f}с")
            return response

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Error processing query after {total_time:.2f}с: {e}")
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
                    api_key=self.settings.deepseek_api_key,
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=kwargs.get('max_tokens', None)
                )
                self.rag_chain = self._create_rag_chain()

            self.logger.info("Updated RAG chain parameters")

        except Exception as e:
            self.logger.error(f"Error updating chain parameters: {e}")
            raise
