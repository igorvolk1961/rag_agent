"""
Vector store implementation using LangChain
"""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings

from models.schemas import Chunk


class VectorStoreManager:
    """Vector store manager using LangChain vector stores"""

    def __init__(self, store_type: str = "chroma", store_path: Optional[Path] = None,
                 embeddings: Optional[Embeddings] = None, collection_name: str = "rag_documents"):
        self.store_type = store_type
        self.store_path = store_path or Path("data/vector_store")
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)

        # Initialize store
        self.vector_store = self._init_vector_store()

    def _init_vector_store(self) -> VectorStore:
        """Initialize LangChain vector store"""
        try:
            if self.store_type == "chroma":
                return self._init_chroma()
            elif self.store_type == "faiss":
                return self._init_faiss()
            else:
                raise ValueError(f"Unsupported vector store type: {self.store_type}")
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {e}")
            raise

    def _init_chroma(self) -> Chroma:
        """Initialize ChromaDB vector store"""
        try:
            self.store_path.mkdir(parents=True, exist_ok=True)

            vector_store = Chroma(
                persist_directory=str(self.store_path),
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )

            self.logger.info("Initialized ChromaDB vector store with LangChain")
            return vector_store

        except ImportError:
            raise ImportError("chromadb is required. Install with: pip install chromadb")
        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB: {e}")
            raise

    def _init_faiss(self) -> FAISS:
        """Initialize FAISS vector store"""
        try:
            self.store_path.mkdir(parents=True, exist_ok=True)

            # Create empty FAISS store
            vector_store = FAISS.from_texts(
                texts=["dummy"],  # FAISS requires at least one document
                embedding=self.embeddings
            )

            # Remove dummy document
            vector_store.delete([vector_store.index_to_docstore_id[0]])

            self.logger.info("Initialized FAISS vector store with LangChain")
            return vector_store

        except ImportError:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
        except Exception as e:
            self.logger.error(f"Error initializing FAISS: {e}")
            raise

    def add_chunks(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None) -> None:
        """Add chunks to the vector store"""
        start_time = time.time()

        try:
            # Convert chunks to LangChain Documents
            convert_start = time.time()
            self.logger.info(f"🔄 Конвертация {len(chunks)} чанков в LangChain Documents...")
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk.content,
                    metadata={
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "start_pos": chunk.start_pos,
                        "end_pos": chunk.end_pos,
                        **chunk.metadata
                    }
                )
                documents.append(doc)
            convert_time = time.time() - convert_start
            self.logger.info(f"✅ Конвертация завершена за {convert_time:.2f}с")

            # Add documents to vector store
            vector_start = time.time()
            self.logger.info(f"🔄 Добавление {len(documents)} документов в {self.store_type} векторное хранилище...")
            if self.store_type == "chroma":
                self.vector_store.add_documents(documents)
            elif self.store_type == "faiss":
                self.vector_store.add_documents(documents)
            vector_time = time.time() - vector_start
            self.logger.info(f"✅ Документы добавлены в векторное хранилище за {vector_time:.2f}с")

            total_time = time.time() - start_time
            self.logger.info(f"✅ Добавлено {len(chunks)} чанков в {self.store_type} векторное хранилище за {total_time:.2f}с")

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Error adding chunks to vector store after {total_time:.2f}с: {e}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Search for similar chunks"""
        try:
            # Search using LangChain vector store
            docs = self.vector_store.similarity_search(query, k=top_k)

            # Convert back to Chunk objects
            chunks = []
            for doc in docs:
                chunk = Chunk(
                    id=doc.metadata.get("id", ""),
                    document_id=doc.metadata.get("document_id", ""),
                    content=doc.page_content,
                    start_pos=doc.metadata.get("start_pos", 0),
                    end_pos=doc.metadata.get("end_pos", len(doc.page_content)),
                    metadata=doc.metadata
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            self.logger.error(f"Error searching vector store: {e}")
            return []

    def search_with_scores(self, query: str, top_k: int = 5) -> List[tuple]:
        """Search for similar chunks with similarity scores"""
        try:
            # Search with scores
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k)

            # Convert to (chunk, score) tuples
            results = []
            for doc, score in docs_with_scores:
                chunk = Chunk(
                    id=doc.metadata.get("id", ""),
                    document_id=doc.metadata.get("document_id", ""),
                    content=doc.page_content,
                    start_pos=doc.metadata.get("start_pos", 0),
                    end_pos=doc.metadata.get("end_pos", len(doc.page_content)),
                    metadata=doc.metadata
                )
                results.append((chunk, score))

            return results

        except Exception as e:
            self.logger.error(f"Error searching vector store with scores: {e}")
            return []

    def get_document_count(self) -> int:
        """Get total number of documents"""
        try:
            if self.store_type == "chroma":
                return self.vector_store._collection.count()
            elif self.store_type == "faiss":
                return len(self.vector_store.docstore._dict)
            return 0
        except Exception as e:
            self.logger.error(f"Error getting document count: {e}")
            return 0

    def get_chunk_count(self) -> int:
        """Get total number of chunks"""
        return self.get_document_count()

    def clear(self) -> None:
        """Clear all data from the vector store"""
        try:
            if self.store_type == "chroma":
                # Delete and recreate collection
                self.vector_store._client.delete_collection(self.collection_name)
                self.vector_store = self._init_chroma()
            elif self.store_type == "faiss":
                # Create new empty FAISS store
                self.vector_store = self._init_faiss()

            self.logger.info("Cleared vector store")

        except Exception as e:
            self.logger.error(f"Error clearing vector store: {e}")
            raise

    def save(self) -> None:
        """Save vector store to disk"""
        start_time = time.time()

        try:
            if self.store_type == "chroma":
                # Chroma auto-saves
                self.logger.info("🔄 ChromaDB автоматически сохраняет данные...")
                pass
            elif self.store_type == "faiss":
                self.logger.info(f"🔄 Сохранение FAISS векторного хранилища в {self.store_path}...")
                self.vector_store.save_local(str(self.store_path))

            save_time = time.time() - start_time
            self.logger.info(f"✅ Векторное хранилище сохранено за {save_time:.2f}с")

        except Exception as e:
            self.logger.error(f"Error saving vector store: {e}")
            raise

    def load(self) -> None:
        """Load vector store from disk"""
        try:
            if self.store_type == "faiss":
                self.vector_store = FAISS.load_local(
                    str(self.store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.logger.info("Loaded FAISS vector store from disk")

        except Exception as e:
            self.logger.error(f"Error loading vector store: {e}")
            raise

    def get_langchain_vector_store(self) -> VectorStore:
        """Get the underlying LangChain vector store"""
        return self.vector_store
