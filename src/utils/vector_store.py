"""
Vector store implementation
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from ..models.schemas import Chunk


class VectorStore:
    """Vector store for storing and retrieving document embeddings"""
    
    def __init__(self, store_type: str = "chroma", store_path: Optional[Path] = None):
        self.store_type = store_type
        self.store_path = store_path or Path("data/vector_store")
        self.logger = logging.getLogger(__name__)
        
        # Initialize store
        if store_type == "chroma":
            self._init_chroma()
        elif store_type == "faiss":
            self._init_faiss()
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
    
    def _init_chroma(self):
        """Initialize ChromaDB vector store"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.store_path.mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=str(self.store_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.client.get_or_create_collection(
                name="rag_documents",
                metadata={"description": "RAG Agent document collection"}
            )
            
            self.logger.info("Initialized ChromaDB vector store")
            
        except ImportError:
            raise ImportError("chromadb is required. Install with: pip install chromadb")
        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def _init_faiss(self):
        """Initialize FAISS vector store"""
        try:
            import faiss
            
            self.store_path.mkdir(parents=True, exist_ok=True)
            self.index = None
            self.chunks = []
            self.embeddings = []
            
            self.logger.info("Initialized FAISS vector store")
            
        except ImportError:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
        except Exception as e:
            self.logger.error(f"Error initializing FAISS: {e}")
            raise
    
    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Add chunks with embeddings to the vector store"""
        if self.store_type == "chroma":
            self._add_chunks_chroma(chunks, embeddings)
        elif self.store_type == "faiss":
            self._add_chunks_faiss(chunks, embeddings)
    
    def _add_chunks_chroma(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Add chunks to ChromaDB"""
        try:
            ids = [chunk.id for chunk in chunks]
            texts = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            self.logger.info(f"Added {len(chunks)} chunks to ChromaDB")
            
        except Exception as e:
            self.logger.error(f"Error adding chunks to ChromaDB: {e}")
            raise
    
    def _add_chunks_faiss(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Add chunks to FAISS"""
        try:
            import faiss
            
            embeddings_array = np.array(embeddings).astype('float32')
            
            if self.index is None:
                dimension = embeddings_array.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            self.index.add(embeddings_array)
            self.chunks.extend(chunks)
            self.embeddings.extend(embeddings)
            
            # Save index
            faiss.write_index(self.index, str(self.store_path / "faiss_index.bin"))
            
            self.logger.info(f"Added {len(chunks)} chunks to FAISS")
            
        except Exception as e:
            self.logger.error(f"Error adding chunks to FAISS: {e}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Chunk]:
        """Search for similar chunks"""
        if self.store_type == "chroma":
            return self._search_chroma(query_embedding, top_k)
        elif self.store_type == "faiss":
            return self._search_faiss(query_embedding, top_k)
    
    def _search_chroma(self, query_embedding: List[float], top_k: int) -> List[Chunk]:
        """Search in ChromaDB"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            chunks = []
            for i in range(len(results['ids'][0])):
                chunk_id = results['ids'][0][i]
                content = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                
                chunk = Chunk(
                    id=chunk_id,
                    document_id=metadata.get('document_id', ''),
                    content=content,
                    start_pos=0,
                    end_pos=len(content),
                    metadata=metadata
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def _search_faiss(self, query_embedding: List[float], top_k: int) -> List[Chunk]:
        """Search in FAISS"""
        try:
            import faiss
            
            if self.index is None:
                return []
            
            query_array = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_array)
            
            scores, indices = self.index.search(query_array, top_k)
            
            chunks = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    chunk.metadata['similarity_score'] = float(score)
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error searching FAISS: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Get total number of documents"""
        if self.store_type == "chroma":
            return self.collection.count()
        elif self.store_type == "faiss":
            return len(set(chunk.document_id for chunk in self.chunks))
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks"""
        if self.store_type == "chroma":
            return self.collection.count()
        elif self.store_type == "faiss":
            return len(self.chunks)
    
    def clear(self) -> None:
        """Clear all data from the vector store"""
        if self.store_type == "chroma":
            self.client.delete_collection("rag_documents")
            self.collection = self.client.create_collection("rag_documents")
        elif self.store_type == "faiss":
            self.index = None
            self.chunks = []
            self.embeddings = []
        
        self.logger.info("Cleared vector store")
