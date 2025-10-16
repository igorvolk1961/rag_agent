"""
Configuration settings for RAG Agent
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Application settings
    app_name: str = "RAG Agent"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # Data paths
    data_dir: Path = Path("data")
    input_dir: Path = Path("data/input")
    output_dir: Path = Path("data/output")
    
    # Model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-3.5-turbo"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Vector store settings
    vector_store_type: str = "chroma"  # chroma, faiss, pinecone
    vector_store_path: Path = Path("data/vector_store")
    
    # API settings
    openai_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.vector_store_path.mkdir(exist_ok=True)
        
        # Set log file if not specified
        if self.log_file is None:
            self.log_file = self.data_dir / "rag_agent.log"
