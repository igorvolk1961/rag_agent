"""
Configuration settings for RAG Agent with LangChain
"""

import os
from pathlib import Path
from typing import Optional, Literal
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings for LangChain-based RAG Agent"""

    # Application settings
    app_name: str = "RAG Agent (LangChain)"
    app_version: str = "0.2.0"
    debug: bool = False

    # Data paths
    data_dir: Path = Path("data")
    input_dir: Path = Path("data/input")
    output_dir: Path = Path("data/output")

    # LangChain Model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_type: Literal["huggingface", "openai"] = "huggingface"
    llm_model: str = "deepseek-chat"

    # Text processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    text_splitter_type: str = "recursive"  # recursive, character, token

    # Vector store settings
    vector_store_type: Literal["chroma", "faiss"] = "chroma"
    vector_store_path: Path = Path("data/vector_store")
    collection_name: str = "rag_documents"

    # Retrieval settings
    retrieval_k: int = 5
    similarity_threshold: float = 0.0

    # LLM settings
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # API settings
    openai_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    deepseek_api_key: Optional[str] = "sk-163567b8e043481bb97dd4cbbc3a965b"
    deepseek_api_base: Optional[str] = None  # Use official endpoint

    # LangChain specific settings
    use_streaming: bool = False
    enable_memory: bool = False
    memory_type: str = "conversation_buffer"  # conversation_buffer, conversation_summary

    # Advanced settings
    enable_callbacks: bool = False
    callback_handlers: list = []
    custom_prompts: dict = {}

    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    enable_langchain_logging: bool = False

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

        # Set up environment variables for LangChain
        self._setup_langchain_env()

    def _setup_langchain_env(self):
        """Set up environment variables for LangChain"""
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key

        if self.huggingface_api_key:
            os.environ["HUGGINGFACE_API_KEY"] = self.huggingface_api_key

        if self.openai_api_base:
            os.environ["OPENAI_API_BASE"] = self.openai_api_base

        if self.deepseek_api_key:
            os.environ["DEEPSEEK_API_KEY"] = self.deepseek_api_key

        if self.deepseek_api_base:
            os.environ["DEEPSEEK_API_BASE"] = self.deepseek_api_base

        # LangChain specific environment variables
        if self.enable_langchain_logging:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_VERBOSE"] = "true"

    def get_embedding_config(self) -> dict:
        """Get embedding configuration"""
        return {
            "model_name": self.embedding_model,
            "embedding_type": self.embedding_type,
            "api_key": self.huggingface_api_key if self.embedding_type == "huggingface" else self.openai_api_key
        }

    def get_llm_config(self) -> dict:
        """Get LLM configuration"""
        return {
            "model": self.llm_model,
            "api_key": self.deepseek_api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }

    def get_vector_store_config(self) -> dict:
        """Get vector store configuration"""
        return {
            "store_type": self.vector_store_type,
            "store_path": self.vector_store_path,
            "collection_name": self.collection_name
        }

    def get_retrieval_config(self) -> dict:
        """Get retrieval configuration"""
        return {
            "k": self.retrieval_k,
            "similarity_threshold": self.similarity_threshold
        }

    def get_text_splitter_config(self) -> dict:
        """Get text splitter configuration"""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "splitter_type": self.text_splitter_type
        }

    def validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []

        # Check required API keys
        if self.embedding_type == "openai" and not self.openai_api_key:
            errors.append("OpenAI API key is required for OpenAI embeddings")

        if self.llm_model.startswith("gpt") and not self.openai_api_key:
            errors.append("OpenAI API key is required for GPT models")

        if self.llm_model.startswith("deepseek") and not self.deepseek_api_key:
            errors.append("DeepSeek API key is required for DeepSeek models")

        # Check vector store path
        if not self.vector_store_path.parent.exists():
            errors.append(f"Vector store parent directory does not exist: {self.vector_store_path.parent}")

        # Check chunk settings
        if self.chunk_size <= 0:
            errors.append("Chunk size must be positive")

        if self.chunk_overlap >= self.chunk_size:
            errors.append("Chunk overlap must be less than chunk size")

        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")

        return True
