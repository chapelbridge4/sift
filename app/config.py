"""
Configuration module for Brain-inspired RAG system.
Handles environment variables and application settings.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Brain-Inspired RAG"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_API_KEY: str | None = None
    QDRANT_TIMEOUT: int = 60

    # Collection Configuration
    DEFAULT_COLLECTION_NAME: str = "brain_rag_collection"
    DENSE_VECTOR_SIZE: int = 384  # all-MiniLM-L6-v2 dimension

    # Ollama Configuration
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    OLLAMA_TIMEOUT: int = 120

    # Embedding Models
    DENSE_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    SPARSE_MODEL_NAME: str = "Qdrant/bm25"

    # Document Processing
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 128
    MAX_FILE_SIZE_MB: int = 50
    BATCH_SIZE: int = 32
    MAX_WORKERS: int = 4

    # Hybrid Search Configuration
    HYBRID_FUSION_METHOD: str = "rrf"  # "rrf" or "dbsf"
    TOP_K_RESULTS: int = 10
    DENSE_WEIGHT: float = 0.5
    SPARSE_WEIGHT: float = 0.5

    # Brain Module Weights (Amygdala importance scoring)
    RECENCY_WEIGHT: float = 0.3
    RELEVANCE_WEIGHT: float = 0.5
    IMPORTANCE_WEIGHT: float = 0.2

    # Working Memory
    CONTEXT_WINDOW_SIZE: int = 5
    MAX_CONVERSATION_HISTORY: int = 10

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
