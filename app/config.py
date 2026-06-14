"""
Configuration module for Brain-inspired RAG system.
Handles environment variables and application settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import Optional, List
import threading

from app.config_manager import ConfigManager


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Brain-Inspired RAG"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    CORS_ALLOW_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_API_KEY: str | None = None
    QDRANT_TIMEOUT: int = 60
    QDRANT_HARDWARE_PROFILE: str = "auto"  # 'auto', 'low', 'medium', 'high'

    # Collection Configuration
    DEFAULT_COLLECTION_NAME: str = "brain_rag_collection"
    DENSE_VECTOR_SIZE: int = 384

    # Inference backend — 'gguf' uses llama-cpp-python (Metal), 'mlx' uses MLX-VLM
    # 'gguf' is the cross-platform default; 'mlx' requires Apple Silicon + MLX stack.
    INFERENCE_BACKEND: str = Field(
        default="gguf",
        description="Inference backend: 'gguf' (llama-cpp-python) or 'mlx' (MLX-VLM)"
    )

    # GGUF backend configuration
    GGUF_MODEL_PATH: Optional[str] = Field(
        default=None,
        description="Explicit local path to GGUF model file (overrides default ~/.cache/gguf)"
    )
    GGUF_N_GPU_LAYERS: int = Field(
        default=-1,
        description="Number of layers to offload to Metal (-1 = all)"
    )
    GGUF_N_CTX: int = Field(
        default=4096,
        description="Context window size for GGUF model"
    )

    # MLX Configuration (Apple Silicon native LLM)
    MLX_MODEL_CACHE_DIR: str = Field(
        default="~/.cache/mlx-lm",
        description="Directory for cached MLX models"
    )
    MLX_WARMUP_ON_LOAD: bool = True  # Run 1-token warmup on engine init
    MLX_METAL_CLEAR_CACHE_BETWEEN_PROFILES: bool = True  # mx.metal.clear_cache() on profile switch

    # Model Selection Configuration
    MODEL_PROFILE: str = "fast"
    CUSTOM_MODEL_NAME: str | None = None

    # Model Profiles — MLX-VLM models for Apple Silicon (8GB M1 safe, ≤4GB per profile)
    # NOTE: 2B-MLX-4bit is broken (attention corruption at 4-bit). All profiles use 4B.
    MODEL_PROFILES: dict = {
        "fast": {
            "model": "mlx-community/Qwen3.5-4B-MLX-4bit",
            "backend": "mlx-vlm",
            "context": 256000,
            "max_tokens": 400,
            "temperature": 0.7,
            "thinking": False,
            "thinking_budget": 300,
            "repetition_penalty": 1.15,
            "description": "~2.6GB loaded, ~50-70 t/s, text + vision, default profile",
        },
        "balanced": {
            "model": "mlx-community/Qwen3.5-4B-MLX-4bit",
            "backend": "mlx-vlm",
            "context": 256000,
            "max_tokens": 600,
            "temperature": 0.7,
            "thinking": False,
            "thinking_budget": 300,
            "repetition_penalty": 1.15,
            "description": "~2.6GB loaded, ~50-70 t/s, text + vision, best tool calling",
        },
        "quality": {
            "model": "mlx-community/Qwen3.5-4B-MLX-4bit",
            "backend": "mlx-vlm",
            "context": 256000,
            "max_tokens": 800,
            "temperature": 0.7,
            "thinking": True,
            "thinking_budget": 300,
            "repetition_penalty": 1.15,
            "description": "same model as balanced but with --enable-thinking for deep multi-hop RAG reasoning",
        },
    }

    # Generation guardrails for MLX-VLM
    # repetition_penalty: 1.0 = off, 1.1 = mild, 1.2 = strong, >1.3 degrades small model quality
    REPETITION_PENALTY: float = 1.15
    # Used when backend supports stopping_criteria (ollama, vLLM, future mlx-vlm)
    STOP_SEQUENCES: List[str] = ["<|endoftext|>"]

    # Retrieval quality threshold (0.0-1.0) — logs warning when top score is below this
    RETRIEVAL_SCORE_THRESHOLD: float = 0.5

    # Tuning suite — MLX optimization experiments
    # All values read from config, no hardcoding in tuning code
    KV_CACHE_QUANTIZATION: Optional[str] = None  # None, "q8", or "q4"
    SPECULATIVE_DECODING_ENABLED: bool = False
    SPECULATIVE_DRAFT_MODEL: str = "z-lab/Qwen3.5-4B-DFlash"
    SPECULATIVE_MIN_ACCEPTANCE_RATE: float = 0.50
    PREFIX_CACHE_ENABLED: bool = False
    PREFIX_CACHE_MAX_ENTRIES: int = 3
    BENCHMARK_OUTPUT_DIR: str = "app/tuning/results"
    DEFAULT_BENCHMARK_QUERIES: List[str] = [
        "What are the key innovations in transformer architectures?",
        "What challenges exist in training large language models?",
        "How do language models achieve reasoning capabilities?",
        "What techniques are used for model compression and efficiency?",
        "How are language models evaluated on benchmarks?",
        "What is the role of attention mechanisms in neural networks?",
        "What are scaling laws in neural network training?",
        "What is prompt engineering and what techniques exist?",
        "What are the different fine-tuning methods for language models?",
        "How do multimodal models combine different data types?",
    ]

    # Tuning configuration — centralized for benchmark and optimization experiments
    TUNING: dict = {
        "kv_cache_dir": ".cache/kv",
        "kv_max_sessions": 3,
        "kv_max_file_mb": 50,
        "benchmark_output_dir": "app/tuning/results",
    }

    # Engine / KV cache settings
    KV_CACHE_DIR: str = ".cache/kv"
    KV_MAX_SESSIONS: int = 3
    KV_MAX_MB_PER_FILE: int = 50
    ENGINE_WARMUP: bool = True
    TEMPERATURE: float = 0.1

    # Embedding Models
    DENSE_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    SPARSE_MODEL_NAME: str = "Qdrant/bm25"
    SPARSE_STRATEGY: str = Field(
        default="bm25",
        description="Sparse embedding strategy: 'bm25' for standard BM25, 'bm25plus' for stronger BM25 variant"
    )

    # Document Processing
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 128
    MAX_FILE_SIZE_MB: int = 50
    BATCH_SIZE: int = 32
    MAX_WORKERS: int = 4

    # Retrieval Configuration
    RETRIEVAL_STRATEGY: str = Field(
        default="dense",
        description="Retrieval strategy: 'dense', 'sparse', or 'hybrid'"
    )
    FUSION_METHOD: str = Field(
        default="rrf",
        description="Fusion method for hybrid retrieval: 'rrf' (Reciprocal Rank Fusion), 'dbsf' (Distribution-Based Score Fusion), or 'linear' (weighted average)"
    )

    # Hybrid Search Configuration
    HYBRID_FUSION_METHOD: str = "rrf"
    TOP_K_RESULTS: int = 10
    DENSE_WEIGHT: float = 0.5
    SPARSE_WEIGHT: float = 0.5

    # Brain Module Weights (Amygdala importance scoring)
    RECENCY_WEIGHT: float = Field(default=0.3, ge=0.0, le=1.0)
    RELEVANCE_WEIGHT: float = Field(default=0.5, ge=0.0, le=1.0)
    IMPORTANCE_WEIGHT: float = Field(default=0.2, ge=0.0, le=1.0)

    # Reranking / Salience Layer
    RERANK_ENABLED: bool = False
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K: int = 5
    RERANK_MIN_AVAILABLE_GB: float = 1.5
    DIVERSIFY_SOURCES: bool = True

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

    def cors_allow_origins(self) -> List[str]:
        """Return configured CORS origins as a clean list."""
        return [
            origin.strip()
            for origin in self.CORS_ALLOW_ORIGINS.split(",")
            if origin.strip()
        ]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


_config_manager: Optional[ConfigManager] = None
_config_lock = threading.Lock()

def get_config_manager() -> ConfigManager:
    """Get or create ConfigManager singleton (thread-safe)."""
    global _config_manager
    if _config_manager is None:
        with _config_lock:
            if _config_manager is None:  # Double-check
                _config_manager = ConfigManager()
    return _config_manager

def get_qdrant_settings() -> dict:
    """Get Qdrant settings from ConfigManager (DB-backed)."""
    return get_config_manager().get_qdrant_config()
