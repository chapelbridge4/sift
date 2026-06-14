"""Inference backend selection. GGUF (cross-platform) is default; MLX is the Apple fast-path."""
from __future__ import annotations
import os
from typing import Protocol, List, runtime_checkable

from app.config import get_settings


@runtime_checkable
class InferenceBackend(Protocol):
    async def generate_rag_response(
        self,
        query: str,
        retrieved_contexts: List[str],
        model_profile: str = "fast",
        max_tokens: int = 200,
    ) -> str: ...


def get_inference_backend() -> InferenceBackend:
    """
    Instantiate and return the configured inference backend.

    Backend is selected by the INFERENCE_BACKEND env var (checked first, so
    monkeypatching works even when settings is cached by lru_cache), falling
    back to settings.INFERENCE_BACKEND, then to 'gguf'.

    Imports are deferred into each branch so heavy ML libraries (mlx, llama_cpp)
    are never imported at module load time — backends construct without loading a model.
    """
    settings = get_settings()
    backend = (
        os.getenv("INFERENCE_BACKEND")
        or getattr(settings, "INFERENCE_BACKEND", "gguf")
        or "gguf"
    ).lower()

    if backend == "gguf":
        from app.services.gguf_service import GGUFService
        return GGUFService()

    if backend == "mlx":
        from app.services.llm_service import LLMService
        return LLMService()

    raise ValueError(
        f"Unknown INFERENCE_BACKEND: {backend!r} (expected 'gguf' or 'mlx')"
    )
