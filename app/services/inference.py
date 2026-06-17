"""Inference backend selection. MLX (Apple Silicon) is the default RAG backend; GGUF is available for direct callers."""
from __future__ import annotations

import os
from typing import List, Protocol, runtime_checkable

from app.config import get_settings


@runtime_checkable
class InferenceBackend(Protocol):
    # Keyword-only optional params so both real backends (GGUFService, LLMService)
    # — which order their positional params differently (temperature/conversation_history
    # before model_profile) — structurally satisfy this Protocol. Callers MUST pass
    # model_profile/max_tokens/temperature by keyword.
    async def generate_rag_response(
        self,
        query: str,
        retrieved_contexts: List[str],
        *,
        model_profile: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str: ...


def get_inference_backend() -> InferenceBackend:
    """
    Instantiate and return the configured inference backend.

    Backend is selected by the INFERENCE_BACKEND env var (checked first, so
    monkeypatching works even when settings is cached by lru_cache), falling
    back to settings.INFERENCE_BACKEND, then to 'mlx'.

    Imports are deferred into each branch so heavy ML libraries (mlx, llama_cpp)
    are never imported at module load time — backends construct without loading a model.
    """
    settings = get_settings()
    backend = (
        os.getenv("INFERENCE_BACKEND")
        or getattr(settings, "INFERENCE_BACKEND", "mlx")
        or "mlx"
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
