"""
GGUF inference backend using llama-cpp-python.

Provides async text generation compatible with the LLMService interface so the
app can switch between 'gguf' and 'mlx' backends via config INFERENCE_BACKEND.

Target model: unsloth/Qwen3-4B-GGUF  Qwen3-4B-Q4_K_M.gguf
HF repo verified to exist: 2026-06-13

Download (one-time, ~2.5 GB):
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="unsloth/Qwen3-4B-GGUF",
        filename="Qwen3-4B-Q4_K_M.gguf",
        local_dir="~/.cache/gguf",
    )

TODO: automate download behind a flag once network conditions are confirmed.

Usage:
    svc = GGUFService()
    await svc.initialize()          # loads model lazily
    text = await svc.generate("Hello, world!", max_tokens=100)
"""

import asyncio
from pathlib import Path
from typing import Any, List, Optional

from loguru import logger

from app.config import get_settings

_DEFAULT_REPO_ID = "unsloth/Qwen3-4B-GGUF"
_DEFAULT_FILENAME = "Qwen3-4B-Q4_K_M.gguf"


def _ggml_type(name: str):
    """Map a short type name to the corresponding llama_cpp GGML_TYPE_* constant."""
    import llama_cpp
    return {
        "f16": llama_cpp.GGML_TYPE_F16,
        "q8_0": llama_cpp.GGML_TYPE_Q8_0,
        "q4_0": llama_cpp.GGML_TYPE_Q4_0,
        "q5_0": llama_cpp.GGML_TYPE_Q5_0,
    }[name]


def _resolve_model_path(settings, *, model_path: Optional[str] = None) -> Optional[str]:
    """
    Return local path to GGUF model file, or None if not downloaded yet.

    Search order:
    1. model_path constructor override (knowledge backend)
    2. settings.GGUF_MODEL_PATH (explicit override)
    3. ~/.cache/gguf/<filename>
    """
    if model_path and Path(model_path).exists():
        return model_path

    explicit = getattr(settings, "GGUF_MODEL_PATH", None)
    if explicit and Path(explicit).exists():
        return explicit

    default_path = Path.home() / ".cache" / "gguf" / _DEFAULT_FILENAME
    if default_path.exists():
        return str(default_path)

    return None


class GGUFService:
    """
    GGUF-backed text generation service (llama-cpp-python / Metal).

    Lazy-loads the model on first call to initialize(). Safe to construct at
    import time — no heavy work happens in __init__.
    """

    def __init__(self, *, model_path: Optional[str] = None):
        self.settings = get_settings()
        self._model_path_override = model_path
        self._llm = None  # llama_cpp.Llama instance
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Load the GGUF model if not already loaded."""
        async with self._lock:
            if self._llm is not None:
                return

            model_path = _resolve_model_path(
                self.settings, model_path=self._model_path_override
            )
            if model_path is None:
                raise FileNotFoundError(
                    "GGUF model not found. Download it first:\n"
                    "  from huggingface_hub import hf_hub_download\n"
                    f"  hf_hub_download(repo_id='{_DEFAULT_REPO_ID}', "
                    f"filename='{_DEFAULT_FILENAME}', local_dir='~/.cache/gguf')\n"
                    "Or set GGUF_MODEL_PATH env var to the local file path."
                )

            logger.info(f"Loading GGUF model from: {model_path}")
            try:
                self._llm = await asyncio.to_thread(self._load_model, model_path)
                logger.info("GGUF model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load GGUF model: {e}")
                raise

    def _load_model(self, model_path: str):
        """Synchronous model load — runs in thread pool."""
        from llama_cpp import Llama

        n_gpu_layers = getattr(self.settings, "GGUF_N_GPU_LAYERS", -1)
        n_ctx = getattr(self.settings, "GGUF_N_CTX", 4096)
        cache_type_k = getattr(self.settings, "GGUF_CACHE_TYPE_K", "q8_0")
        cache_type_v = getattr(self.settings, "GGUF_CACHE_TYPE_V", "q4_0")

        return Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,  # -1 = offload all layers to Metal
            n_ctx=n_ctx,
            flash_attn=True,            # required for KV-cache quantization
            type_k=_ggml_type(cache_type_k),  # K at q8_0 (keys more sensitive)
            type_v=_ggml_type(cache_type_v),  # V at q4_0 (values more compressible)
            verbose=False,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 400,
        **_kwargs,  # absorb unknown kwargs for interface compatibility
    ) -> str:
        """
        Generate text from a prompt.

        Interface-compatible with LLMService.generate() for the parameters
        that retrieval eval and benchmarks use.
        """
        await self.initialize()

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        logger.debug(f"GGUF generate: prompt_len={len(full_prompt)}, max_tokens={max_tokens}")

        try:
            result = await asyncio.to_thread(
                self._llm,
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
            )
            text = result["choices"][0]["text"]
            logger.debug(f"GGUF generated {len(text)} characters")
            return text.strip()
        except Exception as e:
            logger.error(f"GGUF generate error: {e}")
            raise

    async def generate_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        *,
        temperature: float = 0.2,
        max_tokens: int = 400,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate JSON constrained by schema via llama-cpp response_format."""
        await self.initialize()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response_format: dict[str, Any] = {"type": "json_object"}
        if json_schema:
            response_format["schema"] = json_schema

        logger.debug(
            "GGUF generate_structured: prompt_len=%s, max_tokens=%s",
            len(prompt),
            max_tokens,
        )

        try:
            result = await asyncio.to_thread(
                self._llm.create_chat_completion,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
            text = result["choices"][0]["message"]["content"]
            logger.debug("GGUF structured generated %s characters", len(text))
            return text.strip()
        except Exception as e:
            logger.error(f"GGUF generate_structured error: {e}")
            raise

    async def generate_rag_response(
        self,
        query: str,
        retrieved_contexts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model_profile: Optional[str] = None,
        **_kwargs,
    ) -> str:
        """
        Generate RAG response. Interface-compatible with LLMService.generate_rag_response().
        """
        await self.initialize()

        settings = self.settings
        final_temp = temperature if temperature is not None else 0.7
        final_max_tokens = max_tokens if max_tokens is not None else 400

        context_str = "\n\n---\n\n".join(
            f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(retrieved_contexts)
        )

        system_prompt = (
            "You are a helpful AI assistant. Answer questions concisely "
            "based on the provided context. If the context does not contain "
            "enough information, say so briefly."
        )

        prompt = (
            f"{system_prompt}\n\n"
            f"Context Information:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        return await self.generate(
            prompt=prompt,
            temperature=final_temp,
            max_tokens=final_max_tokens,
        )

    async def close(self) -> None:
        """Release model resources."""
        if self._llm is not None:
            logger.info("Releasing GGUF model")
            self._llm = None
