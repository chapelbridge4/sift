"""
Optimization wrappers for MLX inference on M1 8GB.

Each optimizer exposes:
  - enabled: bool (from config)
  - wrap_generate(fn) -> wrapped function

All values read from app/config.py. No hardcoded model IDs or paths.
"""

import hashlib
import inspect
import os
from typing import Any, Callable, Dict, Optional

from loguru import logger

from app.config import get_settings

# Module-level KV cache for prefix caching (max 3 entries to respect 8GB limit)
_prefix_cache: Dict[str, tuple[Any, int]] = {}
_prefix_cache_ttl = 300  # seconds


class KVCacheOptimizer:
    """
    KV Cache Quantization — reduces memory footprint of KV cache.

    Config: KV_CACHE_QUANTIZATION = None | "q8" | "q4"
    """

    @property
    def enabled(self) -> bool:
        settings = get_settings()
        return settings.KV_CACHE_QUANTIZATION in ("q8", "q4")

    @property
    def quant_mode(self) -> str | None:
        return get_settings().KV_CACHE_QUANTIZATION

    def wrap_generate(
        self, generate_fn: Callable, model, tokenizer, prompt: str, **kwargs
    ) -> Any:
        """Wrap mlx_vlm.generate with KV cache quantization if supported."""
        if not self.enabled:
            return generate_fn(model, tokenizer, prompt=prompt, **kwargs)

        # Check if mlx_vlm.generate supports kv_cache_quantization
        sig = inspect.signature(generate_fn)
        if "kv_cache_quantization" not in sig.parameters:
            self._log_unsupported()
            return generate_fn(model, tokenizer, prompt=prompt, **kwargs)

        kwargs["kv_cache_quantization"] = self.quant_mode

        log_path = os.path.join(get_settings().BENCHMARK_OUTPUT_DIR, "kv_cache_unsupported.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"KV cache quantization supported: {self.quant_mode}\n")

        return generate_fn(model, tokenizer, prompt=prompt, **kwargs)

    def _log_unsupported(self):
        log_path = os.path.join(get_settings().BENCHMARK_OUTPUT_DIR, "kv_cache_unsupported.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write("KV cache quantization not supported by this mlx-vlm version\n")


class SpeculativeDecoder:
    """
    Speculative Decoding — uses draft model to predict tokens, accepts if confident.

    Config: SPECULATIVE_DECODING_ENABLED, SPECULATIVE_DRAFT_MODEL
    Only for balanced/quality (4B models) — NOT for fast (2B).
    """

    @property
    def enabled(self) -> bool:
        settings = get_settings()
        return settings.SPECULATIVE_DECODING_ENABLED

    @property
    def draft_model(self) -> str:
        return get_settings().SPECULATIVE_DRAFT_MODEL

    @property
    def min_acceptance_rate(self) -> float:
        return settings.SPECULATIVE_MIN_ACCEPTANCE_RATE

    def is_allowed_for_profile(self, profile: str) -> bool:
        """Only enable for 4B models (balanced/quality), not fast (2B)."""
        return profile in ("balanced", "quality")

    def wrap_generate(
        self,
        generate_fn: Callable,
        model,
        tokenizer,
        prompt: str,
        profile: str = "fast",
        **kwargs
    ) -> tuple[Any, dict]:
        """
        Wrap mlx_vlm.generate with speculative decoding.

        Returns:
            tuple of (result, stats_dict) where stats_dict has draft_acceptance_rate
        """
        stats = {"draft_acceptance_rate": None, "speculative_used": False}

        if not self.enabled or not self.is_allowed_for_profile(profile):
            result = generate_fn(model, tokenizer, prompt=prompt, **kwargs)
            return result, stats

        # Check if mlx_vlm.generate supports speculative kwargs
        sig = inspect.signature(generate_fn)
        supported_kwargs = ["draft_model", "num_draft_tokens", "speculative"]
        has_support = any(k in sig.parameters for k in supported_kwargs)

        if not has_support:
            self._log_unsupported()
            result = generate_fn(model, tokenizer, prompt=prompt, **kwargs)
            return result, stats

        # Try speculative decoding
        try:
            kwargs["draft_model"] = self.draft_model
            kwargs["num_draft_tokens"] = 4
            stats["speculative_used"] = True

            result = generate_fn(model, tokenizer, prompt=prompt, **kwargs)

            # If result has acceptance rate metadata, use it
            if hasattr(result, "acceptance_rate"):
                stats["draft_acceptance_rate"] = result.acceptance_rate

            return result, stats

        except Exception as e:
            logger.warning(f"Speculative decoding failed: {e}, falling back to normal")
            stats["speculative_used"] = False
            result = generate_fn(model, tokenizer, prompt=prompt, **kwargs)
            return result, stats

    def _log_unsupported(self):
        log_path = os.path.join(get_settings().BENCHMARK_OUTPUT_DIR, "speculative_unsupported.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write("Speculative decoding not supported by this mlx-vlm version\n")


class PrefixCache:
    """
    Prefix Caching for RAG — caches KV state of system_prompt + retrieved_contexts.

    Config: PREFIX_CACHE_ENABLED, PREFIX_CACHE_MAX_ENTRIES
    """

    @property
    def enabled(self) -> bool:
        return get_settings().PREFIX_CACHE_ENABLED

    @property
    def max_entries(self) -> int:
        return get_settings().PREFIX_CACHE_MAX_ENTRIES

    def cache_key(self, system_prompt: str, retrieved_contexts: list[str]) -> str:
        """Compute cache key from prompt + contexts hash."""
        content = system_prompt + "|".join(retrieved_contexts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_cached(self, cache_key: str) -> Optional[Any]:
        """Return cached KV state if valid."""
        if cache_key in _prefix_cache:
            cached_val, timestamp = _prefix_cache[cache_key]
            import time
            if time.time() - timestamp < _prefix_cache_ttl:
                return cached_val
            else:
                del _prefix_cache[cache_key]
        return None

    def set_cached(self, cache_key: str, kv_state: Any):
        """Store KV state in cache with TTL. Evict oldest if at max entries."""
        import time

        # Evict oldest if at capacity
        if len(_prefix_cache) >= self.max_entries:
            oldest_key = min(_prefix_cache, key=lambda k: _prefix_cache[k][1])
            del _prefix_cache[oldest_key]

        _prefix_cache[cache_key] = (kv_state, time.time())

    def wrap_generate(
        self,
        generate_fn: Callable,
        model,
        tokenizer,
        prompt: str,
        system_prompt: str = "",
        retrieved_contexts: list[str] | None = None,
        **kwargs
    ) -> tuple[Any, dict]:
        """
        Wrap mlx_vlm.generate with prefix caching for RAG workloads.

        Returns:
            tuple of (result, stats_dict) where stats_dict has cache_hit bool
        """
        stats = {"cache_hit": False, "cache_key": None}

        if not self.enabled or not retrieved_contexts:
            result = generate_fn(model, tokenizer, prompt=prompt, **kwargs)
            return result, stats

        cache_key = self.cache_key(system_prompt, retrieved_contexts)
        stats["cache_key"] = cache_key

        # Check cache
        cached_state = self.get_cached(cache_key)
        if cached_state is not None:
            stats["cache_hit"] = True
            logger.debug(f"Prefix cache hit: {cache_key}")

            # If mlx_vlm.generate accepts cached state, use it
            sig = inspect.signature(generate_fn)
            if "cache" in sig.parameters or "prefix_cache" in sig.parameters:
                kwargs["cache"] = cached_state

        result = generate_fn(model, tokenizer, prompt=prompt, **kwargs)

        # If result has kv state we can cache, store it
        if hasattr(result, "kv_state") and result.kv_state is not None:
            self.set_cached(cache_key, result.kv_state)

        return result, stats


# Export singleton instances
kv_optimizer = KVCacheOptimizer()
speculative_decoder = SpeculativeDecoder()
prefix_cache = PrefixCache()