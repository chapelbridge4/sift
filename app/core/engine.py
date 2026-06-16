"""
Narrow-purpose mlx-vlm engine for Qwen3.5 on Apple Silicon.
Adapted from ds4 philosophy: single model family, first-class KV persistence,
zero abstraction overhead.
"""

import hashlib
import re
from pathlib import Path
from typing import Optional

from app.config import get_settings

# Module-level engine cache (singleton per profile)
_engines = {}


def _get_engine(profile: str = "fast") -> "QwenEngine":
    """Lazy-load or return cached QwenEngine for a profile."""
    if profile not in _engines:
        settings = get_settings()
        model_id = settings.MODEL_PROFILES[profile]["model"]
        _engines[profile] = QwenEngine(model_id)
    return _engines[profile]


class QwenEngine:
    """
    Owns: model weights, processor, tokenizer, KV cache lifecycle.

    All numeric tunables come from config. No hardcoding.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.settings = get_settings()
        self.cache_dir = Path(self.settings.KV_CACHE_DIR)
        self.max_sessions = int(self.settings.KV_MAX_SESSIONS)
        self.max_mb_per_file = int(self.settings.KV_MAX_MB_PER_FILE)

        # Load model via mlx-vlm
        from mlx_vlm import load

        self.model, self.processor = load(model_id)

        # Warm-up to pre-allocate Metal buffers (avoid first-request delay)
        if self.settings.ENGINE_WARMUP:
            self._warmup()

    def _warmup(self):
        """Run a single token through to pre-allocate Metal buffers."""
        from mlx_vlm import generate

        try:
            generate(
                self.model,
                self.processor,
                prompt="hi",
                max_tokens=1,
                verbose=False,
            )
        except Exception:
            pass

    def chat(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        session_id: Optional[str] = None,
        image: Optional[str] = None,
    ) -> str:
        """
        Single chat completion call.

        Args:
            messages: [{role: str, content: str}, ...]
            max_tokens: Override profile default if provided
            temperature: Override profile default if provided
            repetition_penalty: Override profile default if provided
            session_id: KV cache session identifier (for persistence)
            image: Optional image path for vision-enabled models
        """
        # Resolve config defaults
        profile_cfg = self.settings.MODEL_PROFILES.get(
            self.settings.MODEL_PROFILE,
            self.settings.MODEL_PROFILES["fast"]
        )

        final_max_tokens = (
            max_tokens
            if max_tokens is not None
            else profile_cfg.get("max_tokens", 400)
        )
        final_temp = (
            temperature
            if temperature is not None
            else profile_cfg.get("temperature", 0.1)
        )
        final_rep_pen = (
            repetition_penalty
            if repetition_penalty is not None
            else self.settings.REPETITION_PENALTY
        )

        # Build prompt with thinking disabled at template level
        prompt = self._build_prompt(messages)

        # Load KV from disk if session provided
        kv_cache = self._load_kv(session_id) if session_id else None

        # Build generate kwargs
        kwargs = {
            "prompt": prompt,
            "max_tokens": final_max_tokens,
            "temp": final_temp,
            "repetition_penalty": final_rep_pen,
            "verbose": False,
        }

        # Pass image if provided
        if image:
            kwargs["image"] = image

        # Pass KV cache if supported
        if kv_cache is not None:
            try:
                kwargs["kv_cache"] = kv_cache
            except TypeError:
                pass

        # Generate
        from mlx_vlm import generate

        output = generate(self.model, self.processor, **kwargs)

        # Save KV to disk if session provided
        if session_id:
            new_kv = getattr(self.model, "cache", None)
            if new_kv is not None:
                self._save_kv(session_id, new_kv)

        # Sanitize and return
        return self._sanitize(output)

    def _build_prompt(self, messages: list[dict]) -> str:
        """Build prompt with /no_think guard and thinking disabled at template level."""
        # Prompt guard: prepend /no_think to system prompt for Qwen models
        if "qwen" in self.model_id.lower() and messages:
            if messages[0].get("role") == "system":
                content = messages[0].get("content", "")
                if not content.startswith("/no_think"):
                    messages = messages.copy()
                    messages[0] = messages[0].copy()
                    messages[0]["content"] = "/no_think " + content

        # Apply chat template with thinking disabled
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            template_kwargs["enable_thinking"] = False
        except TypeError:
            pass  # Older tokenizer versions may reject this

        return self.processor.tokenizer.apply_chat_template(messages, **template_kwargs)

    def _sanitize(self, text: str) -> str:
        """Strip any leaked thinking blocks from Qwen model output."""
        # Strip Thinking Process blocks
        text = re.sub(r"Thinking Process:.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
        # Strip Chinese thinking tokens
        text = text.replace("(思考中)", "").replace("(思考完毕)", "").strip()
        return text

    def _session_hash(self, session_id: str) -> str:
        """Derive a short hash for the session ID."""
        return hashlib.sha256(session_id.encode()).hexdigest()[:16]

    def _load_kv(self, session_id: str) -> Optional[dict]:
        """Load KV cache from disk for a session. Returns None if not found."""
        key = self._session_hash(session_id)
        path = self.cache_dir / f"{key}.npz"
        if not path.exists():
            return None
        try:
            import mlx.core as mx
            data = mx.load(str(path))
            return data
        except Exception:
            return None

    def _save_kv(self, session_id: str, kv_cache) -> None:
        """Save KV cache to disk with LRU eviction and size cap."""
        key = self._session_hash(session_id)
        path = self.cache_dir / f"{key}.npz"
        try:
            import mlx.core as mx
            mx.savez(str(path), cache=kv_cache)
        except Exception as e:
            print(f"KV save failed: {e}")
            return

        # LRU eviction: keep only max_sessions newest files
        files = sorted(
            self.cache_dir.glob("*.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        for old in files[self.max_sessions:]:
            old.unlink(missing_ok=True)

        # Size cap: if any file exceeds max_mb, delete it
        for f in files[:self.max_sessions]:
            if f.stat().st_size > self.max_mb_per_file * 1024 * 1024:
                f.unlink(missing_ok=True)

    def clear_cache(self) -> None:
        """Clear all KV cache files from disk."""
        for f in self.cache_dir.glob("*.npz"):
            f.unlink(missing_ok=True)
