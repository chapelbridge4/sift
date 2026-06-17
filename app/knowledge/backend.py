"""Structured JSON extraction wrapper over an injected inference backend."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from app.config import get_settings
from app.knowledge.config import KnowledgeProfile, load_profile

T = TypeVar("T", bound=BaseModel)

_JSON_RETRY_REMINDER = (
    "\n\nReturn ONLY valid JSON matching the requested schema. "
    "No markdown, no code fences, no commentary."
)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


@runtime_checkable
class KnowledgeBackend(Protocol):
    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 400,
        **kwargs: Any,
    ) -> str: ...


class KnowledgeLLM:
    """DI wrapper: JSON parse, Pydantic validate, retry on malformed output."""

    def __init__(self, backend: Any, *, max_retries: int = 2) -> None:
        self._backend = backend
        self._max_retries = max_retries

    async def extract(
        self,
        prompt: str,
        model_cls: type[T],
        *,
        max_tokens: int,
        temperature: float,
        json_schema: dict[str, Any] | None = None,
    ) -> T:
        current_prompt = prompt
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                raw = await self._generate(
                    current_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_schema=json_schema,
                )
                return model_cls.model_validate(_parse_json_object(raw))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                current_prompt = prompt + _JSON_RETRY_REMINDER

        raise ValueError(
            f"structured extract failed after {self._max_retries + 1} attempts "
            f"for {model_cls.__name__}: {last_error}"
        ) from last_error

    async def _generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        json_schema: dict[str, Any] | None,
    ) -> str:
        if json_schema is not None and hasattr(self._backend, "generate_structured"):
            return await self._backend.generate_structured(
                prompt,
                json_schema,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return await self._backend.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _CODE_FENCE_RE.sub("", text.strip()).strip()
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object, got {type(data).__name__}")
    return data


def _resolve_knowledge_model_path(profile: KnowledgeProfile) -> str:
    settings = get_settings()
    if profile.llm.model_path:
        return profile.llm.model_path
    return settings.KNOWLEDGE_GGUF_MODEL_PATH


def get_knowledge_backend(profile: KnowledgeProfile | str | None = None) -> Any:
    """Factory: GGUF for profile backend=gguf, MLX fallback otherwise."""
    prof = profile if isinstance(profile, KnowledgeProfile) else load_profile(
        profile or get_settings().KNOWLEDGE_PROFILE
    )

    if prof.llm.backend == "gguf":
        from app.services.gguf_service import GGUFService

        return GGUFService(model_path=_resolve_knowledge_model_path(prof))

    from app.services.llm_service import LLMService

    return LLMService()