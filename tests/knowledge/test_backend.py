import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from app.knowledge.backend import KnowledgeLLM


class Out(BaseModel):
    title: str
    claims: list[str]


def test_extract_parses_valid_json():
    backend = AsyncMock()
    backend.generate = AsyncMock(return_value='{"title": "T", "claims": ["a", "b"]}')
    llm = KnowledgeLLM(backend, max_retries=2)
    out = asyncio.run(llm.extract("prompt", Out, max_tokens=200, temperature=0.2))
    assert out.title == "T" and out.claims == ["a", "b"]


def test_extract_retries_then_succeeds():
    backend = AsyncMock()
    backend.generate = AsyncMock(side_effect=["not json", '{"title":"T","claims":[]}'])
    llm = KnowledgeLLM(backend, max_retries=2)
    out = asyncio.run(llm.extract("p", Out, max_tokens=200, temperature=0.2))
    assert out.title == "T"
    assert backend.generate.await_count == 2


def test_extract_raises_after_max_retries():
    backend = AsyncMock()
    backend.generate = AsyncMock(return_value="never json")
    llm = KnowledgeLLM(backend, max_retries=2)
    with pytest.raises(ValueError):
        asyncio.run(llm.extract("p", Out, max_tokens=200, temperature=0.2))


def test_extract_prefers_generate_structured():
    backend = AsyncMock()
    backend.generate_structured = AsyncMock(
        return_value='{"title": "S", "claims": ["x"]}'
    )
    backend.generate = AsyncMock()
    llm = KnowledgeLLM(backend, max_retries=2)
    out = asyncio.run(
        llm.extract("p", Out, max_tokens=200, temperature=0.2, json_schema={"type": "object"})
    )
    assert out.title == "S"
    backend.generate_structured.assert_awaited_once()
    backend.generate.assert_not_awaited()


def test_extract_strips_code_fences():
    backend = AsyncMock()
    backend.generate = AsyncMock(
        return_value='```json\n{"title": "T", "claims": []}\n```'
    )
    llm = KnowledgeLLM(backend, max_retries=2)
    out = asyncio.run(llm.extract("p", Out, max_tokens=200, temperature=0.2))
    assert out.title == "T"