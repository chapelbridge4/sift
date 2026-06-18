import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock

from app.knowledge.backend import KnowledgeLLM
from app.knowledge.config import load_profile
from app.knowledge.models import ClaimSpan
from app.knowledge.tier1_extract import extract_paper


@dataclass
class _Section:
    name: str
    text: str


@dataclass
class _ParsedDoc:
    paper_id: str
    source_file: str
    title: str
    authors: list[str]
    sections: list[_Section]


def test_extract_paper_calls_llm_and_returns_summary(caplog):
    parsed = _ParsedDoc(
        paper_id="2301.12345",
        source_file="papers/x.pdf",
        title="RAG Advances",
        authors=["Alice"],
        sections=[_Section("abstract", "We improve retrieval augmented generation.")],
    )
    spans = [
        ClaimSpan(
            paper_id="2301.12345",
            text="RAG improves factual grounding.",
            section="abstract",
            embedding_id="e1",
        )
    ]

    backend = AsyncMock()
    backend.generate_structured = AsyncMock(
        return_value=(
            '{"title":"RAG Advances","authors":["Alice"],'
            '"claims":[{"text":"RAG improves factual grounding.","section":"abstract"}],'
            '"methods":"retrieval","results":"better recall","limitations":"latency",'
            '"topic_tags":["rag"]}'
        )
    )
    llm = KnowledgeLLM(backend)
    profile = load_profile("papers")

    summary = asyncio.run(extract_paper(parsed, spans, llm, profile))

    assert summary.paper_id == "2301.12345"
    assert summary.title == "RAG Advances"
    assert summary.topics == ["rag"]
    assert summary.claims[0].text == "RAG improves factual grounding."
    backend.generate_structured.assert_awaited_once()

    # No raw section text in logs (only structured metadata).
    for record in caplog.records:
        assert "We improve retrieval augmented generation." not in record.message


def test_extract_paper_caps_spans_in_prompt():
    parsed = _ParsedDoc(
        paper_id="p1",
        source_file="papers/p1.pdf",
        title="T",
        authors=[],
        sections=[_Section("abstract", "Short abstract.")],
    )
    spans = [
        ClaimSpan(
            paper_id="p1",
            text=f"Claim number {i} with enough text to matter.",
            section="abstract",
            embedding_id=f"e{i}",
        )
        for i in range(50)
    ]
    backend = AsyncMock()
    backend.generate_structured = AsyncMock(
        return_value=(
            '{"title":"T","authors":[],"claims":[],"methods":"","results":"",'
            '"limitations":"","topic_tags":[]}'
        )
    )
    llm = KnowledgeLLM(backend)
    profile = load_profile("papers")

    asyncio.run(extract_paper(parsed, spans, llm, profile))

    prompt = backend.generate_structured.await_args.args[0]
    assert prompt.count("Claim number") == profile.tier1.max_spans_per_paper


def test_extract_paper_uses_profile_token_cap():
    parsed = _ParsedDoc(
        paper_id="p1",
        source_file="papers/p1.pdf",
        title="T",
        authors=[],
        sections=[_Section("abstract", "Short abstract.")],
    )
    backend = AsyncMock()
    backend.generate_structured = AsyncMock(
        return_value=(
            '{"title":"T","authors":[],"claims":[],"methods":"","results":"",'
            '"limitations":"","topic_tags":[]}'
        )
    )
    llm = KnowledgeLLM(backend)
    profile = load_profile("papers")

    asyncio.run(extract_paper(parsed, [], llm, profile))

    call_kwargs = backend.generate_structured.await_args.kwargs
    assert call_kwargs["max_tokens"] == profile.tier1.max_output_tokens