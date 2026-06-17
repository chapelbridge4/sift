"""Slow mocked end-to-end: ingest → topic query → drill-down returns paper chunk."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.knowledge.artifacts import read_artifact
from app.knowledge.backend import KnowledgeLLM
from app.knowledge.config import load_profile
from app.knowledge.index import index_artifacts
from app.knowledge.models import TopicSheet
from app.knowledge.pipeline import KnowledgePipeline
from app.pipeline.document_store import DocumentStore

RAW_INGEST_BASELINE_CHUNKS = 66_941


@dataclass
class _Section:
    name: str
    text: str


@dataclass
class _ParsedDoc:
    paper_id: str
    source_file: str
    title: str
    authors: list[str] = field(default_factory=list)
    sections: list[_Section] = field(default_factory=list)


def _claim_text(topic: str) -> str:
    return (
        f"We study {topic} across three benchmarks and report consistent gains "
        f"over competitive baselines on retrieval quality metrics."
    )


def _three_paper_fixture() -> list[_ParsedDoc]:
    return [
        _ParsedDoc(
            paper_id="p1",
            source_file="papers/rag_a.pdf",
            title="RAG Paper A",
            sections=[_Section("Abstract", _claim_text("retrieval augmented generation"))],
        ),
        _ParsedDoc(
            paper_id="p2",
            source_file="papers/rag_b.pdf",
            title="RAG Paper B",
            sections=[_Section("Abstract", _claim_text("retrieval augmented generation"))],
        ),
        _ParsedDoc(
            paper_id="p3",
            source_file="papers/quant_c.pdf",
            title="Quant Paper",
            sections=[_Section("Abstract", _claim_text("quantization for edge devices"))],
        ),
    ]


def _mock_embedder(span_count: int) -> MagicMock:
    vectors = []
    for i in range(span_count):
        vectors.append([1.0, 0.0] if i < span_count // 2 else [0.0, 1.0])
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=vectors)
    return embedder


def _mock_llm_backend() -> AsyncMock:
    backend = AsyncMock()

    async def _structured(prompt, json_schema, **kwargs):
        if "Paper ID:" in prompt:
            if "Paper ID: p1" in prompt:
                pid, title = "p1", "RAG Paper A"
            elif "Paper ID: p2" in prompt:
                pid, title = "p2", "RAG Paper B"
            else:
                pid, title = "p3", "Quant Paper"
            return json.dumps(
                {
                    "title": title,
                    "authors": [],
                    "claims": [{"text": "Key claim.", "section": "abstract"}],
                    "methods": "m",
                    "results": "r",
                    "limitations": "l",
                    "topic_tags": ["rag" if pid != "p3" else "quant"],
                }
            )
        return json.dumps(
            {
                "title": "Retrieval Augmented Generation",
                "slug": "rag",
                "body": "## Overview\nMerged RAG topic.",
                "sources": [{"paper_id": "p1", "section": "abstract"}],
            }
        )

    backend.generate_structured = AsyncMock(side_effect=_structured)
    return backend


class _InMemoryQdrant:
    """Minimal Qdrant stub that stores upserted chunks for recall tests."""

    def __init__(self) -> None:
        self._points: list[dict] = []
        self._next_id = 1

    async def initialize(self) -> None:
        return None

    async def upsert_documents(
        self,
        collection_name: str,
        texts: list[str],
        metadatas: list[dict],
        batch_size: int = 32,
    ) -> int:
        for text, meta in zip(texts, metadatas, strict=True):
            self._points.append(
                {
                    "id": f"pt-{self._next_id}",
                    "score": 0.5,
                    "text": text,
                    "metadata": dict(meta),
                }
            )
            self._next_id += 1
        return len(texts)

    async def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 10,
        fusion_method: str = "rrf",
        filters=None,
    ) -> list[dict]:
        candidates = list(self._points)

        if filters is not None:
            must = getattr(filters, "must", []) or []
            allowed_doc_types: set[str] = set()
            allowed_doc_ids: set[str] = set()
            for cond in must:
                key = getattr(cond, "key", None)
                match = getattr(cond, "match", None)
                if key == "doc_type" and hasattr(match, "value"):
                    allowed_doc_types.add(match.value)
                if key == "doc_id" and hasattr(match, "any"):
                    allowed_doc_ids.update(match.any)

            filtered = []
            for pt in candidates:
                meta = pt["metadata"]
                if allowed_doc_types and meta.get("doc_type") not in allowed_doc_types:
                    continue
                if allowed_doc_ids and meta.get("doc_id") not in allowed_doc_ids:
                    continue
                filtered.append(pt)
            candidates = filtered

        for pt in candidates:
            meta = pt["metadata"]
            if meta.get("doc_type") == "topic":
                pt["score"] = 0.95
            elif meta.get("doc_type") == "paper_summary":
                pt["score"] = 0.75

        candidates.sort(key=lambda p: p["score"], reverse=True)
        return candidates[:top_k]


@pytest.mark.slow
def test_e2e_ingest_query_drill_down_returns_paper_chunk(tmp_path):
    docs = _three_paper_fixture()
    parser = AsyncMock()
    parser.parse_for_knowledge = AsyncMock(return_value=docs)

    embedder = _mock_embedder(len(docs))
    llm = KnowledgeLLM(_mock_llm_backend())
    profile = load_profile("papers")

    pipeline = KnowledgePipeline(
        parser=parser,
        embedder=embedder,
        llm=llm,
        profile=profile,
        output_dir=tmp_path,
        skip_hardware_guard=True,
    )

    collection = "e2e_knowledge"
    file_paths = [d.source_file for d in docs]

    stats = asyncio.run(pipeline.run(file_paths, collection))

    assert stats.papers == 3
    assert stats.topics >= 1
    assert (tmp_path / collection / "cluster_manifest.json").exists()
    assert len(list((tmp_path / collection / "papers").glob("*.md"))) == 3
    assert len(list((tmp_path / collection / "topics").glob("*.md"))) >= 1

    for topic_path in (tmp_path / collection / "topics").glob("*.md"):
        sheet = read_artifact(topic_path)
        assert isinstance(sheet, TopicSheet)
        assert len(sheet.links_to) >= 1

    qdrant = _InMemoryQdrant()
    indexed = asyncio.run(
        index_artifacts(
            collection_name=collection,
            artifact_dir=tmp_path / collection,
            profile=profile,
            qdrant_service=qdrant,
        )
    )

    assert indexed > 0
    assert indexed < RAW_INGEST_BASELINE_CHUNKS

    store = DocumentStore()
    store.qdrant_service = qdrant

    topic_results = asyncio.run(
        store.recall_memories(
            collection_name=collection,
            query="retrieval augmented generation overview",
            top_k=5,
            use_hybrid=True,
            drill_down=False,
        )
    )
    assert any(r["metadata"]["doc_type"] == "topic" for r in topic_results)

    drill_results = asyncio.run(
        store.recall_memories(
            collection_name=collection,
            query="retrieval augmented generation method details",
            top_k=5,
            use_hybrid=True,
            drill_down=True,
        )
    )

    doc_types = [r["metadata"]["doc_type"] for r in drill_results]
    assert doc_types[0] == "topic"
    assert "paper_summary" in doc_types