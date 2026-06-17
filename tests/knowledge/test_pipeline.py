import asyncio
import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

from app.knowledge.artifacts import read_artifact
from app.knowledge.backend import KnowledgeLLM
from app.knowledge.config import load_profile
from app.knowledge.models import TopicSheet
from app.knowledge.pipeline import KnowledgePipeline


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


def _long(topic: str) -> str:
    return (
        f"We study {topic} across three benchmarks and report consistent gains "
        f"over competitive baselines on retrieval quality metrics."
    )


def _three_doc_fixture() -> list[_ParsedDoc]:
    return [
        _ParsedDoc(
            paper_id="p1",
            source_file="papers/a.pdf",
            title="RAG Paper A",
            sections=[_Section("Abstract", _long("retrieval augmented generation"))],
        ),
        _ParsedDoc(
            paper_id="p2",
            source_file="papers/b.pdf",
            title="RAG Paper B",
            sections=[_Section("Abstract", _long("retrieval augmented generation"))],
        ),
        _ParsedDoc(
            paper_id="p3",
            source_file="papers/c.pdf",
            title="Quant Paper",
            sections=[_Section("Abstract", _long("quantization for edge devices"))],
        ),
    ]


def _mock_embedder_for_spans(spans_count: int) -> MagicMock:
    """Two tight groups in embedding space (rag pair + quant singleton pair)."""
    vectors = []
    for i in range(spans_count):
        if i < spans_count // 2:
            vectors.append([1.0, 0.0])
        else:
            vectors.append([0.0, 1.0])
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=vectors)
    return embedder


def _mock_llm_backend() -> AsyncMock:
    backend = AsyncMock()

    async def _structured(prompt, json_schema, **kwargs):
        if "Paper ID:" in prompt:
            if "Paper ID: p1" in prompt:
                pid = "p1"
                title = "RAG Paper A"
            elif "Paper ID: p2" in prompt:
                pid = "p2"
                title = "RAG Paper B"
            else:
                pid = "p3"
                title = "Quant Paper"
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
                "title": "Merged Topic",
                "slug": "merged-topic",
                "body": "## Overview\nMerged.",
                "sources": [{"paper_id": "p1", "section": "abstract"}],
            }
        )

    backend.generate_structured = AsyncMock(side_effect=_structured)
    return backend


def test_pipeline_run_writes_artifacts_and_returns_stats(tmp_path):
    docs = _three_doc_fixture()
    parser = AsyncMock()
    parser.parse_for_knowledge = AsyncMock(return_value=docs)

    # Estimate span count: 3 docs × ~1 span each from abstract
    embedder = _mock_embedder_for_spans(3)
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

    stats = asyncio.run(pipeline.run(["papers/a.pdf", "papers/b.pdf", "papers/c.pdf"], "test_coll"))

    assert stats.papers == 3
    assert stats.topics >= 1
    assert (tmp_path / "test_coll" / "cluster_manifest.json").exists()
    assert len(list((tmp_path / "test_coll" / "papers").glob("*.md"))) == 3
    assert len(list((tmp_path / "test_coll" / "topics").glob("*.md"))) >= 1

    for topic_path in (tmp_path / "test_coll" / "topics").glob("*.md"):
        sheet = read_artifact(topic_path)
        assert isinstance(sheet, TopicSheet)
        assert len(sheet.links_to) >= 1

    # Canonical artifacts << raw PDF chunk baseline (66k+)
    artifact_count = stats.papers + stats.topics
    assert artifact_count < 1000


def test_pipeline_skips_unparseable_files(tmp_path):
    parser = AsyncMock()
    parser.parse_for_knowledge = AsyncMock(return_value=_three_doc_fixture()[:2])

    embedder = _mock_embedder_for_spans(2)
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

    stats = asyncio.run(pipeline.run(["papers/a.pdf", "papers/b.pdf", "bad.pdf"], "coll"))

    assert stats.papers == 2