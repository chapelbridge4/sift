import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.knowledge.artifacts import read_artifact
from app.knowledge.backend import KnowledgeLLM
from app.knowledge.config import load_profile
from app.knowledge.degraded import (
    KnowledgeHardwareError,
    check_hardware_guard,
    degraded_topic_sheet,
    paper_summary_from_spans,
)
from app.knowledge.models import ClaimSpan, PaperSummary, TopicSheet
from app.knowledge.pipeline import KnowledgePipeline
from app.knowledge.tier2_merge import TopicCluster


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
        f"We evaluate {topic} on public benchmarks and show measurable improvements "
        f"over strong baselines in controlled experiments."
    )


def test_paper_summary_from_spans_sets_degraded():
    doc = _ParsedDoc(paper_id="p1", source_file="p1.pdf", title="T")
    spans = [
        ClaimSpan(paper_id="p1", text="Claim one here.", section="abstract", embedding_id="e1"),
    ]
    summary = paper_summary_from_spans(doc, spans)
    assert summary.degraded is True
    assert summary.claims[0].text == "Claim one here."


def test_degraded_topic_sheet_concatenates_and_flags_degraded():
    cluster = TopicCluster(
        cluster_id=1,
        label="rag-topic",
        spans=[
            ClaimSpan(paper_id="p1", text="Span A text.", section="s", embedding_id="e1"),
        ],
    )
    papers = [
        PaperSummary(paper_id="p1", title="P1", source_file="p1.pdf", claims=[]),
    ]
    sheet = degraded_topic_sheet(cluster, papers)
    assert sheet.degraded is True
    assert sheet.links_to == ["p1"]
    assert "Span A text." in sheet.body


def test_check_hardware_guard_raises_on_nonzero_exit(tmp_path):
    script = tmp_path / "guard.sh"
    script.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    script.chmod(0o755)
    with pytest.raises(KnowledgeHardwareError):
        check_hardware_guard(script)


def test_check_hardware_guard_passes_on_zero_exit(tmp_path):
    script = tmp_path / "guard.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    check_hardware_guard(script)  # no raise


def test_pipeline_tier1_fallback_on_llm_failure(tmp_path):
    doc = _ParsedDoc(
        paper_id="p1",
        source_file="p1.pdf",
        title="T",
        sections=[_Section("Abstract", _long("rag systems"))],
    )
    parser = AsyncMock()
    parser.parse_for_knowledge = AsyncMock(return_value=[doc])

    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[1.0, 0.0]])

    backend = AsyncMock()
    backend.generate_structured = AsyncMock(side_effect=ValueError("llm down"))
    llm = KnowledgeLLM(backend, max_retries=0)

    pipeline = KnowledgePipeline(
        parser=parser,
        embedder=embedder,
        llm=llm,
        profile=load_profile("papers"),
        output_dir=tmp_path,
        skip_hardware_guard=True,
    )

    stats = asyncio.run(pipeline.run(["p1.pdf"], "coll"))
    assert stats.papers == 1
    summary = read_artifact(tmp_path / "coll" / "papers" / "p1.md")
    assert summary.degraded is True


def test_pipeline_zero_clusters_papers_only(tmp_path):
    doc = _ParsedDoc(
        paper_id="solo",
        source_file="solo.pdf",
        title="Solo",
        sections=[_Section("Abstract", _long("solo topic research"))],
    )
    parser = AsyncMock()
    parser.parse_for_knowledge = AsyncMock(return_value=[doc])

    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[1.0, 0.0]])  # one span → no clusters

    backend = AsyncMock()
    backend.generate_structured = AsyncMock(
        return_value=(
            '{"title":"Solo","authors":[],"claims":[{"text":"c","section":"abstract"}],'
            '"methods":"","results":"","limitations":"","topic_tags":["solo"]}'
        )
    )
    llm = KnowledgeLLM(backend)

    pipeline = KnowledgePipeline(
        parser=parser,
        embedder=embedder,
        llm=llm,
        profile=load_profile("papers"),
        output_dir=tmp_path,
        skip_hardware_guard=True,
    )

    stats = asyncio.run(pipeline.run(["solo.pdf"], "coll"))
    assert stats.papers == 1
    assert stats.topics == 0
    assert not (tmp_path / "coll" / "topics").exists() or not list((tmp_path / "coll" / "topics").glob("*.md"))


def test_pipeline_tier2_fallback_writes_degraded_topic(tmp_path):
    docs = [
        _ParsedDoc(
            paper_id="p1",
            source_file="p1.pdf",
            title="A",
            sections=[_Section("Abstract", _long("rag retrieval"))],
        ),
        _ParsedDoc(
            paper_id="p2",
            source_file="p2.pdf",
            title="B",
            sections=[_Section("Abstract", _long("rag retrieval methods"))],
        ),
    ]
    parser = AsyncMock()
    parser.parse_for_knowledge = AsyncMock(return_value=docs)

    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[1.0, 0.0], [0.99, 0.01]])

    call_count = 0

    async def _structured(prompt, json_schema, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return (
                '{"title":"T","authors":[],"claims":[{"text":"c","section":"abstract"}],'
                '"methods":"","results":"","limitations":"","topic_tags":["rag"]}'
            )
        raise ValueError("merge failed")

    backend = AsyncMock()
    backend.generate_structured = AsyncMock(side_effect=_structured)
    llm = KnowledgeLLM(backend, max_retries=0)

    pipeline = KnowledgePipeline(
        parser=parser,
        embedder=embedder,
        llm=llm,
        profile=load_profile("papers"),
        output_dir=tmp_path,
        skip_hardware_guard=True,
    )

    stats = asyncio.run(pipeline.run(["p1.pdf", "p2.pdf"], "coll"))
    assert stats.topics >= 1
    topic = read_artifact(next((tmp_path / "coll" / "topics").glob("*.md")))
    assert isinstance(topic, TopicSheet)
    assert topic.degraded is True


def test_pipeline_raises_hardware_error_when_guard_fails(tmp_path):
    guard = tmp_path / "fail_guard.sh"
    guard.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    guard.chmod(0o755)

    parser = AsyncMock()
    parser.parse_for_knowledge = AsyncMock(return_value=[])

    pipeline = KnowledgePipeline(
        parser=AsyncMock(),
        embedder=MagicMock(),
        llm=KnowledgeLLM(AsyncMock()),
        profile=load_profile("papers"),
        output_dir=tmp_path,
        skip_hardware_guard=False,
        hardware_guard_script=guard,
    )
    parser.parse_for_knowledge = AsyncMock(return_value=[])

    with pytest.raises(KnowledgeHardwareError):
        asyncio.run(pipeline.run([], "coll"))