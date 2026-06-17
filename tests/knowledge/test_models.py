import pytest
from pydantic import ValidationError

from app.knowledge.models import (
    Claim,
    ClaimSpan,
    KnowledgeStats,
    PaperSummary,
    TopicSheet,
)


def test_paper_summary_roundtrips_required_fields():
    p = PaperSummary(
        paper_id="2301.12345",
        title="X",
        authors=["A"],
        source_file="papers/x.pdf",
        topics=["rag"],
        claims=[Claim(text="c1", section="Method")],
        methods="m",
        limitations="l",
    )
    assert p.paper_id == "2301.12345"
    assert p.claims[0].text == "c1"


def test_topic_sheet_links_to_paper_ids():
    t = TopicSheet(
        topic_id="rag",
        slug="rag",
        title="RAG",
        body="...",
        links_to=["2301.12345"],
        sources=[{"paper_id": "2301.12345", "section": "Intro"}],
    )
    assert t.links_to == ["2301.12345"]


def test_knowledge_stats_counts():
    s = KnowledgeStats(topics=10, papers=30, chunks=420, links=55)
    assert s.chunks < 1000


def test_claim_span_validation_rejects_empty():
    with pytest.raises(ValidationError):
        ClaimSpan(paper_id="p", text="", section="X", embedding_id="e1")