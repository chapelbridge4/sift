import asyncio
from unittest.mock import AsyncMock

from app.knowledge.backend import KnowledgeLLM
from app.knowledge.config import load_profile
from app.knowledge.models import Claim, ClaimSpan, PaperSummary
from app.knowledge.tier2_merge import TopicCluster, merge_topic


def test_merge_topic_populates_links_to_and_preserves_conflicts():
    cluster = TopicCluster(
        cluster_id=3,
        label="retrieval-augmented-generation",
        spans=[
            ClaimSpan(
                paper_id="2301.1",
                text="RAG improves recall on long corpora.",
                section="results",
                embedding_id="e1",
            ),
            ClaimSpan(
                paper_id="2301.2",
                text="RAG may hurt latency on small corpora.",
                section="limitations",
                embedding_id="e2",
            ),
        ],
    )
    papers = [
        PaperSummary(
            paper_id="2301.1",
            title="Paper A",
            source_file="papers/a.pdf",
            topics=["rag"],
            claims=[Claim(text="RAG improves recall on long corpora.", section="results")],
        ),
        PaperSummary(
            paper_id="2301.2",
            title="Paper B",
            source_file="papers/b.pdf",
            topics=["rag"],
            claims=[
                Claim(text="RAG may hurt latency on small corpora.", section="limitations")
            ],
        ),
    ]

    backend = AsyncMock()
    backend.generate_structured = AsyncMock(
        return_value=(
            '{"title":"Retrieval-Augmented Generation","slug":"rag",'
            '"body":"## Overview\\nMerged topic.\\n\\n## Nuances\\n| Paper | Claim |\\n|---|---|\\n| 2301.1 | improves recall |\\n| 2301.2 | latency tradeoff |",'
            '"sources":[{"paper_id":"2301.1","section":"results"},{"paper_id":"2301.2","section":"limitations"}]}'
        )
    )
    llm = KnowledgeLLM(backend)
    profile = load_profile("papers")

    sheet = asyncio.run(merge_topic(cluster, papers, llm, profile))

    assert sheet.links_to == ["2301.1", "2301.2"]
    assert sheet.slug == "rag"
    assert "Nuances" in sheet.body
    assert {s.paper_id for s in sheet.sources} == {"2301.1", "2301.2"}