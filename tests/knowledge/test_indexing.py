import asyncio
from unittest.mock import AsyncMock

from app.knowledge.artifacts import write_artifact
from app.knowledge.config import load_profile
from app.knowledge.index import index_artifacts
from app.knowledge.models import Claim, PaperSummary, TopicSheet


def test_index_artifacts_chunks_with_knowledge_metadata(tmp_path):
    profile = load_profile("papers")
    artifact_dir = tmp_path / "artifacts"
    write_artifact(
        PaperSummary(
            paper_id="2301.1",
            title="RAG Advances",
            source_file="papers/x.pdf",
            topics=["rag"],
            claims=[Claim(text="RAG improves recall.", section="Method")],
            methods="retrieval",
        ),
        artifact_dir,
    )
    write_artifact(
        TopicSheet(
            topic_id="rag",
            slug="rag",
            title="RAG",
            body="## Overview\nRetrieval augmented generation.\n",
            links_to=["2301.1"],
        ),
        artifact_dir,
    )

    captured_texts: list[str] = []
    captured_metas: list[dict] = []

    async def _upsert(collection_name, texts, metadatas, batch_size=32):
        captured_texts.extend(texts)
        captured_metas.extend(metadatas)
        return len(texts)

    qdrant = AsyncMock()
    qdrant.upsert_documents = AsyncMock(side_effect=_upsert)

    total = asyncio.run(
        index_artifacts(
            collection_name="test_coll",
            artifact_dir=artifact_dir,
            profile=profile,
            qdrant_service=qdrant,
        )
    )

    assert total == len(captured_texts) >= 2
    assert qdrant.upsert_documents.await_count == 1

    doc_types = {m["doc_type"] for m in captured_metas}
    assert doc_types == {"paper_summary", "topic"}

    for meta in captured_metas:
        assert meta["knowledge_built"] is True
        assert meta["knowledge_profile"] == "papers"
        assert "doc_id" in meta
        assert "chunk_index" in meta
        assert "heading_path" in meta

    topic_meta = next(m for m in captured_metas if m["doc_type"] == "topic")
    assert topic_meta["links_to"] == ["2301.1"]

    paper_meta = next(m for m in captured_metas if m["doc_type"] == "paper_summary")
    assert paper_meta["source_file"] == "papers/x.pdf"

    # No raw PDF paths indexed
    assert not any(".pdf" in t for t in captured_texts if "Page" in t)


def test_index_artifacts_empty_dir_returns_zero(tmp_path):
    profile = load_profile("papers")
    qdrant = AsyncMock()
    total = asyncio.run(
        index_artifacts(
            collection_name="empty",
            artifact_dir=tmp_path / "empty",
            profile=profile,
            qdrant_service=qdrant,
        )
    )
    assert total == 0
    qdrant.upsert_documents.assert_not_awaited()