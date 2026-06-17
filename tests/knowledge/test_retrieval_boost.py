import asyncio
from unittest.mock import AsyncMock

from app.knowledge.retrieval import apply_topic_score_boost
from app.pipeline.document_store import DocumentStore


def _memory(doc_type: str, score: float, doc_id: str = "x") -> dict:
    return {
        "id": f"{doc_type}-{doc_id}",
        "score": score,
        "text": f"{doc_type} text",
        "metadata": {
            "doc_type": doc_type,
            "doc_id": doc_id,
            "knowledge_built": True,
            "knowledge_profile": "papers",
        },
    }


def test_apply_topic_score_boost_reranks_topic_above_paper():
    memories = [
        _memory("paper_summary", 0.9, "p1"),
        _memory("topic", 0.8, "rag"),
    ]

    boosted = apply_topic_score_boost(memories, boost=1.2)

    assert boosted[0]["metadata"]["doc_type"] == "topic"
    assert boosted[0]["score"] == 0.8 * 1.2
    assert boosted[0]["metadata"]["topic_boost_applied"] is True
    assert boosted[1]["metadata"]["doc_type"] == "paper_summary"
    assert boosted[1]["score"] == 0.9


def test_recall_memories_applies_topic_boost_for_knowledge_collection():
    store = DocumentStore()
    store.qdrant_service = AsyncMock()
    raw = [
        _memory("paper_summary", 0.95, "2301.1"),
        _memory("topic", 0.85, "rag"),
    ]
    store.qdrant_service.hybrid_search = AsyncMock(return_value=raw)

    async def _run():
        return await store.recall_memories(
            collection_name="knowledge_coll",
            query="retrieval augmented generation",
            top_k=10,
            use_hybrid=True,
        )

    results = asyncio.run(_run())

    assert results[0]["metadata"]["doc_type"] == "topic"
    assert results[0]["score"] == 0.85 * 1.2


def test_recall_memories_unchanged_for_non_knowledge_collection():
    store = DocumentStore()
    store.qdrant_service = AsyncMock()
    raw = [
        {
            "id": "raw-1",
            "score": 0.9,
            "text": "plain chunk",
            "metadata": {"source_file": "doc.pdf", "chunk_index": 0},
        }
    ]
    store.qdrant_service.hybrid_search = AsyncMock(return_value=raw)

    async def _run():
        return await store.recall_memories(
            collection_name="raw_coll",
            query="test",
            top_k=5,
        )

    results = asyncio.run(_run())

    assert results[0]["score"] == 0.9
    assert "topic_boost_applied" not in results[0]["metadata"]