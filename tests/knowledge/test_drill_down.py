import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.schemas import QueryRequest
from app.pipeline.document_store import DocumentStore
from app.pipeline.orchestrator import RagOrchestrator


def _topic_memory(score: float = 0.9, links_to: list[str] | None = None) -> dict:
    return {
        "id": "topic-rag",
        "score": score,
        "text": "Topic overview of RAG patterns.",
        "metadata": {
            "doc_type": "topic",
            "doc_id": "rag",
            "links_to": links_to or ["2301.1", "2305.2"],
            "knowledge_built": True,
            "knowledge_profile": "papers",
        },
    }


def _paper_memory(paper_id: str, score: float = 0.7) -> dict:
    return {
        "id": f"paper-{paper_id}",
        "score": score,
        "text": f"Paper {paper_id} summary.",
        "metadata": {
            "doc_type": "paper_summary",
            "doc_id": paper_id,
            "knowledge_built": True,
            "knowledge_profile": "papers",
            "source_file": f"papers/{paper_id}.pdf",
        },
    }


def test_query_request_accepts_drill_down_flag():
    request = QueryRequest(
        collection_name="knowledge_coll",
        query="Which paper introduced HyDE?",
        drill_down=True,
    )
    assert request.drill_down is True


def test_query_request_drill_down_defaults_false():
    request = QueryRequest(collection_name="coll", query="test")
    assert request.drill_down is False


def test_recall_memories_drill_down_fetches_linked_papers():
    store = DocumentStore()
    store.qdrant_service = AsyncMock()

    topic_hit = _topic_memory()
    paper_hits = [_paper_memory("2301.1", 0.75)]

    store.qdrant_service.hybrid_search = AsyncMock(
        side_effect=[[topic_hit], paper_hits]
    )

    async def _run():
        return await store.recall_memories(
            collection_name="knowledge_coll",
            query="HyDE method details",
            top_k=5,
            drill_down=True,
        )

    results = asyncio.run(_run())

    assert store.qdrant_service.hybrid_search.await_count == 2
    second_call = store.qdrant_service.hybrid_search.await_args_list[1]
    assert second_call.kwargs["filters"] is not None

    doc_types = [r["metadata"]["doc_type"] for r in results]
    assert doc_types[0] == "topic"
    assert "paper_summary" in doc_types


def test_recall_memories_skips_second_pass_when_drill_down_false():
    store = DocumentStore()
    store.qdrant_service = AsyncMock()
    store.qdrant_service.hybrid_search = AsyncMock(return_value=[_topic_memory()])

    async def _run():
        return await store.recall_memories(
            collection_name="knowledge_coll",
            query="overview",
            drill_down=False,
        )

    asyncio.run(_run())
    assert store.qdrant_service.hybrid_search.await_count == 1


@patch("app.pipeline.orchestrator.ConversationMemory")
@patch("app.pipeline.orchestrator.Reranker")
@patch("app.pipeline.orchestrator.DocumentStore")
def test_orchestrator_exposes_retrieval_layers_on_drill_down(
    mock_document_store,
    mock_reranker,
    mock_conversation_memory,
):
    topic = _topic_memory()
    paper = _paper_memory("2301.1")
    merged = [topic, paper]

    pfc = RagOrchestrator(llm_service=AsyncMock())
    pfc.document_store = AsyncMock()
    pfc.reranker = MagicMock()
    pfc.conversation_memory = AsyncMock()
    pfc.llm_service = AsyncMock()
    pfc.llm_service.get_model_for_request = MagicMock(return_value=("test-model", {}))
    pfc.llm_service.generate_rag_response = AsyncMock(return_value="Answer with citation.")

    pfc.document_store.recall_memories = AsyncMock(return_value=merged)
    pfc.reranker.rank_by_importance = MagicMock(return_value=merged)

    async def _run():
        return await pfc.reason_with_context(
            query="method details",
            collection_name="knowledge_coll",
            drill_down=True,
            use_llm=True,
        )

    result = asyncio.run(_run())

    assert result["retrieval_layers"] == ["topic", "paper"]
    assert len(result["sources"]) == 2
    layers = {s["layer"] for s in result["sources"]}
    assert layers == {"topic", "paper"}