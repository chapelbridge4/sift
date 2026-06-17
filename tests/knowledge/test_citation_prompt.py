import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.knowledge.retrieval import KNOWLEDGE_CITATION_INSTRUCTION
from app.pipeline.orchestrator import RagOrchestrator


def _knowledge_memory(doc_type: str = "topic") -> dict:
    return {
        "id": "k-1",
        "score": 0.9,
        "text": "Knowledge context chunk.",
        "metadata": {
            "doc_type": doc_type,
            "doc_id": "rag",
            "knowledge_built": True,
            "knowledge_profile": "papers",
        },
    }


@patch("app.pipeline.orchestrator.ConversationMemory")
@patch("app.pipeline.orchestrator.Reranker")
@patch("app.pipeline.orchestrator.DocumentStore")
def test_orchestrator_adds_citation_instruction_for_knowledge_collection(
    mock_document_store,
    mock_reranker,
    mock_conversation_memory,
):
    memories = [_knowledge_memory()]

    pfc = RagOrchestrator(llm_service=AsyncMock())
    pfc.document_store = AsyncMock()
    pfc.reranker = MagicMock()
    pfc.conversation_memory = AsyncMock()
    pfc.llm_service = AsyncMock()
    pfc.llm_service.get_model_for_request = MagicMock(return_value=("test-model", {}))
    pfc.llm_service.generate_rag_response = AsyncMock(return_value="See [topic:rag].")

    pfc.document_store.recall_memories = AsyncMock(return_value=memories)
    pfc.reranker.rank_by_importance = MagicMock(return_value=memories)

    async def _run():
        return await pfc.reason_with_context(
            query="What is RAG?",
            collection_name="knowledge_coll",
            use_llm=True,
        )

    asyncio.run(_run())

    call_kwargs = pfc.llm_service.generate_rag_response.await_args.kwargs
    assert call_kwargs["extra_system_instruction"] == KNOWLEDGE_CITATION_INSTRUCTION


@patch("app.pipeline.orchestrator.ConversationMemory")
@patch("app.pipeline.orchestrator.Reranker")
@patch("app.pipeline.orchestrator.DocumentStore")
def test_orchestrator_omits_citation_instruction_for_raw_collection(
    mock_document_store,
    mock_reranker,
    mock_conversation_memory,
):
    memories = [
        {
            "id": "raw-1",
            "score": 0.8,
            "text": "Plain document chunk.",
            "metadata": {"source_file": "doc.pdf", "chunk_index": 0},
        }
    ]

    pfc = RagOrchestrator(llm_service=AsyncMock())
    pfc.document_store = AsyncMock()
    pfc.reranker = MagicMock()
    pfc.conversation_memory = AsyncMock()
    pfc.llm_service = AsyncMock()
    pfc.llm_service.get_model_for_request = MagicMock(return_value=("test-model", {}))
    pfc.llm_service.generate_rag_response = AsyncMock(return_value="Plain answer.")

    pfc.document_store.recall_memories = AsyncMock(return_value=memories)
    pfc.reranker.rank_by_importance = MagicMock(return_value=memories)

    async def _run():
        return await pfc.reason_with_context(
            query="What is in the doc?",
            collection_name="raw_coll",
            use_llm=True,
        )

    asyncio.run(_run())

    call_kwargs = pfc.llm_service.generate_rag_response.await_args.kwargs
    assert call_kwargs.get("extra_system_instruction") is None