import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.knowledge.index import collection_has_knowledge_marker
from app.knowledge.models import KnowledgeStats
from app.pipeline.document_store import DocumentStore


def test_collection_has_knowledge_marker_detects_payload():
    assert collection_has_knowledge_marker([{"knowledge_built": True}])
    assert not collection_has_knowledge_marker([{"knowledge_built": False}])
    assert not collection_has_knowledge_marker([{}])


def test_form_memories_make_knowledge_indexes_artifacts_only():
    store = DocumentStore()
    store.qdrant_service = AsyncMock()
    store.qdrant_service.upsert_documents = AsyncMock(return_value=5)

    mock_stats = KnowledgeStats(topics=2, papers=3, chunks=5, links=4)
    mock_pipeline = AsyncMock()
    mock_pipeline.run = AsyncMock(return_value=mock_stats)

    with patch("app.pipeline.document_store.index_artifacts", new_callable=AsyncMock) as mock_index:
        mock_index.return_value = 5

        async def _run():
            return await store.form_memories(
                collection_name="knowledge_coll",
                file_paths=["papers/a.pdf"],
                make_knowledge=True,
                knowledge_profile="papers",
                knowledge_pipeline=mock_pipeline,
            )

        result = asyncio.run(_run())

    mock_pipeline.run.assert_awaited_once()
    mock_index.assert_awaited_once()
    store.qdrant_service.upsert_documents.assert_not_awaited()

    assert result["success"] is True
    assert result["knowledge"]["topics"] == 2
    assert result["knowledge"]["papers"] == 3
    assert result["knowledge"]["chunks"] == 5
    assert result["knowledge"]["links"] == 4
    assert result["knowledge_built"] is True


def test_form_memories_default_path_unchanged():
    store = DocumentStore()
    store.qdrant_service = AsyncMock()
    store.qdrant_service.upsert_documents = AsyncMock(return_value=10)
    store.document_parser = AsyncMock()

    chunk = MagicMock()
    chunk.content = "raw chunk"
    chunk.metadata = {"source_file": "doc.pdf"}
    store.document_parser.parse_files = AsyncMock(return_value=[chunk])

    async def _run():
        return await store.form_memories(
            collection_name="raw_coll",
            file_paths=["doc.pdf"],
            make_knowledge=False,
        )

    result = asyncio.run(_run())

    store.document_parser.parse_files.assert_awaited_once()
    store.qdrant_service.upsert_documents.assert_awaited_once()
    assert result["success"] is True
    assert result["total_chunks"] == 10
    assert "knowledge" not in result