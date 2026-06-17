"""Tier 3: chunk canonical knowledge artifacts and index into Qdrant."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from app.knowledge.artifacts import artifact_body_text, read_artifact
from app.knowledge.config import KnowledgeProfile
from app.knowledge.models import PaperSummary, TopicSheet
from app.services.text_splitter import MarkdownChunker


async def index_artifacts(
    *,
    collection_name: str,
    artifact_dir: Path,
    profile: KnowledgeProfile,
    qdrant_service: Any,
    batch_size: int = 32,
) -> int:
    """Chunk markdown artifacts and upsert with knowledge metadata (no raw PDF chunks)."""
    artifact_dir = Path(artifact_dir)
    paths = sorted(artifact_dir.glob("papers/*.md")) + sorted(artifact_dir.glob("topics/*.md"))

    if not paths:
        logger.warning("index_artifacts: no artifacts under {}", artifact_dir)
        return 0

    chunker = MarkdownChunker(
        chunk_size=profile.chunk.chunk_size,
        chunk_overlap=profile.chunk.chunk_overlap,
    )

    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for path in paths:
        artifact = read_artifact(path)
        doc_type = artifact.doc_type
        if isinstance(artifact, PaperSummary):
            doc_id = artifact.paper_id
            links_to: list[str] = []
            source_file = artifact.source_file
        elif isinstance(artifact, TopicSheet):
            doc_id = artifact.topic_id
            links_to = list(artifact.links_to)
            source_file = str(path.relative_to(artifact_dir))
        else:
            continue

        chunks = chunker.chunk(artifact_body_text(artifact))

        for chunk in chunks:
            meta = {
                "doc_type": doc_type,
                "doc_id": doc_id,
                "source_file": source_file,
                "links_to": links_to,
                "heading_path": chunk.metadata.get("heading_path", []),
                "chunk_index": chunk.metadata.get("position", 0),
                "knowledge_profile": profile.name,
                "knowledge_built": True,
                "degraded": artifact.degraded,
            }
            texts.append(chunk.content)
            metadatas.append(meta)

    total = await qdrant_service.upsert_documents(
        collection_name=collection_name,
        texts=texts,
        metadatas=metadatas,
        batch_size=batch_size,
    )
    logger.info(
        "index_artifacts collection={} artifacts={} chunks={}",
        collection_name,
        len(paths),
        total,
    )
    return total


def collection_has_knowledge_marker(payloads: list[dict[str, Any]]) -> bool:
    """Return True if any chunk payload carries knowledge_built=true."""
    return any(p.get("knowledge_built") for p in payloads)