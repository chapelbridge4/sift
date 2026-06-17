"""Knowledge-collection retrieval helpers (topic boost, drill-down, citations)."""

from __future__ import annotations

from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from app.knowledge.index import collection_has_knowledge_marker

KNOWLEDGE_CITATION_INSTRUCTION = (
    "When citing sources from the context, use [paper:{paper_id}] for "
    "paper-level claims and [topic:{slug}] for merged topic concepts."
)


def is_knowledge_collection(memories: list[dict[str, Any]]) -> bool:
    """Return True if recalled chunks carry the knowledge_built marker."""
    payloads = [m.get("metadata", {}) for m in memories]
    return collection_has_knowledge_marker(payloads)


def apply_topic_score_boost(
    memories: list[dict[str, Any]],
    boost: float,
) -> list[dict[str, Any]]:
    """Post-retrieval multiplier for doc_type=topic chunks; re-rank by score."""
    if boost == 1.0:
        return memories

    boosted: list[dict[str, Any]] = []
    for mem in memories:
        updated = dict(mem)
        meta = dict(updated.get("metadata", {}))
        if meta.get("doc_type") == "topic":
            updated["score"] = updated.get("score", 0.0) * boost
            meta["topic_boost_applied"] = True
            updated["metadata"] = meta
        boosted.append(updated)

    boosted.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return boosted


def extract_drill_down_paper_ids(
    memories: list[dict[str, Any]],
    top_k: int,
) -> list[str]:
    """Collect paper IDs from links_to on the top topic chunks."""
    paper_ids: list[str] = []
    seen: set[str] = set()
    topic_count = 0

    for mem in memories:
        meta = mem.get("metadata", {})
        if meta.get("doc_type") != "topic":
            continue
        if topic_count >= top_k:
            break
        topic_count += 1
        for paper_id in meta.get("links_to", []):
            if paper_id not in seen:
                seen.add(paper_id)
                paper_ids.append(paper_id)

    return paper_ids


def build_paper_drill_down_filter(paper_ids: list[str]) -> Filter:
    """Qdrant filter: doc_type=paper_summary AND doc_id IN paper_ids."""
    return Filter(
        must=[
            FieldCondition(key="doc_type", match=MatchValue(value="paper_summary")),
            FieldCondition(key="doc_id", match=MatchAny(any=paper_ids)),
        ]
    )


def merge_drill_down_memories(
    topic_layer: list[dict[str, Any]],
    paper_layer: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge topic hits first, then linked paper summaries; dedupe by point id."""
    merged: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()

    for mem in topic_layer:
        point_id = mem.get("id")
        if point_id is not None and point_id in seen_ids:
            continue
        if point_id is not None:
            seen_ids.add(point_id)
        merged.append(mem)

    for mem in paper_layer:
        point_id = mem.get("id")
        if point_id is not None and point_id in seen_ids:
            continue
        if point_id is not None:
            seen_ids.add(point_id)
        merged.append(mem)

    return merged


def build_retrieval_sources(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build sources[] metadata with layer (topic | paper) per chunk."""
    sources: list[dict[str, Any]] = []
    for mem in memories:
        meta = mem.get("metadata", {})
        doc_type = meta.get("doc_type")
        if doc_type == "topic":
            layer = "topic"
            source_id = meta.get("doc_id") or meta.get("topic_id")
        elif doc_type == "paper_summary":
            layer = "paper"
            source_id = meta.get("doc_id") or meta.get("paper_id")
        else:
            continue

        sources.append(
            {
                "layer": layer,
                "doc_type": doc_type,
                "doc_id": source_id,
                "score": mem.get("score", 0.0),
                "source_file": meta.get("source_file"),
            }
        )

    return sources


def infer_retrieval_layers(memories: list[dict[str, Any]]) -> list[str]:
    """Return ordered retrieval layer names present in memories."""
    layers: list[str] = []
    doc_types = {m.get("metadata", {}).get("doc_type") for m in memories}
    if "topic" in doc_types:
        layers.append("topic")
    if "paper_summary" in doc_types:
        layers.append("paper")
    return layers