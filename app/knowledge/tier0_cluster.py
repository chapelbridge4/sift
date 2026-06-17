"""Tier 0: claim-span extraction and agglomerative clustering (no LLM)."""

from __future__ import annotations

import hashlib
from typing import Any, Protocol, Sequence

from app.knowledge.config import KnowledgeProfile
from app.knowledge.models import ClaimSpan, ClusterManifest
from app.services.text_splitter import split_sentences

# Sections processed first when present in parsed metadata.
_PRIORITY_SECTION_NAMES = frozenset({"abstract", "conclusion", "summary"})

_MAX_SENTENCES_PER_CLAIM = 3


class ParsedSection(Protocol):
    name: str
    text: str


class ParsedDoc(Protocol):
    paper_id: str
    source_file: str
    sections: Sequence[ParsedSection]


def _section_name(section: Any) -> str:
    if isinstance(section, dict):
        return str(section["name"])
    return str(section.name)


def _section_text(section: Any) -> str:
    if isinstance(section, dict):
        return str(section["text"])
    return str(section.text)


def _section_priority(name: str) -> tuple[int, str]:
    normalized = name.strip().lower()
    if normalized in _PRIORITY_SECTION_NAMES:
        return (0, normalized)
    return (1, normalized)


def _embedding_id(paper_id: str, section: str, text: str) -> str:
    digest = hashlib.sha256(f"{paper_id}:{section}:{text}".encode()).hexdigest()
    return digest[:16]


def _group_sentences(
    sentences: list[str],
    *,
    min_chars: int,
    max_chars: int,
    max_sentences: int = _MAX_SENTENCES_PER_CLAIM,
) -> list[str]:
    """Group 1–*max_sentences* consecutive sentences into claim-sized spans."""
    spans: list[str] = []
    index = 0
    while index < len(sentences):
        best: list[str] = []
        for size in range(1, min(max_sentences, len(sentences) - index) + 1):
            candidate = " ".join(sentences[index : index + size])
            if len(candidate) <= max_chars:
                best = sentences[index : index + size]
            else:
                break

        if not best:
            index += 1
            continue

        text = " ".join(best)
        if len(text) >= min_chars:
            spans.append(text)
        index += len(best)

    return spans


def extract_claim_spans(parsed_doc: ParsedDoc, profile: KnowledgeProfile) -> list[ClaimSpan]:
    """Extract 1–3 sentence claim candidates from parsed section text."""
    tier0 = profile.tier0
    ordered = sorted(parsed_doc.sections, key=lambda s: _section_priority(_section_name(s)))

    spans: list[ClaimSpan] = []
    for section in ordered:
        section_name = _section_name(section)
        for text in _group_sentences(
            split_sentences(_section_text(section)),
            min_chars=tier0.claim_min_chars,
            max_chars=tier0.claim_max_chars,
        ):
            spans.append(
                ClaimSpan(
                    paper_id=parsed_doc.paper_id,
                    text=text,
                    section=section_name,
                    embedding_id=_embedding_id(parsed_doc.paper_id, section_name, text),
                )
            )

    return spans


def cluster_spans(
    spans: list[ClaimSpan],
    *,
    embedder: Any | None = None,
    profile: KnowledgeProfile,
    vectors: list[list[float]] | None = None,
) -> ClusterManifest:
    """Embed claim spans and cluster them with agglomerative (cosine) grouping."""
    if not spans:
        return ClusterManifest()

    tier0 = profile.tier0
    if len(spans) < tier0.min_cluster_size:
        return ClusterManifest()

    from sklearn.cluster import AgglomerativeClustering

    if vectors is None:
        if embedder is None:
            raise ValueError("cluster_spans requires embedder or precomputed vectors")
        vectors = embedder.embed_texts([s.text for s in spans])
    n_clusters = min(
        tier0.max_clusters,
        len(spans) // tier0.min_cluster_size,
    )
    if n_clusters < 1:
        return ClusterManifest()

    labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    ).fit_predict(vectors)

    clusters: dict[int, list[ClaimSpan]] = {}
    for span, label in zip(spans, labels, strict=True):
        clusters.setdefault(int(label), []).append(span)

    return ClusterManifest(
        clusters={
            cluster_id: members
            for cluster_id, members in clusters.items()
            if len(members) >= tier0.min_cluster_size
        }
    )