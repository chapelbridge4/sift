"""KnowledgePipeline orchestrates Tier 0→2 artifact generation (DI, mockable)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from loguru import logger

from app.knowledge.artifacts import write_artifact
from app.knowledge.config import KnowledgeProfile
from app.knowledge.degraded import (
    check_hardware_guard,
    degraded_topic_sheet,
    paper_summary_from_spans,
)
from app.knowledge.models import (
    ClaimSpan,
    ClusterManifest,
    KnowledgeStats,
    PaperSummary,
    TopicSheet,
)
from app.knowledge.tier0_cluster import cluster_spans, extract_claim_spans
from app.knowledge.tier1_extract import extract_paper, slugify
from app.knowledge.tier2_merge import TopicCluster, merge_topic


@runtime_checkable
class KnowledgeParser(Protocol):
    async def parse_for_knowledge(self, file_paths: Sequence[str]) -> list[Any]: ...


class KnowledgePipeline:
    """Orchestrates parse → Tier 0 cluster → Tier 1 extract → Tier 2 merge → artifacts."""

    def __init__(
        self,
        parser: KnowledgeParser,
        embedder: Any,
        llm: Any,
        profile: KnowledgeProfile,
        output_dir: Path,
        *,
        skip_hardware_guard: bool = False,
        hardware_guard_script: Path | None = None,
    ) -> None:
        self._parser = parser
        self._embedder = embedder
        self._llm = llm
        self._profile = profile
        self._output_dir = Path(output_dir)
        self._skip_hardware_guard = skip_hardware_guard
        self._hardware_guard_script = hardware_guard_script

    async def run(self, file_paths: Sequence[str], collection_name: str) -> KnowledgeStats:
        correlation_id = uuid.uuid4().hex[:12]
        job_dir = self._output_dir / collection_name
        job_dir.mkdir(parents=True, exist_ok=True)

        log = logger.bind(correlation_id=correlation_id, collection=collection_name)
        log.info("knowledge pipeline start file_count={}", len(file_paths))

        parsed_docs = await self._parser.parse_for_knowledge(file_paths)
        all_spans: list[ClaimSpan] = []
        doc_spans: dict[str, list[ClaimSpan]] = {}

        for doc in parsed_docs:
            paper_id = getattr(doc, "paper_id", "unknown")
            spans = extract_claim_spans(doc, self._profile)
            doc_spans[paper_id] = spans
            all_spans.extend(spans)
            log.info("tier=0 paper_id={} span_count={}", paper_id, len(spans))

        vectors = await self._embed_vectors([s.text for s in all_spans])
        manifest = cluster_spans(all_spans, profile=self._profile, vectors=vectors)
        self._write_cluster_manifest(job_dir, manifest)

        if not self._skip_hardware_guard:
            check_hardware_guard(self._hardware_guard_script)

        paper_summaries: list[PaperSummary] = []
        for doc in parsed_docs:
            paper_id = getattr(doc, "paper_id", "unknown")
            spans = doc_spans.get(paper_id, [])
            try:
                summary = await extract_paper(doc, spans, self._llm, self._profile)
                paper_summaries.append(summary)
                log.info("tier=1 paper_id={} degraded={}", paper_id, summary.degraded)
            except Exception as exc:
                log.warning(
                    "tier=1 paper_id={} llm_failed error={} — using tier0 fallback",
                    paper_id,
                    type(exc).__name__,
                )
                paper_summaries.append(paper_summary_from_spans(doc, spans))

        topic_sheets: list[TopicSheet] = []
        if manifest.clusters:
            for cluster_id, members in manifest.clusters.items():
                label = _cluster_label(cluster_id, members)
                cluster = TopicCluster(cluster_id=cluster_id, label=label, spans=members)
                try:
                    sheet = await merge_topic(cluster, paper_summaries, self._llm, self._profile)
                    topic_sheets.append(sheet)
                    log.info(
                        "tier=2 cluster_id={} links={} degraded={}",
                        cluster_id,
                        len(sheet.links_to),
                        sheet.degraded,
                    )
                except Exception as exc:
                    log.warning(
                        "tier=2 cluster_id={} merge_failed error={} — degraded concat",
                        cluster_id,
                        type(exc).__name__,
                    )
                    topic_sheets.append(degraded_topic_sheet(cluster, paper_summaries))
        else:
            log.warning("tier=0 zero_clusters — paper summaries only (degraded Option B)")

        for summary in paper_summaries:
            write_artifact(summary, job_dir)
        for sheet in topic_sheets:
            write_artifact(sheet, job_dir)

        links = sum(len(t.links_to) for t in topic_sheets)
        stats = KnowledgeStats(
            topics=len(topic_sheets),
            papers=len(paper_summaries),
            chunks=0,
            links=links,
        )
        log.info(
            "knowledge pipeline complete topics={} papers={} links={}",
            stats.topics,
            stats.papers,
            stats.links,
        )
        return stats

    async def _embed_vectors(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        from inspect import iscoroutinefunction

        embed_texts = getattr(self._embedder, "embed_texts", None)
        if embed_texts is not None and callable(embed_texts) and not iscoroutinefunction(embed_texts):
            return embed_texts(texts)

        gen_dense = getattr(self._embedder, "generate_dense_embeddings", None)
        if gen_dense is not None and callable(gen_dense):
            result = gen_dense(texts)
            if hasattr(result, "__await__"):
                return await result
            return result

        raise TypeError("embedder must provide embed_texts or generate_dense_embeddings")

    def _write_cluster_manifest(self, job_dir: Path, manifest: ClusterManifest) -> None:
        payload = {
            "clusters": {
                str(cluster_id): [span.model_dump() for span in members]
                for cluster_id, members in manifest.clusters.items()
            }
        }
        path = job_dir / "cluster_manifest.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cluster_label(cluster_id: int, members: Sequence[ClaimSpan]) -> str:
    if not members:
        return f"topic-{cluster_id}"
    words = slugify(members[0].text[:80]).replace("-", " ")
    return words or f"topic-{cluster_id}"