"""Tier 2: per-topic-cluster LLM merge → TopicSheet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from loguru import logger
from pydantic import BaseModel, Field

from app.knowledge.backend import KnowledgeLLM
from app.knowledge.config import KnowledgeProfile
from app.knowledge.models import ClaimSpan, PaperSummary, TopicSheet, TopicSource
from app.knowledge.prompts import format_prompt, load_prompt
from app.knowledge.tier1_extract import slugify


@dataclass(frozen=True)
class TopicCluster:
    """Input bundle for Tier 2 merge."""

    cluster_id: int
    label: str
    spans: Sequence[ClaimSpan]


class TopicMergeOutput(BaseModel):
    title: str
    slug: str
    body: str
    sources: list[TopicSource] = Field(default_factory=list)


def contributing_papers(
    cluster: TopicCluster, paper_summaries: Sequence[PaperSummary]
) -> list[PaperSummary]:
    paper_ids = {s.paper_id for s in cluster.spans}
    return [p for p in paper_summaries if p.paper_id in paper_ids]


def _truncate_field(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def _format_paper_summaries(
    summaries: Sequence[PaperSummary],
    *,
    max_claims_per_paper: int,
    max_papers: int,
    field_preview_chars: int,
) -> str:
    blocks: list[str] = []
    for paper in summaries[:max_papers]:
        claims = "; ".join(c.text for c in paper.claims[:max_claims_per_paper])
        blocks.append(
            f"### {paper.paper_id}: {paper.title}\n"
            f"Topics: {', '.join(paper.topics)}\n"
            f"Claims: {claims}\n"
            f"Methods: {_truncate_field(paper.methods, field_preview_chars)}\n"
            f"Results: {_truncate_field(paper.results, field_preview_chars)}\n"
            f"Limitations: {_truncate_field(paper.limitations, field_preview_chars)}"
        )
    return "\n\n".join(blocks) if blocks else "(no papers)"


def _format_conflicting_spans(
    spans: Sequence[ClaimSpan], *, max_spans: int
) -> str:
    if not spans:
        return "(none flagged)"
    selected = list(spans[:max_spans])
    return "\n".join(f"- [{s.paper_id}/{s.section}] {s.text}" for s in selected)


async def merge_topic(
    cluster: TopicCluster,
    paper_summaries: Sequence[PaperSummary],
    llm: KnowledgeLLM,
    profile: KnowledgeProfile,
) -> TopicSheet:
    """One LLM call per cluster; links_to populated from contributing paper_ids."""
    contract = load_prompt("topic_merge")
    contributing = contributing_papers(cluster, paper_summaries)
    links_to = sorted({p.paper_id for p in contributing})

    prompt = format_prompt(
        contract,
        cluster_id=cluster.cluster_id,
        cluster_label=cluster.label,
        paper_summaries=_format_paper_summaries(
            contributing,
            max_claims_per_paper=profile.tier2.max_claims_per_paper,
            max_papers=profile.tier2.max_papers_in_merge,
            field_preview_chars=profile.tier2.field_preview_chars,
        ),
        conflicting_spans=_format_conflicting_spans(
            cluster.spans,
            max_spans=profile.tier2.max_spans_in_merge,
        ),
        max_output_tokens=profile.tier2.max_output_tokens,
    )

    logger.info(
        "tier2 merge cluster_id={} paper_count={}",
        cluster.cluster_id,
        len(contributing),
    )

    merged = await llm.extract(
        prompt,
        TopicMergeOutput,
        max_tokens=profile.tier2.max_output_tokens,
        temperature=profile.tier2.temperature,
        json_schema=contract.output_schema,
    )

    slug = merged.slug or slugify(merged.title)
    sources = merged.sources or [
        TopicSource(paper_id=pid, section="") for pid in links_to
    ]

    return TopicSheet(
        topic_id=f"cluster-{cluster.cluster_id}",
        slug=slug,
        title=merged.title,
        body=merged.body,
        links_to=links_to,
        sources=sources,
    )