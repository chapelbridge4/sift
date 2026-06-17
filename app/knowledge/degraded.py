"""Degraded-mode fallbacks and hardware guard for the knowledge pipeline (spec §8)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from app.knowledge.models import Claim, ClaimSpan, PaperSummary, TopicSheet, TopicSource
from app.knowledge.tier1_extract import slugify
from app.knowledge.tier2_merge import TopicCluster, _contributing_papers


class KnowledgeHardwareError(Exception):
    """Raised when hardware_guard.sh fails before an LLM batch."""


_DEFAULT_GUARD_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "hardware_guard.sh"
)


def check_hardware_guard(script_path: Path | None = None) -> None:
    """Run hardware_guard.sh; raise KnowledgeHardwareError on non-zero exit."""
    script = Path(script_path) if script_path else _DEFAULT_GUARD_SCRIPT
    if not script.is_file():
        raise KnowledgeHardwareError(f"hardware guard script not found: {script}")

    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip() or "insufficient free memory"
        raise KnowledgeHardwareError(
            f"hardware guard failed (exit {result.returncode}): {detail}"
        )


def paper_summary_from_spans(parsed_doc: object, spans: list[ClaimSpan]) -> PaperSummary:
    """Tier 1 fallback: build a PaperSummary from Tier 0 claim spans only."""
    title = getattr(parsed_doc, "title", None) or getattr(parsed_doc, "paper_id", "unknown")
    authors = list(getattr(parsed_doc, "authors", None) or [])
    source_file = getattr(parsed_doc, "source_file", "")
    paper_id = getattr(parsed_doc, "paper_id", "unknown")

    claims = [Claim(text=s.text, section=s.section) for s in spans]
    topics = sorted({slugify(s.section) for s in spans if s.section})

    return PaperSummary(
        paper_id=paper_id,
        title=title,
        authors=authors,
        source_file=source_file,
        topics=topics,
        claims=claims,
        degraded=True,
    )


def degraded_topic_sheet(
    cluster: TopicCluster,
    paper_summaries: list[PaperSummary],
) -> TopicSheet:
    """Tier 2 fallback: raw concatenation of cluster spans and paper digests."""
    contributing = _contributing_papers(cluster, paper_summaries)
    links_to = sorted({p.paper_id for p in contributing})

    body_lines = [f"# {cluster.label}", "", "## Cluster spans", ""]
    for span in cluster.spans:
        body_lines.append(f"- [{span.paper_id}/{span.section}] {span.text}")
    body_lines.extend(["", "## Paper digests", ""])
    for paper in contributing:
        body_lines.append(f"### {paper.paper_id}: {paper.title}")
        for claim in paper.claims:
            section = f" ({claim.section})" if claim.section else ""
            body_lines.append(f"- {claim.text}{section}")
        body_lines.append("")

    slug = slugify(cluster.label)
    return TopicSheet(
        topic_id=f"cluster-{cluster.cluster_id}",
        slug=slug,
        title=cluster.label,
        body="\n".join(body_lines).strip() + "\n",
        links_to=links_to,
        sources=[TopicSource(paper_id=pid, section="") for pid in links_to],
        degraded=True,
    )