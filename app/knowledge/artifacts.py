"""Serialize/deserialize knowledge artifacts as YAML-frontmatter markdown files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.knowledge.models import (
    Claim,
    KnowledgeArtifact,
    PaperSummary,
    TopicSheet,
    TopicSource,
)


def artifact_body_text(artifact: KnowledgeArtifact) -> str:
    """Return markdown body text for chunking (excludes YAML frontmatter)."""
    if isinstance(artifact, PaperSummary):
        return _paper_summary_body(artifact)
    if isinstance(artifact, TopicSheet):
        return artifact.body
    raise TypeError(f"unsupported artifact type: {type(artifact)!r}")


def write_artifact(artifact: KnowledgeArtifact, output_dir: Path) -> Path:
    """Write a knowledge artifact to markdown with YAML frontmatter."""
    output_dir = Path(output_dir)
    if isinstance(artifact, PaperSummary):
        subdir = output_dir / "papers"
        filename = f"{artifact.paper_id}.md"
        body = _paper_summary_body(artifact)
        frontmatter = _paper_summary_frontmatter(artifact)
    elif isinstance(artifact, TopicSheet):
        subdir = output_dir / "topics"
        filename = f"{artifact.slug}.md"
        body = artifact.body
        frontmatter = _topic_sheet_frontmatter(artifact)
    else:
        raise TypeError(f"unsupported artifact type: {type(artifact)!r}")

    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / filename
    path.write_text(_render_document(frontmatter, body), encoding="utf-8")
    return path


def read_artifact(path: Path) -> KnowledgeArtifact:
    """Read a knowledge artifact from markdown with YAML frontmatter."""
    text = Path(path).read_text(encoding="utf-8")
    frontmatter, body = _parse_document(text)
    doc_type = frontmatter.get("doc_type")
    if doc_type == "paper_summary":
        return _parse_paper_summary(frontmatter)
    if doc_type == "topic":
        return _parse_topic_sheet(frontmatter, body)
    raise ValueError(f"unknown doc_type: {doc_type!r} in {path}")


def _paper_summary_frontmatter(artifact: PaperSummary) -> dict[str, Any]:
    data: dict[str, Any] = {
        "doc_type": artifact.doc_type,
        "paper_id": artifact.paper_id,
        "title": artifact.title,
        "authors": artifact.authors,
        "source_file": artifact.source_file,
        "topics": artifact.topics,
        "claims": [{"text": c.text, "section": c.section} for c in artifact.claims],
        "methods": artifact.methods,
        "results": artifact.results,
        "limitations": artifact.limitations,
    }
    if artifact.degraded:
        data["degraded"] = True
    return data


def _topic_sheet_frontmatter(artifact: TopicSheet) -> dict[str, Any]:
    data: dict[str, Any] = {
        "doc_type": artifact.doc_type,
        "topic_id": artifact.topic_id,
        "slug": artifact.slug,
        "title": artifact.title,
        "links_to": artifact.links_to,
        "sources": [
            {"paper_id": s.paper_id, "section": s.section} for s in artifact.sources
        ],
    }
    if artifact.degraded:
        data["degraded"] = True
    return data


def _paper_summary_body(artifact: PaperSummary) -> str:
    lines = [f"# {artifact.title}", ""]
    if artifact.claims:
        lines.extend(["## Key claims", ""])
        for claim in artifact.claims:
            section = f" ({claim.section})" if claim.section else ""
            lines.append(f"- {claim.text}{section}")
        lines.append("")
    if artifact.methods:
        lines.extend(["## Methods", "", artifact.methods, ""])
    if artifact.results:
        lines.extend(["## Results", "", artifact.results, ""])
    if artifact.limitations:
        lines.extend(["## Limitations", "", artifact.limitations, ""])
    return "\n".join(lines).rstrip() + "\n"


def _render_document(frontmatter: dict[str, Any], body: str) -> str:
    yaml_block = yaml.safe_dump(
        frontmatter,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    ).rstrip()
    body = body.rstrip() + "\n" if body.strip() else ""
    return f"---\n{yaml_block}\n---\n{body}"


def _parse_document(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        raise ValueError("artifact must start with YAML frontmatter delimiter '---'")
    rest = text[len("---") :]
    if not rest.startswith("\n"):
        raise ValueError("invalid frontmatter: expected newline after opening ---")
    rest = rest[1:]
    end_idx = rest.find("\n---")
    if end_idx == -1:
        raise ValueError("invalid frontmatter: missing closing ---")
    yaml_text = rest[:end_idx]
    body = rest[end_idx + len("\n---") :].lstrip("\n")
    frontmatter = yaml.safe_load(yaml_text)
    if not isinstance(frontmatter, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return frontmatter, body


def _parse_paper_summary(frontmatter: dict[str, Any]) -> PaperSummary:
    claims_raw = frontmatter.get("claims", []) or []
    claims = [Claim(**c) for c in claims_raw]
    return PaperSummary(
        doc_type=frontmatter.get("doc_type", "paper_summary"),
        paper_id=frontmatter["paper_id"],
        title=frontmatter["title"],
        authors=frontmatter.get("authors", []) or [],
        source_file=frontmatter["source_file"],
        topics=frontmatter.get("topics", []) or [],
        claims=claims,
        methods=frontmatter.get("methods", "") or "",
        results=frontmatter.get("results", "") or "",
        limitations=frontmatter.get("limitations", "") or "",
        degraded=bool(frontmatter.get("degraded", False)),
    )


def _parse_topic_sheet(frontmatter: dict[str, Any], body: str) -> TopicSheet:
    sources_raw = frontmatter.get("sources", []) or []
    sources = [TopicSource(**s) for s in sources_raw]
    return TopicSheet(
        doc_type=frontmatter.get("doc_type", "topic"),
        topic_id=frontmatter["topic_id"],
        slug=frontmatter["slug"],
        title=frontmatter["title"],
        body=body.rstrip(),
        links_to=frontmatter.get("links_to", []) or [],
        sources=sources,
        degraded=bool(frontmatter.get("degraded", False)),
    )