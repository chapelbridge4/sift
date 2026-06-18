"""Tier 1: per-paper LLM structured extract → PaperSummary."""

from __future__ import annotations

import re
from typing import Sequence

from loguru import logger
from pydantic import BaseModel, Field

from app.knowledge.backend import KnowledgeLLM
from app.knowledge.config import KnowledgeProfile
from app.knowledge.models import Claim, ClaimSpan, PaperSummary
from app.knowledge.prompts import format_prompt, load_prompt
from app.knowledge.tier0_cluster import ParsedDoc


class PaperExtractOutput(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    methods: str = ""
    results: str = ""
    limitations: str = ""
    topic_tags: list[str] = Field(default_factory=list)


def _section_outline(parsed_doc: ParsedDoc, profile: KnowledgeProfile) -> str:
    tier1 = profile.tier1
    lines: list[str] = []
    for section in parsed_doc.sections[: tier1.max_sections_in_outline]:
        name = section["name"] if isinstance(section, dict) else section.name
        text = section["text"] if isinstance(section, dict) else section.text
        preview = " ".join(text.split())[: tier1.section_preview_chars]
        lines.append(f"- {name}: {preview}...")
    return "\n".join(lines) if lines else "(no sections)"


def _select_spans_for_prompt(
    spans: Sequence[ClaimSpan], profile: KnowledgeProfile
) -> list[ClaimSpan]:
    """Cap spans so Tier 1 prompt fits n_ctx (spans are section-priority ordered)."""
    return list(spans[: profile.tier1.max_spans_per_paper])


def _format_claim_spans(spans: Sequence[ClaimSpan]) -> str:
    if not spans:
        return "(no claim spans)"
    return "\n".join(f"- [{s.section}] {s.text}" for s in spans)


async def extract_paper(
    parsed_doc: ParsedDoc,
    top_spans: Sequence[ClaimSpan],
    llm: KnowledgeLLM,
    profile: KnowledgeProfile,
) -> PaperSummary:
    """One LLM call per paper; returns a validated PaperSummary."""
    contract = load_prompt("paper_extract")
    title = getattr(parsed_doc, "title", None) or parsed_doc.paper_id
    authors = getattr(parsed_doc, "authors", None) or []

    selected_spans = _select_spans_for_prompt(top_spans, profile)
    prompt = format_prompt(
        contract,
        paper_id=parsed_doc.paper_id,
        title=title,
        authors=", ".join(authors) if authors else "(unknown)",
        source_file=parsed_doc.source_file,
        section_outline=_section_outline(parsed_doc, profile),
        claim_spans=_format_claim_spans(selected_spans),
        max_output_tokens=profile.tier1.max_output_tokens,
    )

    logger.info(
        "tier1 extract paper_id={} span_count={} prompt_spans={}",
        parsed_doc.paper_id,
        len(top_spans),
        len(selected_spans),
    )

    extracted = await llm.extract(
        prompt,
        PaperExtractOutput,
        max_tokens=profile.tier1.max_output_tokens,
        temperature=profile.tier1.temperature,
        json_schema=contract.output_schema,
    )

    return PaperSummary(
        paper_id=parsed_doc.paper_id,
        title=extracted.title,
        authors=extracted.authors,
        source_file=parsed_doc.source_file,
        topics=extracted.topic_tags,
        claims=extracted.claims,
        methods=extracted.methods,
        results=extracted.results,
        limitations=extracted.limitations,
    )


def slugify(text: str) -> str:
    """Derive a URL-safe slug from a title or label."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "topic"