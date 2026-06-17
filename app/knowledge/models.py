"""Domain models for the make_knowledge corpus-intelligence pipeline."""

from __future__ import annotations

from typing import List, Union

from pydantic import BaseModel, Field, field_validator


class ClaimSpan(BaseModel):
    """A 1–3 sentence candidate claim extracted from a paper (Tier 0)."""

    paper_id: str
    text: str = Field(min_length=1)
    section: str
    embedding_id: str

    @field_validator("text")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("claim text must be non-empty")
        return v


class Claim(BaseModel):
    text: str = Field(min_length=1)
    section: str = ""


class PaperSummary(BaseModel):
    """Slim per-paper digest (Tier 1 output)."""

    doc_type: str = "paper_summary"
    paper_id: str
    title: str
    authors: List[str] = []
    source_file: str
    topics: List[str] = []
    claims: List[Claim] = []
    methods: str = ""
    results: str = ""
    limitations: str = ""
    degraded: bool = False


class TopicSource(BaseModel):
    paper_id: str
    section: str = ""


class TopicSheet(BaseModel):
    """Merged cross-paper concept (Tier 2 output)."""

    doc_type: str = "topic"
    topic_id: str
    slug: str
    title: str
    body: str = ""
    links_to: List[str] = []
    sources: List[TopicSource] = []
    degraded: bool = False


class KnowledgeStats(BaseModel):
    topics: int = 0
    papers: int = 0
    chunks: int = 0
    links: int = 0


class ClusterManifest(BaseModel):
    """Tier 0 output: cluster_id → member claim spans."""

    clusters: dict[int, List[ClaimSpan]] = Field(default_factory=dict)


KnowledgeArtifact = Union[PaperSummary, TopicSheet]