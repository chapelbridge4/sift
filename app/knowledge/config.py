"""Knowledge-profile loader. All tunables live in profiles/*.toml — never hardcoded."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomllib

_PROFILE_DIR = Path(__file__).parent / "profiles"


@dataclass(frozen=True)
class Tier0Cfg:
    max_clusters: int
    min_cluster_size: int
    claim_min_chars: int
    claim_max_chars: int
    max_sentences_per_claim: int


@dataclass(frozen=True)
class TierLLMCfg:
    max_output_tokens: int
    temperature: float
    max_claims_per_paper: int = 5
    max_spans_per_paper: int = 20
    max_sections_in_outline: int = 8
    section_preview_chars: int = 80


@dataclass(frozen=True)
class LLMCfg:
    backend: str
    model_path: str
    model_id: str
    model_hf_repo: str
    fallback_model_id: str
    max_retries: int = 2
    retry_backoff_base_seconds: float = 0.5


@dataclass(frozen=True)
class ParseCfg:
    extensions: list[str]


@dataclass(frozen=True)
class ChunkCfg:
    strategy: str
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class RetrievalCfg:
    topic_score_boost: float
    drill_down_top_k: int


@dataclass(frozen=True)
class KnowledgeProfile:
    name: str
    description: str
    tier0: Tier0Cfg
    tier1: TierLLMCfg
    tier2: TierLLMCfg
    llm: LLMCfg
    chunk: ChunkCfg
    retrieval: RetrievalCfg
    parse: ParseCfg


def load_profile(name: str) -> KnowledgeProfile:
    path = _PROFILE_DIR / f"knowledge_{name}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"unknown knowledge profile: {name!r} (looked in {path})")
    raw = tomllib.loads(path.read_text())
    tier1_raw = dict(raw["tier1"])
    tier2_raw = dict(raw["tier2"])
    llm_raw = dict(raw["llm"])
    parse_raw = raw.get("parse", {"extensions": ["pdf", "docx", "txt", "md", "html"]})

    return KnowledgeProfile(
        name=raw["profile"]["name"],
        description=raw["profile"]["description"],
        tier0=Tier0Cfg(**raw["tier0"]),
        tier1=TierLLMCfg(**tier1_raw),
        tier2=TierLLMCfg(**tier2_raw),
        llm=LLMCfg(**llm_raw),
        chunk=ChunkCfg(**raw["chunk"]),
        retrieval=RetrievalCfg(**raw["retrieval"]),
        parse=ParseCfg(**parse_raw),
    )