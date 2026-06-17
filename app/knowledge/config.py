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


@dataclass(frozen=True)
class TierLLMCfg:
    max_output_tokens: int
    temperature: float


@dataclass(frozen=True)
class LLMCfg:
    backend: str
    model_path: str
    model_id: str
    model_hf_repo: str
    fallback_model_id: str


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


def load_profile(name: str) -> KnowledgeProfile:
    path = _PROFILE_DIR / f"knowledge_{name}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"unknown knowledge profile: {name!r} (looked in {path})")
    raw = tomllib.loads(path.read_text())
    return KnowledgeProfile(
        name=raw["profile"]["name"],
        description=raw["profile"]["description"],
        tier0=Tier0Cfg(**raw["tier0"]),
        tier1=TierLLMCfg(**raw["tier1"]),
        tier2=TierLLMCfg(**raw["tier2"]),
        llm=LLMCfg(**raw["llm"]),
        chunk=ChunkCfg(**raw["chunk"]),
        retrieval=RetrievalCfg(**raw["retrieval"]),
    )