"""
Tuning profiles for MLX optimization experiments.

Reads all model IDs, paths, and thresholds from app/config.py — no hardcoding.
"""

from app.config import get_settings
from typing import Dict, Any


class TuningProfile:
    """Configuration for a benchmark experiment run."""

    def __init__(
        self,
        name: str,
        model_profile: str,
        kv_cache_quantization: str | None = None,
        speculative_decoding: bool = False,
        prefix_cache: bool = False,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ):
        self.name = name
        self.model_profile = model_profile
        self.kv_cache_quantization = kv_cache_quantization
        self.speculative_decoding = speculative_decoding
        self.prefix_cache = prefix_cache
        self.max_tokens = max_tokens
        self.temperature = temperature

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model_profile": self.model_profile,
            "kv_cache_quantization": self.kv_cache_quantization,
            "speculative_decoding": self.speculative_decoding,
            "prefix_cache": self.prefix_cache,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


def get_baseline_profile(profile: str = "fast") -> TuningProfile:
    """Baseline — all optimizations disabled."""
    settings = get_settings()
    profile_cfg = settings.MODEL_PROFILES.get(profile, settings.MODEL_PROFILES["fast"])

    return TuningProfile(
        name=f"{profile}_baseline",
        model_profile=profile,
        kv_cache_quantization=None,
        speculative_decoding=False,
        prefix_cache=False,
        max_tokens=profile_cfg.get("max_tokens", 400),
        temperature=profile_cfg.get("temperature", 0.7),
    )


def get_kv_cache_q8_profile(profile: str = "fast") -> TuningProfile:
    return TuningProfile(
        name=f"{profile}_kv_q8",
        model_profile=profile,
        kv_cache_quantization="q8",
        speculative_decoding=False,
        prefix_cache=False,
    )


def get_kv_cache_q4_profile(profile: str = "fast") -> TuningProfile:
    return TuningProfile(
        name=f"{profile}_kv_q4",
        model_profile=profile,
        kv_cache_quantization="q4",
        speculative_decoding=False,
        prefix_cache=False,
    )


def get_speculative_profile(profile: str = "balanced") -> TuningProfile:
    """Speculative decoding only for balanced/quality (4B models)."""
    return TuningProfile(
        name=f"{profile}_speculative",
        model_profile=profile,
        kv_cache_quantization=None,
        speculative_decoding=True,
        prefix_cache=False,
    )


def get_prefix_cache_profile(profile: str = "fast") -> TuningProfile:
    return TuningProfile(
        name=f"{profile}_prefix_cache",
        model_profile=profile,
        kv_cache_quantization=None,
        speculative_decoding=False,
        prefix_cache=True,
    )