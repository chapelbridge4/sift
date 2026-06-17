"""YAML prompt contract loader for Tier 1/2 LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_PROMPT_DIR = Path(__file__).parent


@dataclass(frozen=True)
class PromptContract:
    name: str
    description: str
    system: str
    template: str
    output_schema: dict[str, Any]


def load_prompt(name: str) -> PromptContract:
    """Load a prompt contract by name (e.g. 'paper_extract')."""
    path = _PROMPT_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"unknown knowledge prompt: {name!r} (looked in {path})")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PromptContract(
        name=raw["name"],
        description=raw.get("description", ""),
        system=raw["system"].strip(),
        template=raw["template"].strip(),
        output_schema=raw.get("output_schema", {}),
    )


def format_prompt(contract: PromptContract, **kwargs: Any) -> str:
    """Format the prompt template with the given keyword arguments."""
    return contract.template.format(**kwargs)