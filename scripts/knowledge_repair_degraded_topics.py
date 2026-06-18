#!/usr/bin/env python3
"""Rewrite degraded topic sheets with capped span/paper bodies (no LLM)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.knowledge.artifacts import read_artifact, write_artifact
from app.knowledge.config import load_profile
from app.knowledge.degraded import degraded_topic_sheet
from app.knowledge.models import ClaimSpan, PaperSummary, TopicSheet
from app.knowledge.tier2_merge import TopicCluster


def _load_papers(artifact_dir: Path) -> dict[str, PaperSummary]:
    papers: dict[str, PaperSummary] = {}
    for path in sorted((artifact_dir / "papers").glob("*.md")):
        artifact = read_artifact(path)
        if isinstance(artifact, PaperSummary):
            papers[artifact.paper_id] = artifact
    return papers


def _cluster_label(cluster_id: int, members: list[ClaimSpan]) -> str:
    if not members:
        return f"topic-{cluster_id}"
    from app.knowledge.tier1_extract import slugify

    words = slugify(members[0].text[:80]).replace("-", " ")
    return words or f"topic-{cluster_id}"


def repair(artifact_dir: Path, profile_name: str) -> int:
    profile = load_profile(profile_name)
    manifest_path = artifact_dir / "cluster_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clusters = {
        int(cluster_id): [ClaimSpan.model_validate(m) for m in members]
        for cluster_id, members in manifest["clusters"].items()
    }
    papers = _load_papers(artifact_dir)
    repaired = 0

    for path in sorted((artifact_dir / "topics").glob("*.md")):
        old = read_artifact(path)
        if not isinstance(old, TopicSheet) or not old.degraded:
            continue

        if not old.topic_id.startswith("cluster-"):
            continue
        cluster_id = int(old.topic_id.removeprefix("cluster-"))
        members = clusters.get(cluster_id, [])
        label = old.title or _cluster_label(cluster_id, members)
        cluster = TopicCluster(cluster_id=cluster_id, label=label, spans=members)
        sheet = degraded_topic_sheet(cluster, list(papers.values()), profile)
        sheet = sheet.model_copy(update={"slug": old.slug, "title": old.title})
        path.unlink()
        write_artifact(sheet, artifact_dir)
        repaired += 1

    return repaired


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair degraded topic artifact bodies")
    parser.add_argument(
        "--artifact-dir",
        default="data/corpus/.knowledge/ai_papers_knowledge",
    )
    parser.add_argument("--profile", default="papers")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = ROOT / artifact_dir

    count = repair(artifact_dir, args.profile)
    print(f"repaired {count} degraded topic sheets under {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())