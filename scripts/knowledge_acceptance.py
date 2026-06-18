#!/usr/bin/env python3
"""make_knowledge acceptance: optional build + keyword-recall eval + artifact stats.

Records chunk count, recall@k proxy, paper/topic counts for BENCHMARKS.md.

Usage:
    ./scripts/hardware_guard.sh
    .venv/bin/python scripts/knowledge_acceptance.py --build \\
        --input papers/ --collection ai_papers_knowledge --profile papers

    # Eval only (collection already built):
    .venv/bin/python scripts/knowledge_acceptance.py \\
        --collection ai_papers_knowledge --probes data/evaluation/papers_probes.json

    .venv/bin/python scripts/knowledge_acceptance.py --build ... --output reports/knowledge/acceptance.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.kbforge.eval.keyword_recall import compute_keyword_recall  # noqa: E402
from app.kbforge.eval.probe_evaluator import load_probes  # noqa: E402
from app.knowledge.artifacts import read_artifact  # noqa: E402
from app.knowledge.backend import build_knowledge_llm  # noqa: E402
from app.knowledge.cli import scan_input_files  # noqa: E402
from app.knowledge.config import load_profile  # noqa: E402
from app.knowledge.degraded import check_hardware_guard  # noqa: E402
from app.knowledge.index import index_artifacts  # noqa: E402
from app.knowledge.pipeline import KnowledgePipeline  # noqa: E402
from app.services.document_parser import DocumentParser  # noqa: E402
from app.services.embeddings import EmbeddingService  # noqa: E402
from app.services.qdrant_service import QdrantService  # noqa: E402

_DEFAULT_GUARD = ROOT / "scripts" / "hardware_guard.sh"


def _artifact_stats(artifact_dir: Path) -> dict:
    papers_dir = artifact_dir / "papers"
    topics_dir = artifact_dir / "topics"
    paper_paths = sorted(papers_dir.glob("*.md")) if papers_dir.is_dir() else []
    topic_paths = sorted(topics_dir.glob("*.md")) if topics_dir.is_dir() else []

    links = 0
    topics_with_links = 0
    for path in topic_paths:
        sheet = read_artifact(path)
        if sheet.links_to:
            topics_with_links += 1
            links += len(sheet.links_to)

    return {
        "papers": len(paper_paths),
        "topics": len(topic_paths),
        "links": links,
        "topics_with_links": topics_with_links,
    }


async def _maybe_build(args: argparse.Namespace, profile) -> Path:
    settings = get_settings()
    if settings.KNOWLEDGE_OUTPUT_DIR:
        output_root = Path(settings.KNOWLEDGE_OUTPUT_DIR).expanduser()
    else:
        output_root = Path(settings.ALLOWED_CORPUS_DIR) / ".knowledge"
    if args.output_dir:
        output_root = Path(args.output_dir).expanduser()

    artifact_dir = output_root / args.collection
    input_dir = Path(args.input).expanduser()
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir

    file_paths = scan_input_files(input_dir, profile)
    if not file_paths:
        raise FileNotFoundError(f"no ingestible files under {input_dir}")

    pipeline = KnowledgePipeline(
        parser=DocumentParser(),
        embedder=EmbeddingService(),
        llm=build_knowledge_llm(profile),
        profile=profile,
        output_dir=output_root,
        skip_hardware_guard=args.skip_hardware_guard,
    )
    print(f"Building knowledge: {len(file_paths)} files → {args.collection}")
    t0 = time.perf_counter()
    stats = await pipeline.run(file_paths, args.collection)
    build_s = round(time.perf_counter() - t0, 1)
    print(f"  pipeline: papers={stats.papers} topics={stats.topics} links={stats.links} ({build_s}s)")

    qdrant = QdrantService()
    await qdrant.initialize()
    indexed = await index_artifacts(
        collection_name=args.collection,
        artifact_dir=artifact_dir,
        profile=profile,
        qdrant_service=qdrant,
    )
    print(f"  indexed chunks={indexed}")
    return artifact_dir


async def _ensure_indexed(
    collection_name: str,
    artifact_dir: Path,
    profile,
) -> int:
    """Index artifacts when collection is missing or empty (eval-only path)."""
    qdrant = QdrantService()
    await qdrant.initialize()
    try:
        info = await qdrant.get_collection_info(collection_name)
        points = info.get("points_count") or info.get("vectors_count") or 0
    except Exception:
        points = 0

    if points > 0:
        print(f"  collection {collection_name} already has {points} points — skip index")
        return points

    print(f"  indexing artifacts from {artifact_dir} → {collection_name}")
    return await index_artifacts(
        collection_name=collection_name,
        artifact_dir=artifact_dir,
        profile=profile,
        qdrant_service=qdrant,
    )


async def _eval_collection(
    collection_name: str,
    probes_path: Path,
    top_k: int,
) -> dict:
    probes = load_probes(probes_path)
    qdrant = QdrantService()
    await qdrant.initialize()

    info = await qdrant.get_collection_info(collection_name)
    chunk_count = info.get("points_count") or info.get("vectors_count") or 0

    per_query: list[dict] = []
    recalls: list[float] = []

    for probe in probes:
        t0 = time.perf_counter()
        hits = await qdrant.dense_search(
            collection_name=collection_name,
            query_text=probe.query,
            top_k=top_k,
        )
        texts = [h.get("text", "") for h in hits]
        recall = compute_keyword_recall(texts, probe.expected_keywords)
        recalls.append(recall)
        per_query.append(
            {
                "probe_id": probe.id,
                "query": probe.query,
                "recall": round(recall, 4),
                "retrieval_ms": round((time.perf_counter() - t0) * 1000, 1),
                "top_k": top_k,
            }
        )

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    return {
        "collection": collection_name,
        "chunk_count": chunk_count,
        "probe_count": len(probes),
        "top_k": top_k,
        "avg_keyword_recall": round(avg_recall, 4),
        "target_recall": 0.50,
        "target_chunks": 1000,
        "chunks_under_target": chunk_count < 1000,
        "recall_above_target": avg_recall > 0.50,
        "passed": chunk_count < 1000 and avg_recall > 0.50,
        "per_query": per_query,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="make_knowledge acceptance run")
    p.add_argument("--collection", required=True, help="Qdrant collection name")
    p.add_argument("--profile", default="papers", help="Knowledge profile")
    p.add_argument(
        "--probes",
        default="data/evaluation/papers_probes.json",
        help="Probe fixture JSON",
    )
    p.add_argument("--top-k", type=int, default=10, help="Retrieval top_k (default: 10)")
    p.add_argument("--build", action="store_true", help="Run full knowledge build before eval")
    p.add_argument("--input", default="papers", help="Input dir for --build")
    p.add_argument(
        "--output-dir",
        default="",
        help="Artifact root (default: {ALLOWED_CORPUS_DIR}/.knowledge)",
    )
    p.add_argument(
        "--artifact-dir",
        default="",
        help="Override artifact dir for stats (default: derived from output-dir + collection)",
    )
    p.add_argument(
        "--skip-hardware-guard",
        action="store_true",
        help="Skip RAM pre-flight (tests only)",
    )
    p.add_argument("--output", default="", help="JSON report path")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> dict:
    profile = load_profile(args.profile)
    probes_path = Path(args.probes)
    if not probes_path.is_absolute():
        probes_path = ROOT / probes_path

    if args.build:
        if not args.skip_hardware_guard:
            check_hardware_guard(_DEFAULT_GUARD)
        artifact_dir = await _maybe_build(args, profile)
    else:
        settings = get_settings()
        if args.artifact_dir:
            artifact_dir = Path(args.artifact_dir).expanduser()
        elif args.output_dir:
            artifact_dir = Path(args.output_dir).expanduser() / args.collection
        elif settings.KNOWLEDGE_OUTPUT_DIR:
            artifact_dir = Path(settings.KNOWLEDGE_OUTPUT_DIR).expanduser() / args.collection
        else:
            artifact_dir = Path(settings.ALLOWED_CORPUS_DIR) / ".knowledge" / args.collection

    artifact_stats = _artifact_stats(artifact_dir)
    if not args.build:
        await _ensure_indexed(args.collection, artifact_dir, profile)
    eval_report = await _eval_collection(args.collection, probes_path, args.top_k)

    report = {
        "mode": "make_knowledge",
        "profile": args.profile,
        "artifacts": artifact_stats,
        "eval": eval_report,
        "acceptance": {
            "papers_ok": artifact_stats["papers"] >= 30,
            "topics_ok": artifact_stats["topics"] >= 8,
            "links_ok": artifact_stats["topics_with_links"] >= 8,
            "chunks_ok": eval_report["chunks_under_target"],
            "recall_ok": eval_report["recall_above_target"],
            "passed": (
                artifact_stats["papers"] >= 30
                and artifact_stats["topics"] >= 8
                and eval_report["chunks_under_target"]
                and eval_report["recall_above_target"]
            ),
        },
    }
    return report


def main() -> int:
    args = parse_args()
    try:
        report = asyncio.run(_run(args))
    except Exception as exc:
        print(f"ACCEPTANCE ABORT: {exc}")
        return 1

    print(json.dumps(report, indent=2))
    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")

    if report["acceptance"]["passed"]:
        print("ACCEPTANCE OK")
        return 0

    print("ACCEPTANCE INCOMPLETE — see acceptance block above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())