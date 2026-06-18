#!/usr/bin/env python3
"""End-to-end query smoke for make_knowledge collections (no LLM generation).

Exercises DocumentStore.recall_memories with topic boost + optional drill_down
against an already-built knowledge collection.

Usage:
    .venv/bin/python scripts/knowledge_query_smoke.py \\
        --collection ai_papers_knowledge \\
        --probes data/evaluation/papers_probes.json

    .venv/bin/python scripts/knowledge_query_smoke.py --collection ai_papers_knowledge --drill-down
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.kbforge.eval.keyword_recall import compute_keyword_recall  # noqa: E402
from app.kbforge.eval.probe_evaluator import load_probes  # noqa: E402
from app.knowledge.retrieval import infer_retrieval_layers, is_knowledge_collection  # noqa: E402
from app.pipeline.document_store import DocumentStore  # noqa: E402


async def _smoke_collection(
    collection: str,
    probes_path: Path,
    top_k: int,
    drill_down: bool,
) -> dict:
    store = DocumentStore()
    await store.initialize()
    qdrant = store.qdrant_service

    info = await qdrant.get_collection_info(collection)
    chunk_count = info.get("points_count") or info.get("vectors_count") or 0

    # Sanity: one probe with layer inspection
    sample_query = "How do language models transcribe endangered languages?"
    sample_hits = await store.recall_memories(
        collection_name=collection,
        query=sample_query,
        top_k=top_k,
        drill_down=drill_down,
    )
    knowledge = is_knowledge_collection(sample_hits)
    layers = infer_retrieval_layers(sample_hits) if knowledge else []
    doc_types = sorted(
        {h.get("metadata", {}).get("doc_type", "?") for h in sample_hits}
    )

    per_query: list[dict] = []
    recalls: list[float] = []
    drill_down_paper_hits = 0

    for probe in load_probes(probes_path):
        hits = await store.recall_memories(
            collection_name=collection,
            query=probe.query,
            top_k=top_k,
            drill_down=drill_down,
        )
        texts = [h.get("text", "") for h in hits]
        recall = compute_keyword_recall(texts, probe.expected_keywords)
        recalls.append(recall)
        paper_count = sum(
            1
            for h in hits
            if h.get("metadata", {}).get("doc_type") == "paper_summary"
        )
        drill_down_paper_hits += paper_count
        per_query.append(
            {
                "probe_id": probe.id,
                "recall": round(recall, 4),
                "hit_count": len(hits),
                "paper_summary_hits": paper_count,
                "doc_types": sorted(
                    {h.get("metadata", {}).get("doc_type", "?") for h in hits}
                ),
            }
        )

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    passed = (
        knowledge
        and chunk_count > 0
        and chunk_count < 1000
        and avg_recall > 0.50
        and (not drill_down or drill_down_paper_hits > 0)
    )

    return {
        "collection": collection,
        "chunk_count": chunk_count,
        "knowledge_collection": knowledge,
        "drill_down": drill_down,
        "sample_layers": layers,
        "sample_doc_types": doc_types,
        "avg_keyword_recall": round(avg_recall, 4),
        "drill_down_paper_hits_total": drill_down_paper_hits,
        "passed": passed,
        "per_query": per_query,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="make_knowledge query-path smoke test")
    p.add_argument("--collection", default="ai_papers_knowledge")
    p.add_argument("--probes", default="data/evaluation/papers_probes.json")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--drill-down",
        action="store_true",
        help="Enable hierarchical drill-down (topic → paper summaries)",
    )
    p.add_argument("--output", default="", help="Optional JSON report path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    probes_path = Path(args.probes)
    if not probes_path.is_absolute():
        probes_path = ROOT / probes_path

    report = asyncio.run(
        _smoke_collection(
            collection=args.collection,
            probes_path=probes_path,
            top_k=args.top_k,
            drill_down=args.drill_down,
        )
    )
    print(json.dumps(report, indent=2))

    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")

    if report["passed"]:
        print("QUERY SMOKE OK")
        return 0

    print("QUERY SMOKE FAILED — see report above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())