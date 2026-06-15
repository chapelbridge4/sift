#!/usr/bin/env python3
"""Retrieval-focused RAG failure triage over BEIR/SciFact.

Reuses the SciFact download/parse/embed/index/query pattern from
scripts/benchmark_beir.py and wires the result into the sift triage
classifier to produce a per-query failure distribution.

IMPORTANT — RETRIEVAL-FOCUSED TRIAGE
--------------------------------------
All traces are built with ``answer=None`` and ``reranked=None``.
This means only *retrieval-stage* failures can surface:
  - RELEVANT_NOT_RETRIEVED  (gold not in top-k)
  - (pass) gold present in top-k, no answer to evaluate

Generation-stage failures (UNFAITHFUL, INCOMPLETE, CONTEXT_IGNORED) and
reranking-stage failures (RELEVANT_DEMOTED, …) require real answers and a
reranker respectively.  They will NOT appear in this report unless those
inputs are supplied.  The distribution below therefore reflects the
*retrieval health* of the MiniLM-L6 dense index, not the full RAG pipeline.

Usage:
    .venv/bin/python scripts/run_triage.py
    .venv/bin/python scripts/run_triage.py --top-k 10 --max-queries 100
    .venv/bin/python scripts/run_triage.py --out reports/triage/scifact_sample.md
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Reuse SciFact loaders from benchmark_beir
from scripts.benchmark_beir import (
    COLLECTION_NAME,
    MODEL_NAME,
    download_scifact,
    parse_corpus,
    parse_qrels,
    parse_queries,
)

from app.triage.classifier import classify
from app.triage.signals import QueryTrace
from app.triage.taxonomy import RAGFailureType

# ---------------------------------------------------------------------------
# Async triage runner
# ---------------------------------------------------------------------------


async def run_triage(
    top_k: int = 10,
    max_queries: int = 100,
    out_path: Path | None = None,
) -> None:
    """Download SciFact, build the dense index, run per-query triage, report."""
    print("=== sift — Retrieval-Focused RAG Triage over BEIR/SciFact ===")
    print("NOTE: answer=None; only retrieval-stage failures are observable.\n")

    # 1. Data
    print("[1/4] Data")
    download_scifact()
    corpus = parse_corpus()
    queries = parse_queries()
    qrels = parse_qrels()
    print(f"  corpus={len(corpus)} docs, queries={len(queries)}, qrels qids={len(qrels)}")

    # 2. Embedding model
    print(f"\n[2/4] Embedding model ({MODEL_NAME})")
    t0 = time.perf_counter()
    model = TextEmbedding(model_name=MODEL_NAME)
    dim = len(list(model.embed(["probe"]))[0])
    print(f"  Loaded in {(time.perf_counter() - t0)*1000:.0f}ms — dim={dim}")

    # 3. Index
    print(f"\n[3/4] Indexing {len(corpus)} documents into in-memory Qdrant")
    t0 = time.perf_counter()
    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
    )

    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]
    dense_vecs = list(model.embed(doc_texts))

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector={"dense": vec.tolist()},
            payload={"doc_id": did, "text": corpus[did]},
        )
        for did, vec in zip(doc_ids, dense_vecs)
    ]

    batch_size = 512
    for i in range(0, len(points), batch_size):
        await client.upsert(COLLECTION_NAME, points=points[i : i + batch_size], wait=True)
    print(f"  Indexed in {(time.perf_counter() - t0)*1000:.0f}ms")

    # 4. Triage
    eval_qids = [qid for qid in queries if qid in qrels][:max_queries]
    n_with_qrels = len([qid for qid in queries if qid in qrels])
    print(f"\n[4/4] Triage — top_k={top_k}, max_queries={max_queries}")
    print(f"  Queries with qrels: {n_with_qrels}, evaluating: {len(eval_qids)}")

    n_pass = 0
    n_fail = 0
    failure_type_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()

    for qid in eval_qids:
        query_text = queries[qid]
        gold_ids: set[str] = qrels[qid]

        # Retrieve
        query_vec = list(model.embed([query_text]))[0].tolist()
        hits = (
            await client.query_points(
                COLLECTION_NAME,
                query=query_vec,
                using="dense",
                limit=top_k,
                with_payload=True,
            )
        ).points

        # Build retrieved list with doc text (required for groundedness signal)
        retrieved = [
            {
                "doc_id": h.payload["doc_id"],
                "score": h.score,
                "text": h.payload.get("text", ""),
            }
            for h in hits
        ]

        # Build trace — answer=None, reranked=None (retrieval-focused)
        trace = QueryTrace(
            query=query_text,
            retrieved=retrieved,
            gold_ids=gold_ids,
            reranked=None,   # no reranker
            answer=None,     # no LLM; generation triage skipped
            top_k=top_k,
        )

        verdict = classify(trace)

        # Retrieval-focused interpretation: when answer=None, the classifier
        # will flag INCOMPLETE (generation stage) for any query where gold IS
        # retrieved but no answer was produced.  That is expected behaviour
        # from the classifier's perspective, but this script only measures
        # retrieval health.  We therefore treat any verdict whose primary_stage
        # is "generation" or "reranking" as a PASS for the purposes of this
        # report, since we cannot evaluate those stages without real answers /
        # a real reranker.  Only retrieval-stage failures are counted.
        is_retrieval_failure = (
            verdict.failure_types
            and verdict.primary_stage == "retrieval"
        )

        if is_retrieval_failure:
            n_fail += 1
            for ft, _conf in verdict.failure_types:
                failure_type_counts[ft.name] += 1
            if verdict.primary_stage:
                stage_counts[verdict.primary_stage] += 1
        else:
            n_pass += 1

    total = n_pass + n_fail
    pass_rate = n_pass / total if total else 0.0

    # --- Summary ---
    print("\n" + "=" * 60)
    print("TRIAGE SUMMARY (retrieval-focused; answer=None)")
    print("=" * 60)
    print(f"Total queries evaluated : {total}")
    print(f"Passed (gold retrieved) : {n_pass}  ({pass_rate*100:.1f}%)")
    print(f"Failed                  : {n_fail}  ({(1-pass_rate)*100:.1f}%)")
    print()
    print("Failure-type breakdown:")
    for ft_name, count in failure_type_counts.most_common():
        pct = count / n_fail * 100 if n_fail else 0
        print(f"  {ft_name:<30} {count:>4}  ({pct:.1f}% of failures)")
    print()
    print("Stage breakdown (among failures):")
    for stage, count in stage_counts.most_common():
        pct = count / n_fail * 100 if n_fail else 0
        print(f"  {stage:<15} {count:>4}  ({pct:.1f}% of failures)")

    # Takeaway
    dominant_ft = failure_type_counts.most_common(1)
    if dominant_ft and n_fail > 0:
        ft_name, ft_count = dominant_ft[0]
        pct = ft_count / n_fail * 100
        takeaway = (
            f"Of {n_fail} failed queries, {pct:.0f}% are {ft_name} "
            f"— retrieval-stage, not generation (generation triage skipped: answer=None)."
        )
    else:
        takeaway = f"All {total} queries passed retrieval triage (gold found in top-{top_k})."
    print(f"\nTakeaway: {takeaway}")

    # --- Write markdown report ---
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_report(
            out_path=out_path,
            corpus_size=len(corpus),
            n_queries=total,
            n_pass=n_pass,
            n_fail=n_fail,
            pass_rate=pass_rate,
            top_k=top_k,
            failure_type_counts=failure_type_counts,
            stage_counts=stage_counts,
            takeaway=takeaway,
        )
        print(f"\nReport written to {out_path}")


def _write_report(
    *,
    out_path: Path,
    corpus_size: int,
    n_queries: int,
    n_pass: int,
    n_fail: int,
    pass_rate: float,
    top_k: int,
    failure_type_counts: Counter[str],
    stage_counts: Counter[str],
    takeaway: str,
) -> None:
    """Write the triage markdown report with real numbers only."""
    lines: list[str] = []

    lines.append("# sift — Retrieval-Focused RAG Triage Report")
    lines.append("")
    lines.append("> **Scope**: RETRIEVAL-FOCUSED triage only (`answer=None`, `reranked=None`).")
    lines.append("> Generation-stage failures (UNFAITHFUL, INCOMPLETE, CONTEXT_IGNORED) and")
    lines.append("> reranking-stage failures (RELEVANT_DEMOTED, …) are **not measured** in this")
    lines.append("> run because no LLM answers or reranker outputs were supplied.")
    lines.append("> This report reflects the retrieval health of the MiniLM-L6 dense index only.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| Dataset | BEIR / SciFact (test split) |")
    lines.append(f"| Corpus size | {corpus_size:,} documents |")
    lines.append(f"| Queries evaluated | {n_queries} |")
    lines.append(f"| top_k | {top_k} |")
    lines.append(f"| Embedding model | sentence-transformers/all-MiniLM-L6-v2 (fastembed) |")
    lines.append(f"| Vector store | Qdrant in-memory (no Docker) |")
    lines.append(f"| answer | None (retrieval-focused; no LLM) |")
    lines.append(f"| reranked | None (no reranker) |")
    lines.append("")
    lines.append("## Retrieval Results")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total queries | {n_queries} |")
    lines.append(f"| Passed (gold in top-{top_k}) | {n_pass} ({pass_rate*100:.1f}%) |")
    lines.append(f"| Failed (gold not retrieved) | {n_fail} ({(1-pass_rate)*100:.1f}%) |")
    lines.append("")
    lines.append("## Failure-Type Distribution")
    lines.append("")
    lines.append("| Failure Type | Stage | Count | % of Failures |")
    lines.append("|---|---|---|---|")
    for ft_name, count in failure_type_counts.most_common():
        try:
            ft = RAGFailureType[ft_name]
            stage = ft.stage
        except KeyError:
            stage = "unknown"
        pct = count / n_fail * 100 if n_fail else 0
        lines.append(f"| `{ft_name}` | {stage} | {count} | {pct:.1f}% |")
    if not failure_type_counts:
        lines.append("| *(none — all queries passed)* | — | 0 | 0.0% |")
    lines.append("")
    lines.append("## Stage Breakdown (among failures)")
    lines.append("")
    lines.append("| Stage | Count | % of Failures |")
    lines.append("|---|---|---|")
    for stage, count in stage_counts.most_common():
        pct = count / n_fail * 100 if n_fail else 0
        lines.append(f"| {stage} | {count} | {pct:.1f}% |")
    if not stage_counts:
        lines.append("| *(none)* | 0 | 0.0% |")
    lines.append("")
    lines.append("## Takeaway")
    lines.append("")
    lines.append(f"> {takeaway}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*Generated by `scripts/run_triage.py`. "
        "To enable generation-stage triage, supply real answers and set `answer=<str>` in the trace.*"
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrieval-focused RAG failure triage over BEIR/SciFact"
    )
    p.add_argument("--top-k", type=int, default=10, help="Retrieve top-k docs per query")
    p.add_argument("--max-queries", type=int, default=100, help="Max queries (default 100)")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports/triage/scifact_sample.md"),
        help="Output markdown report path (default: reports/triage/scifact_sample.md)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_triage(top_k=args.top_k, max_queries=args.max_queries, out_path=args.out))
