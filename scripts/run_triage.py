#!/usr/bin/env python3
"""RAG failure triage over BEIR/SciFact — retrieval-only by default, multi-stage on demand.

Reuses the SciFact download/parse/embed/index/query pattern from
scripts/benchmark_beir.py and wires the result into the sift triage
classifier to produce a per-query failure distribution.

MODES (which signals are active determines which stages can fail)
-----------------------------------------------------------------
* default                  retrieved ✓ · reranked ✗ · answers ✗ · judge ✗
    Only *retrieval-stage* failures surface (RELEVANT_NOT_RETRIEVED). This is
    the zero-dependency report — no model downloads beyond the embedder.
* ``--rerank``             retrieved ✓ · reranked ✓ · answers ✗ · judge ✗
    Runs the brain cross-encoder reranker (app/brain/amygdala.py :: Amygdala)
    over each query's retrieved candidates and feeds the reranked doc_id order
    into the classifier, so *reranking-stage* demotions (RELEVANT_DEMOTED) can
    surface alongside retrieval misses. Downloads a small (~80 MB) cross-encoder
    on first use; CPU, single process.
* ``--with-answers``       retrieved ✓ · reranked ✓/✗ · answers ✓ · judge ✓
    Additionally generates a real answer per query via the configured local
    inference backend and enables the optional LLM judge, so *generation-stage*
    subtypes (UNFAITHFUL / INCOMPLETE / CONTEXT_IGNORED) can be disambiguated.
    Requires a downloaded GGUF/MLX model and is SLOW; skips gracefully per-query
    when no backend/model is available. Never required by tests/CI.

The generated report always states exactly which signals were active, so the
distribution is honestly scoped to what was actually measured.

Usage:
    .venv/bin/python scripts/run_triage.py
    .venv/bin/python scripts/run_triage.py --rerank --max-queries 100 \
        --out reports/triage/scifact_full_sample.md
    .venv/bin/python scripts/run_triage.py --rerank --with-answers --max-queries 20
"""

from __future__ import annotations

import argparse
import asyncio
import gc
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

from app.triage.classifier import classify
from app.triage.signals import QueryTrace
from app.triage.taxonomy import RAGFailureType

# Reuse SciFact loaders from benchmark_beir
from scripts.benchmark_beir import (
    COLLECTION_NAME,
    MODEL_NAME,
    download_dataset,
    parse_corpus,
    parse_qrels,
    parse_queries,
)

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def rerank_to_doc_ids(reranked_docs: list[dict]) -> list[str]:
    """Map reranked document dicts back to an ordered list of ``doc_id`` strings.

    The brain reranker (``Amygdala.rerank``) returns document dicts (copies of
    the candidates, re-sorted by cross-encoder score). The triage classifier
    only needs the post-rerank *order* of doc_ids, which is what
    ``QueryTrace.reranked`` expects. Documents without a ``doc_id`` are skipped.

    Pure and deterministic — covered by tests/triage/test_run_triage_helpers.py.
    """
    return [doc["doc_id"] for doc in reranked_docs if doc.get("doc_id") is not None]


# ---------------------------------------------------------------------------
# Lazy resource builders (never instantiated at import or in default mode)
# ---------------------------------------------------------------------------


def _build_reranker(top_k: int):
    """Instantiate the brain cross-encoder reranker for a clean top-k reorder.

    Overrides the Amygdala instance settings so ``rerank`` is a *pure* reorder
    of all ``top_k`` retrieved candidates:
      * RERANK_ENABLED=True   — otherwise rerank returns the input untouched.
      * RERANK_TOP_K=top_k    — keep every candidate so a demotion is visible
                                as a within-window rank change (the classifier
                                treats gold dropping out of ``reranked`` as
                                "not demoted", so we keep the full window).
      * DIVERSIFY_SOURCES=False — source diversification keys on
                                ``metadata.document_id`` which SciFact docs lack,
                                which would collapse the order; disable it so the
                                reordering reflects the cross-encoder alone.

    The cross-encoder model itself is loaded lazily inside ``Amygdala.rerank``
    on first call (memory-guarded), not here.
    """
    from app.brain.amygdala import Amygdala

    reranker = Amygdala()
    # Route every override through a per-instance copy (model_copy) so the shared
    # global Settings singleton is never mutated — mutating reranker.settings in
    # place would leak process-wide (latent bug if this runner is ever imported
    # into a server). The cross-encoder reads these at rerank time off the copy.
    #
    # RERANK_MIN_AVAILABLE_GB is lowered from the default 1.5: on an 8 GB box the
    # in-memory SciFact index leaves ~1.3 GB free, which would silently trip the
    # guard and skip the cross-encoder. The reranker is tiny (~80 MB
    # ms-marco-MiniLM-L-6), so for this single-process offline eval we let it run.
    reranker.settings = reranker.settings.model_copy(
        update={
            "RERANK_ENABLED": True,
            "RERANK_TOP_K": top_k,
            "DIVERSIFY_SOURCES": False,
            "RERANK_MIN_AVAILABLE_GB": 0.8,
        }
    )
    return reranker


def _build_inference_backend():
    """Return the configured inference backend, or None if unavailable.

    Construction is cheap (no model load); the model is loaded on first
    ``generate_rag_response`` call. Any failure here means generation is skipped.
    """
    try:
        from app.services.inference import get_inference_backend

        return get_inference_backend()
    except Exception as exc:  # pragma: no cover - env-dependent
        print(f"  [--with-answers] no inference backend available ({exc}); skipping answers.")
        return None


# ---------------------------------------------------------------------------
# Async triage runner
# ---------------------------------------------------------------------------


async def run_triage(
    top_k: int = 10,
    max_queries: int = 100,
    out_path: Path | None = None,
    *,
    do_rerank: bool = False,
    with_answers: bool = False,
) -> None:
    """Download SciFact, build the dense index, run per-query triage, report."""
    print("=== sift — RAG Failure Triage over BEIR/SciFact ===")
    print(
        f"Signals: retrieved=ON  reranked={'ON' if do_rerank else 'off'}  "
        f"answers={'ON' if with_answers else 'off'}  judge={'ON' if with_answers else 'off'}\n"
    )

    # 1. Data
    print("[1/4] Data")
    download_dataset("scifact")
    corpus = parse_corpus("scifact")
    queries = parse_queries("scifact")
    qrels = parse_qrels("scifact")
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

    # Reclaim the large indexing intermediates before loading the cross-encoder.
    # On an 8 GB box these vectors leave too little headroom for the reranker's
    # RAM guard (RERANK_MIN_AVAILABLE_GB) to pass, which would silently skip it.
    del dense_vecs, points, doc_texts
    gc.collect()

    # Lazy resources for the chosen mode (only built when their flag is set).
    reranker = _build_reranker(top_k) if do_rerank else None
    backend = _build_inference_backend() if with_answers else None

    # 4. Triage
    eval_qids = [qid for qid in queries if qid in qrels][:max_queries]
    n_with_qrels = len([qid for qid in queries if qid in qrels])
    print(f"\n[4/4] Triage — top_k={top_k}, max_queries={max_queries}")
    print(f"  Queries with qrels: {n_with_qrels}, evaluating: {len(eval_qids)}")

    n_pass = 0
    n_fail = 0
    n_answers = 0  # answers actually generated (with_answers mode)
    n_reorders = 0  # queries where the reranker produced a different top order
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

        # Reranking stage (optional) — produces post-rerank doc_id order.
        reranked: list[str] | None = None
        if reranker is not None:
            retrieved_order = [d["doc_id"] for d in retrieved]
            reranked_docs = reranker.rerank(retrieved, query_text, initial_top_k=top_k)
            reranked = rerank_to_doc_ids(reranked_docs)
            if reranked != retrieved_order:
                n_reorders += 1

        # Generation stage (optional) — real answer via local backend.
        answer: str | None = None
        if backend is not None:
            try:
                contexts = [d["text"] for d in retrieved if d.get("text")]
                answer = await backend.generate_rag_response(
                    query=query_text, retrieved_contexts=contexts, max_tokens=128
                )
                if answer and answer.strip():
                    n_answers += 1
            except Exception as exc:  # pragma: no cover - model-dependent
                print(f"  [--with-answers] generation failed for {qid} ({exc}); no answer.")
                answer = None

        trace = QueryTrace(
            query=query_text,
            retrieved=retrieved,
            gold_ids=gold_ids,
            reranked=reranked,
            answer=answer,
            top_k=top_k,
        )

        verdict = classify(trace, use_llm_judge=with_answers or None)

        # Honest counting: a verdict only counts as a failure if the stage it
        # blames had a *live signal* on this run. Retrieval is always live.
        # Reranking is live only when a reranker actually produced an order.
        # Generation is live only when an answer was actually generated — the
        # classifier otherwise fires INCOMPLETE for "no answer", which is a
        # property of this eval harness, not a real generation failure.
        live_stages = {"retrieval"}
        if reranked is not None:
            live_stages.add("reranking")
        if answer is not None and answer.strip():
            live_stages.add("generation")

        counts_as_failure = (
            bool(verdict.failure_types) and verdict.primary_stage in live_stages
        )

        if counts_as_failure:
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
    print("TRIAGE SUMMARY")
    print("=" * 60)
    print(f"Total queries evaluated : {total}")
    print(f"Passed                  : {n_pass}  ({pass_rate*100:.1f}%)")
    print(f"Failed                  : {n_fail}  ({(1-pass_rate)*100:.1f}%)")
    if do_rerank:
        print(f"Reranker reordered      : {n_reorders}/{total} queries (else cross-encoder was a no-op/skipped)")
    if with_answers:
        print(f"Answers generated       : {n_answers}/{total}")
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

    takeaway = _build_takeaway(
        n_fail=n_fail,
        total=total,
        top_k=top_k,
        failure_type_counts=failure_type_counts,
        stage_counts=stage_counts,
    )
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
            do_rerank=do_rerank,
            with_answers=with_answers,
            n_answers=n_answers,
            n_reorders=n_reorders,
        )
        print(f"\nReport written to {out_path}")


def _build_takeaway(
    *,
    n_fail: int,
    total: int,
    top_k: int,
    failure_type_counts: Counter[str],
    stage_counts: Counter[str],
) -> str:
    """Honest one-line takeaway reflecting only what was measured."""
    if n_fail == 0:
        return f"All {total} queries passed triage (gold found in top-{top_k}, no downstream failure)."

    dominant_ft, ft_count = failure_type_counts.most_common(1)[0]
    pct = ft_count / n_fail * 100
    n_stages = len(stage_counts)
    if n_stages > 1:
        stage_list = ", ".join(f"{s} ({c})" for s, c in stage_counts.most_common())
        return (
            f"Of {n_fail} failed queries, the dominant mode is {dominant_ft} ({pct:.0f}%); "
            f"failures span {n_stages} stages: {stage_list}."
        )
    only_stage = stage_counts.most_common(1)[0][0]
    return (
        f"Of {n_fail} failed queries, {pct:.0f}% are {dominant_ft} "
        f"— all at the {only_stage} stage (no other stage fired with the active signals)."
    )


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
    do_rerank: bool,
    with_answers: bool,
    n_answers: int,
    n_reorders: int,
) -> None:
    """Write the triage markdown report with real numbers only."""

    def yn(active: bool) -> str:
        return "✓" if active else "✗"

    lines: list[str] = []

    lines.append("# sift — Multi-Stage RAG Triage Report")
    lines.append("")
    lines.append("> **Active signals** (these determine which stages can fail):")
    lines.append(f"> retrieved {yn(True)} · reranked {yn(do_rerank)} · "
                 f"answers {yn(with_answers)} · LLM judge {yn(with_answers)}.")
    lines.append(">")
    if do_rerank and with_answers:
        lines.append("> All four pipeline stages with live signals are evaluated: retrieval misses,")
        lines.append("> reranker demotions, and generation-stage subtypes. The distribution below")
        lines.append("> reflects the full pipeline as actually exercised on this run.")
    elif do_rerank:
        lines.append("> Retrieval and reranking stages are evaluated (real cross-encoder reorder).")
        lines.append("> Generation-stage failures are **not measured** (no answers / judge); they")
        lines.append("> require `--with-answers` and a downloaded local model.")
        if n_reorders == 0:
            lines.append(">")
            lines.append("> **NB:** the reranker reordered 0 queries on this run — the cross-encoder")
            lines.append("> was a no-op or was skipped by its RAM guard, so no `RELEVANT_DEMOTED`")
            lines.append("> demotion could surface. The reranking stage was wired but inactive.")
    else:
        lines.append("> RETRIEVAL-FOCUSED only (`reranked=None`, `answer=None`): reranking- and")
        lines.append("> generation-stage failures are **not measured** on this run.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append("| Dataset | BEIR / SciFact (test split) |")
    lines.append(f"| Corpus size | {corpus_size:,} documents |")
    lines.append(f"| Queries evaluated | {n_queries} |")
    lines.append(f"| top_k | {top_k} |")
    lines.append("| Embedding model | sentence-transformers/all-MiniLM-L6-v2 (fastembed) |")
    lines.append("| Vector store | Qdrant in-memory (no Docker) |")
    if do_rerank:
        lines.append("| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (Amygdala.rerank, CPU) |")
        lines.append(f"| Reranker reordered | {n_reorders}/{n_queries} queries |")
    else:
        lines.append("| Reranker | None (retrieval order only) |")
    if with_answers:
        lines.append(f"| Answers | {n_answers}/{n_queries} generated (local backend) |")
        lines.append("| LLM judge | enabled (disambiguates generation subtypes) |")
    else:
        lines.append("| Answers | None (generation stage not exercised) |")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total queries | {n_queries} |")
    lines.append(f"| Passed | {n_pass} ({pass_rate*100:.1f}%) |")
    lines.append(f"| Failed | {n_fail} ({(1-pass_rate)*100:.1f}%) |")
    lines.append("")
    lines.append("## Failure-Type Distribution")
    lines.append("")
    lines.append("| Failure Type | Stage | Count | % of Failures |")
    lines.append("|---|---|---|---|")
    for ft_name, count in failure_type_counts.most_common():
        try:
            stage = RAGFailureType[ft_name].stage
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
        "*Generated by `scripts/run_triage.py`. The active-signals line above scopes the "
        "distribution honestly: a stage with no live signal cannot fail here.*"
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAG failure triage over BEIR/SciFact (retrieval-only by default; multi-stage on demand)"
    )
    p.add_argument("--top-k", type=int, default=10, help="Retrieve top-k docs per query")
    p.add_argument("--max-queries", type=int, default=100, help="Max queries (default 100)")
    p.add_argument(
        "--rerank",
        action="store_true",
        help="Run the brain cross-encoder reranker so reranking-stage demotions can surface",
    )
    p.add_argument(
        "--with-answers",
        action="store_true",
        help="Generate real answers via the local backend + LLM judge (SLOW; needs a model; skips gracefully)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output markdown report path "
        "(default: scifact_full_sample.md when --rerank, else scifact_sample.md)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = args.out
    if out is None:
        out = (
            Path("reports/triage/scifact_full_sample.md")
            if (args.rerank or args.with_answers)
            else Path("reports/triage/scifact_sample.md")
        )
    asyncio.run(
        run_triage(
            top_k=args.top_k,
            max_queries=args.max_queries,
            out_path=out,
            do_rerank=args.rerank,
            with_answers=args.with_answers,
        )
    )
