"""
RAG Evaluation Harness — brain_rag benchmark runner.

Usage:
    .venv/bin/python -m app.tuning.eval_harness --fixture tests/fixtures/rag_queries.json
    .venv/bin/python -m app.tuning.eval_harness --fixture tests/fixtures/rag_queries.json --retrieval-only
    .venv/bin/python -m app.tuning.eval_harness --fixture tests/fixtures/rag_queries.json --skip-llm
"""

import json
import sys
import time
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import psutil

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.qdrant_service import QdrantService
from app.services.llm_service import LLMService


def load_fixtures(fixture_path: str) -> dict:
    """Load evaluation fixtures from JSON file."""
    with open(fixture_path, "r") as f:
        return json.load(f)


SWAP_THRESHOLD_BYTES = 1024 * 1024 * 1024  # 1GB


def check_pre_benchmark_swap() -> bool:
    """Warn but never abort — graceful degradation under memory pressure."""
    swap = psutil.swap_memory()
    if swap.used > SWAP_THRESHOLD_BYTES:
        print(f"[WARN] Swap at {swap.used / 1024 / 1024:.0f}MB — benchmark will proceed under memory pressure.")
    return False


async def check_collection_populated(qdrant: QdrantService, collection_name: str) -> bool:
    """Return True if collection has indexed documents."""
    try:
        await qdrant.initialize()
        result = await qdrant.client.get_collection(collection_name)
        return result.points_count > 0
    except Exception:
        return False


async def run_retrieval_eval(
    qdrant: QdrantService,
    queries: list[dict],
    collection_name: str,
    top_k: int = 10,
) -> list[dict]:
    """Run retrieval for each query and measure recall."""
    results = []
    for q in queries:
        qid = q["id"]
        query_text = q["query"]
        expected_chunks = set(q.get("expected_chunk_ids", []))
        expected_keywords = q.get("expected_keywords", [])

        try:
            retrieved = await qdrant.dense_search(
                collection_name=collection_name,
                query_text=query_text,
                top_k=top_k,
            )

            retrieved_ids = {r.get("id") for r in retrieved}
            retrieved_texts = [r.get("text", "") for r in retrieved]

            # recall = fraction of expected chunks found
            if expected_chunks:
                recall = len(retrieved_ids & expected_chunks) / len(expected_chunks)
            else:
                # keyword-based proxy: check how many expected keywords appear in retrieved texts
                combined = " ".join(retrieved_texts).lower()
                keyword_matches = sum(1 for kw in expected_keywords if kw.lower() in combined)
                recall = keyword_matches / len(expected_keywords) if expected_keywords else 0.0

            results.append({
                "query_id": qid,
                "query": query_text,
                "retrieved_count": len(retrieved),
                "recall": round(recall, 3),
                "retrieved_ids": list(retrieved_ids)[:5],
            })
        except Exception as e:
            results.append({
                "query_id": qid,
                "query": query_text,
                "retrieved_count": 0,
                "recall": 0.0,
                "error": str(e),
            })
    return results


async def run_generation_eval(
    llm: LLMService,
    queries: list[dict],
    chunks_by_query: dict[str, list[dict]],
    top_k: int = 5,
) -> list[dict]:
    """Run generation for each query and measure answer faithfulness."""
    results = []
    for q in queries:
        qid = q["id"]
        query_text = q["query"]
        expected_keywords = q.get("expected_keywords", [])

        chunks = chunks_by_query.get(qid, [])
        if not chunks:
            results.append({"query_id": qid, "faithfulness": 0.0, "answer": "", "latency_ms": 0})
            continue

        # Build context from top chunks
        context_texts = [c.get("text", "")[:500] for c in chunks[:top_k]]
        context = "\n\n".join(context_texts)

        prompt = f"Context:\n{context}\n\nQuestion: {query_text}\n\nAnswer based on the context above."

        start = time.time()
        try:
            response = await llm.generate(
                prompt=prompt,
                model_name="fast",
                max_tokens=200,
            )
            answer = response.get("text", "") if isinstance(response, dict) else str(response)
        except Exception as e:
            answer = f"[ERROR: {e}]"
        elapsed_ms = (time.time() - start) * 1000

        # keyword faithfulness
        answer_lower = answer.lower()
        keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        faithfulness = keyword_hits / len(expected_keywords) if expected_keywords else 0.0

        results.append({
            "query_id": qid,
            "answer_preview": answer[:200],
            "faithfulness": round(faithfulness, 3),
            "keyword_hits": keyword_hits,
            "total_keywords": len(expected_keywords),
            "latency_ms": round(elapsed_ms, 1),
        })
    return results


async def run_full_benchmark(
    fixture_path: str,
    retrieval_only: bool = False,
    skip_llm: bool = False,
    collection_name: str | None = None,
) -> dict:
    """
    Run the full RAG evaluation benchmark.

    Returns a dict with:
        - timestamp, duration_seconds
        - retrieval_results, generation_results (if not retrieval_only)
        - summary (avg_recall, avg_faithfulness, avg_latency_ms)
    """
    from app.config import get_settings
    settings = get_settings()

    collection = collection_name or settings.DEFAULT_COLLECTION_NAME

    # Load fixtures
    fixture = load_fixtures(fixture_path)
    queries = fixture.get("queries", [])
    if not queries:
        raise ValueError(f"No queries found in fixture: {fixture_path}")

    # Check swap before starting — warn but don't abort
    check_pre_benchmark_swap()

    # Init services
    qdrant = QdrantService()
    llm = LLMService()

    # Check if collection is populated
    if not await check_collection_populated(qdrant, collection):
        print("[WARN] Qdrant collection is empty. Run indexing first before benchmarking.")
        print("  .venv/bin/python -m app.services.document_parser --help")
        return {
            "status": "skipped",
            "reason": "empty_collection",
            "queries_count": len(queries),
            "message": "Index documents first before running harness.",
        }

    print(f"[HARNESS] Running evaluation on {len(queries)} queries...")
    print(f"  Collection: {collection}")
    print(f"  LLM generation: {'enabled' if not retrieval_only else 'disabled'}")

    start_time = time.time()

    # Phase 1: Retrieval evaluation
    print("[1/2] Running retrieval evaluation...")
    retrieval_results = await run_retrieval_eval(qdrant, queries, collection)

    # Phase 2: Generation evaluation
    generation_results = []
    if not retrieval_only and not skip_llm:
        print("[2/2] Running generation evaluation...")
        # Collect chunks per query for context
        chunks_by_query = {}
        for q in queries:
            qid = q["id"]
            retrieved = await qdrant.dense_search(
                collection_name=collection,
                query_text=q["query"],
                top_k=10,
            )
            chunks_by_query[qid] = retrieved

        generation_results = await run_generation_eval(llm, queries, chunks_by_query)
    else:
        print("[2/2] Skipping generation (--retrieval-only or --skip-llm)")

    elapsed = time.time() - start_time

    # Build output
    avg_recall = sum(r["recall"] for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0.0
    avg_faithfulness = 0.0
    avg_latency_ms = 0.0
    if generation_results:
        avg_faithfulness = sum(r["faithfulness"] for r in generation_results) / len(generation_results)
        avg_latency_ms = sum(r["latency_ms"] for r in generation_results) / len(generation_results)

    summary = {
        "total_queries": len(queries),
        "retrieval_only": retrieval_only or skip_llm,
        "avg_recall": round(avg_recall, 3),
        "avg_faithfulness": round(avg_faithfulness, 3),
        "avg_latency_ms": round(avg_latency_ms, 1),
        "duration_seconds": round(elapsed, 1),
        "swap_pressure_abort": False,
    }

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixture_version": fixture.get("version", "unknown"),
        "collection": collection,
        "retrieval_results": retrieval_results,
        "generation_results": generation_results,
        "summary": summary,
    }

    return output


def save_results(output: dict, output_dir: str = "app/tuning/results") -> Path:
    """Save benchmark results to JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"eval_{ts}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    return path


def print_summary(output: dict):
    """Print human-readable summary."""
    summary = output.get("summary", {})
    status = output.get("status", "done")
    if status == "skipped":
        reason = output.get("reason", "unknown")
        print(f"\n[HARNESS] Skipped: {reason}")
        if reason == "empty_collection":
            print("  → Index some papers first: .venv/bin/python -m app.services.document_parser")
        return

    print(f"\n[HARNESS] Benchmark complete — {summary.get('duration_seconds', '?')}s")
    print(f"  Queries:          {summary.get('total_queries', '?')}")
    print(f"  Avg recall:       {summary.get('avg_recall', '?')}")
    print(f"  Avg faithfulness: {summary.get('avg_faithfulness', '?')}")
    print(f"  Avg latency (ms): {summary.get('avg_latency_ms', '?')}")

    print("\n  Per-query retrieval recall:")
    for r in output.get("retrieval_results", []):
        qid = r.get("query_id", "?")
        recall = r.get("recall", "?")
        err = r.get("error", "")
        err_str = f"  ERROR: {err}" if err else ""
        print(f"    {qid}: recall={recall}{err_str}")

    if output.get("generation_results"):
        print("\n  Per-query generation:")
        for r in output["generation_results"]:
            qid = r.get("query_id", "?")
            faith = r.get("faithfulness", "?")
            lat = r.get("latency_ms", "?")
            print(f"    {qid}: faithfulness={faith}, latency={lat}ms")


def main():
    parser = argparse.ArgumentParser(description="brain_rag RAG evaluation harness")
    parser.add_argument(
        "--fixture",
        type=str,
        default="tests/fixtures/rag_queries.json",
        help="Path to evaluation fixtures JSON",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only run retrieval evaluation (skip generation)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM generation (same as --retrieval-only)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="app/tuning/results",
        help="Directory to save results JSON",
    )
    args = parser.parse_args()

    result = asyncio.run(run_full_benchmark(
        fixture_path=args.fixture,
        retrieval_only=args.retrieval_only or args.skip_llm,
        skip_llm=args.skip_llm,
        collection_name=args.collection,
    ))

    print_summary(result)

    if result.get("status") != "skipped":
        path = save_results(result, args.output_dir)
        print(f"\n  Results saved to: {path}")

    # Exit 0 if recall is reasonable (>0.3), else 1 to signal harness issue
    summary = result.get("summary", {})
    recall = summary.get("avg_recall", 0)
    if recall >= 0.3:
        sys.exit(0)
    else:
        print(f"\n[WARN] Low recall ({recall}). Index more documents or check retrieval.")
        sys.exit(1)


if __name__ == "__main__":
    main()