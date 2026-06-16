#!/usr/bin/env python3
"""
Local RAG benchmark runner.

Runs the current pipeline on fixture documents and queries,
emitting structured metrics for retrieval and answer quality.

Usage:
    python scripts/run_benchmark.py                    # defaults
    python scripts/run_benchmark.py --model-profile fast --fusion-method dbsf
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

sys.path.insert(0, str(__file__).rsplit('/scripts/', 1)[0])

from tests.benchmark_fixtures import (
    export_fixtures_as_metadata,
    export_fixtures_as_texts,
    get_benchmark_documents,
    get_benchmark_queries,
)

if TYPE_CHECKING:
    from app.brain import PrefrontalCortex


@dataclass
class StageTiming:
    """Timing for a benchmark stage."""
    stage_name: str
    duration_seconds: float


@dataclass
class QueryResult:
    """Result of a single query benchmark."""
    query: str
    question_type: str
    expected_topics: List[str]
    num_retrieved: int
    retrieved_topics: List[str]
    has_answer: bool
    answer_length: int
    model_used: Optional[str]
    timings: List[StageTiming]
    total_latency_seconds: float
    retrieval_score: float = 0.0


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    model_profile: str
    fusion_method: str
    sparse_strategy: str
    total_queries: int
    total_documents_indexed: int
    queries_with_answers: int
    average_latency_seconds: float
    average_retrieval_score: float
    results: List[QueryResult]
    timestamp: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local RAG benchmark")
    parser.add_argument(
        "--model-profile",
        default="balanced",
        choices=["fast", "balanced", "quality", "reasoning"],
        help="Model profile to use"
    )
    parser.add_argument(
        "--fusion-method",
        default="rrf",
        choices=["rrf", "dbsf"],
        help="Fusion method for hybrid search"
    )
    parser.add_argument(
        "--sparse-strategy",
        default="bm25",
        choices=["bm25", "bm25plus"],
        help="Sparse embedding strategy for retrieval comparison"
    )
    parser.add_argument(
        "--collection-name",
        default="benchmark_test_collection",
        help="Collection name for benchmark"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve per query"
    )
    parser.add_argument(
        "--output-json",
        help="Path to write JSON report (optional)"
    )
    return parser.parse_args()


def print_table_row(cols: List[str], widths: List[int]) -> None:
    print("| " + " | ".join(f"{c:<{w}}" for c, w in zip(cols, widths)) + " |")


def print_table_separator(widths: List[int]) -> None:
    print("+" + "+".join("-" * (w + 2) for w in widths) + "+")


def load_prefrontal_cortex_class():
    """Load the heavy runtime dependency only when the benchmark actually runs."""
    from app.brain import PrefrontalCortex

    return PrefrontalCortex


async def setup_collection(brain: PrefrontalCortex, collection_name: str) -> bool:
    """Create collection and index benchmark fixtures."""
    print(f"\n[1/4] Setting up collection '{collection_name}'...")
    start = time.time()

    exists = await brain.hippocampus.memory_exists(collection_name)
    if exists:
        print("  Collection exists, deleting first...")
        await brain.hippocampus.forget_memories(collection_name)

    success = await brain.hippocampus.create_memory_space(collection_name)
    if not success:
        print("  ERROR: Failed to create collection")
        return False

    documents = export_fixtures_as_texts()
    metadata = export_fixtures_as_metadata()

    result = await brain.hippocampus.form_memories(
        collection_name=collection_name,
        file_paths=[]  # We'll inject directly
    )

    texts = [doc.content for doc in get_benchmark_documents()]
    metadatas = [m for m in metadata]

    from app.utils.async_helpers import chunks

    embedding_service = brain.hippocampus.qdrant_service.embedding_service
    total_indexed = 0

    for batch_texts, batch_metadatas in zip(chunks(texts, 32), chunks(metadatas, 32)):
        dense_embs, sparse_embs = await embedding_service.generate_hybrid_embeddings(batch_texts)

        import uuid
        from datetime import datetime

        from qdrant_client.models import PointStruct, SparseVector

        points = []
        for text, meta, dense_emb, sparse_emb in zip(batch_texts, batch_metadatas, dense_embs, sparse_embs):
            meta["indexed_at"] = datetime.utcnow().isoformat()
            meta["text_preview"] = text[:200]
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_emb,
                    "sparse": SparseVector(
                        indices=list(sparse_emb.keys()),
                        values=list(sparse_emb.values())
                    )
                },
                payload={"text": text, **meta}
            ))

        await brain.hippocampus.qdrant_service.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )
        total_indexed += len(points)

    elapsed = time.time() - start
    print(f"  Indexed {total_indexed} documents in {elapsed:.2f}s")
    return True


async def run_query(
    brain: PrefrontalCortex,
    collection_name: str,
    query: str,
    question_type: str,
    expected_topics: List[str],
    model_profile: str,
    fusion_method: str,
    top_k: int
) -> QueryResult:
    """Run a single query and capture timing and results."""
    timings = []
    overall_start = time.time()

    retrieval_start = time.time()
    result = await brain.reason_with_context(
        query=query,
        collection_name=collection_name,
        top_k=top_k,
        use_hybrid=True,
        fusion_method=fusion_method,
        model_profile=model_profile,
        use_llm=True
    )
    retrieval_elapsed = time.time() - retrieval_start
    timings.append(StageTiming("retrieval_and_generation", retrieval_elapsed))

    retrieved_docs = result.get("retrieved_documents", [])
    retrieved_topics = []
    for doc in retrieved_docs:
        text = doc.get("text", "")
        meta = doc.get("metadata", {})
        topics = meta.get("expected_topics", [])
        retrieved_topics.extend(topics)

    retrieved_topics_set = list(set(retrieved_topics))
    expected_topics_set = list(set(expected_topics))
    hits = sum(1 for t in expected_topics_set if any(t.lower() in rt.lower() for rt in retrieved_topics_set))
    retrieval_score = hits / len(expected_topics_set) if expected_topics_set else 0.0

    answer = result.get("answer")
    has_answer = answer is not None and len(answer.strip()) > 0
    answer_length = len(answer) if answer else 0

    total_elapsed = time.time() - overall_start

    return QueryResult(
        query=query,
        question_type=question_type,
        expected_topics=expected_topics,
        num_retrieved=len(retrieved_docs),
        retrieved_topics=list(set(retrieved_topics)),
        has_answer=has_answer,
        answer_length=answer_length,
        model_used=result.get("model_used"),
        timings=timings,
        total_latency_seconds=total_elapsed,
        retrieval_score=retrieval_score
    )


async def cleanup_collection(brain: PrefrontalCortex, collection_name: str) -> None:
    """Clean up benchmark collection."""
    try:
        await brain.hippocampus.forget_memories(collection_name)
    except Exception:
        pass


async def run_benchmark(args: argparse.Namespace) -> BenchmarkReport:
    """Run the complete benchmark suite."""
    print("\n=== Brain_rag Local Benchmark ===")
    print(f"Model profile: {args.model_profile}")
    print(f"Fusion method: {args.fusion_method}")
    print(f"Sparse strategy: {args.sparse_strategy}")
    print(f"Top-K: {args.top_k}")

    brain_class = load_prefrontal_cortex_class()
    brain = brain_class()
    await brain.initialize()

    if not await setup_collection(brain, args.collection_name):
        raise RuntimeError("Failed to setup benchmark collection")

    print(f"\n[2/4] Running {len(get_benchmark_queries())} queries...")
    results: List[QueryResult] = []
    queries_with_answers = 0
    total_latency = 0.0

    for i, query_fixture in enumerate(get_benchmark_queries()):
        print(f"  Query {i+1}/{len(get_benchmark_queries())}: {query_fixture.query[:50]}...")
        result = await run_query(
            brain=brain,
            collection_name=args.collection_name,
            query=query_fixture.query,
            question_type=query_fixture.question_type,
            expected_topics=query_fixture.expected_evidence_topics,
            model_profile=args.model_profile,
            fusion_method=args.fusion_method,
            top_k=args.top_k
        )
        results.append(result)
        if result.has_answer:
            queries_with_answers += 1
        total_latency += result.total_latency_seconds

    print("\n[3/4] Cleaning up collection...")
    await cleanup_collection(brain, args.collection_name)

    from datetime import datetime
    total_retrieval_score = sum(r.retrieval_score for r in results)
    report = BenchmarkReport(
        model_profile=args.model_profile,
        fusion_method=args.fusion_method,
        sparse_strategy=args.sparse_strategy,
        total_queries=len(results),
        total_documents_indexed=len(get_benchmark_documents()),
        queries_with_answers=queries_with_answers,
        average_latency_seconds=total_latency / len(results) if results else 0,
        average_retrieval_score=total_retrieval_score / len(results) if results else 0,
        results=results,
        timestamp=datetime.utcnow().isoformat()
    )

    return report


def print_report(report: BenchmarkReport) -> None:
    """Print benchmark report in table format."""
    print("\n[4/4] Benchmark Report")
    print("=" * 70)

    widths = [40, 8, 6, 10, 8, 8]
    headers = ["Query", "Type", "Docs", "Latency", "Answered", "Score"]

    print_table_separator(widths)
    print_table_row(headers, widths)
    print_table_separator(widths)

    for r in report.results:
        query_short = r.query[:38] + ".." if len(r.query) > 40 else r.query
        latency_str = f"{r.total_latency_seconds:.2f}s"
        answered_str = "Y" if r.has_answer else "N"
        score_str = f"{r.retrieval_score:.2f}"
        print_table_row([query_short, r.question_type, str(r.num_retrieved), latency_str, answered_str, score_str], widths)

    print_table_separator(widths)

    avg_latency = report.average_latency_seconds
    total_time = sum(r.total_latency_seconds for r in report.results)

    print("\n--- Summary ---")
    print(f"Model profile:       {report.model_profile}")
    print(f"Fusion method:       {report.fusion_method}")
    print(f"Sparse strategy:     {report.sparse_strategy}")
    print(f"Total queries:       {report.total_queries}")
    print(f"Documents indexed:   {report.total_documents_indexed}")
    print(f"Queries answered:    {report.queries_with_answers}/{report.total_queries}")
    print(f"Average latency:     {avg_latency:.2f}s")
    print(f"Average retrieval:   {report.average_retrieval_score:.2f}")
    print(f"Total time:          {total_time:.2f}s")

    retrieval_times = [t.duration_seconds for r in report.results for t in r.timings if t.stage_name == "retrieval_and_generation"]
    if retrieval_times:
        print(f"Min query time:      {min(retrieval_times):.2f}s")
        print(f"Max query time:      {max(retrieval_times):.2f}s")

    print("=" * 70)


def write_json_report(report: BenchmarkReport, path: str) -> None:
    """Write JSON report to file."""
    report_dict = {
        "model_profile": report.model_profile,
        "fusion_method": report.fusion_method,
        "sparse_strategy": report.sparse_strategy,
        "total_queries": report.total_queries,
        "total_documents_indexed": report.total_documents_indexed,
        "queries_with_answers": report.queries_with_answers,
        "average_latency_seconds": report.average_latency_seconds,
        "average_retrieval_score": report.average_retrieval_score,
        "timestamp": report.timestamp,
        "results": [
            {
                "query": r.query,
                "question_type": r.question_type,
                "expected_topics": r.expected_topics,
                "num_retrieved": r.num_retrieved,
                "retrieved_topics": r.retrieved_topics,
                "has_answer": r.has_answer,
                "answer_length": r.answer_length,
                "model_used": r.model_used,
                "total_latency_seconds": r.total_latency_seconds,
                "retrieval_score": r.retrieval_score,
                "stage_timings": [
                    {"stage": t.stage_name, "duration_seconds": t.duration_seconds}
                    for t in r.timings
                ]
            }
            for r in report.results
        ]
    }
    with open(path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nJSON report written to: {path}")


async def main() -> None:
    args = parse_args()

    try:
        report = await run_benchmark(args)
        print_report(report)

        if args.output_json:
            write_json_report(report, args.output_json)

    except Exception as e:
        print(f"\nERROR: Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
