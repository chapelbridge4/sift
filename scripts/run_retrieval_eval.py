#!/usr/bin/env python3
"""SMOKE TEST (not a benchmark): verifies the embedded-Qdrant + embedding + dense-search plumbing runs with zero external services. The corpus is synthetic and keyword-aligned, so the recall number is a plumbing check only — do NOT cite it. Real retrieval numbers: scripts/benchmark_beir.py.

Usage:
    .venv/bin/python scripts/run_retrieval_eval.py
    .venv/bin/python scripts/run_retrieval_eval.py --top-k 5
    .venv/bin/python scripts/run_retrieval_eval.py --fixture tests/fixtures/rag_queries.json

Measures:
    recall@k  (keyword-based proxy: fraction of expected keywords found in top-k retrieved texts)

Exits 0 if avg recall >= 0.5, else 1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    Modifier,
)
from fastembed import TextEmbedding, SparseTextEmbedding


# ---------------------------------------------------------------------------
# Synthetic corpus — one passage per fixture query topic, keyed to keywords
# ---------------------------------------------------------------------------

CORPUS: List[Dict[str, Any]] = [
    {
        "id": "doc-attn-1",
        "text": (
            "The attention mechanism in transformers uses query, key, and value matrices. "
            "Scaled dot-product attention computes similarity scores via softmax over "
            "dot products of query and key vectors, then weights the value vectors. "
            "Multi-head attention runs multiple attention heads in parallel, allowing the "
            "model to jointly attend to information from different representation subspaces."
        ),
        "topics": ["attention mechanism", "transformer architecture"],
    },
    {
        "id": "doc-attn-2",
        "text": (
            "Self-attention allows each position in a sequence to attend to all other "
            "positions in the same layer. The softmax over scaled dot-products produces "
            "attention weights that determine how much each value vector contributes to "
            "the output. Query-key-value (QKV) decomposition is central to transformer models."
        ),
        "topics": ["attention mechanism", "transformer architecture"],
    },
    {
        "id": "doc-train-1",
        "text": (
            "Training stability in deep learning relies on careful learning rate warmup "
            "schedules and gradient clipping. Optimizer choices such as Adam and SGD "
            "affect loss convergence. Layer normalization and batch normalization help "
            "stabilize gradients during training of large models."
        ),
        "topics": ["training stability", "optimization"],
    },
    {
        "id": "doc-train-2",
        "text": (
            "Gradient explosion and vanishing gradients are common instabilities during "
            "training. Normalization techniques applied before or after residual connections "
            "improve stability. A cosine learning rate schedule with linear warmup is "
            "widely used for training transformer language models."
        ),
        "topics": ["training stability", "optimization"],
    },
    {
        "id": "doc-compress-1",
        "text": (
            "Model compression encompasses quantization, pruning, and knowledge distillation. "
            "Quantization reduces weight precision (e.g. INT8 or INT4) to shrink model size. "
            "Structured pruning removes entire attention heads or layers. Knowledge distillation "
            "trains a smaller student model to mimic a larger teacher model."
        ),
        "topics": ["model compression", "efficiency"],
    },
    {
        "id": "doc-compress-2",
        "text": (
            "Post-training quantization converts full-precision weights to lower precision "
            "without retraining. Distillation-based compression transfers soft labels from "
            "the teacher to the student. Compression techniques are essential for deploying "
            "large language models on resource-constrained devices."
        ),
        "topics": ["model compression", "efficiency"],
    },
    {
        "id": "doc-eval-1",
        "text": (
            "LLM evaluation uses benchmarks such as MMLU, HELM, BIG-bench, GLUE, and SuperGLUE. "
            "MMLU tests knowledge across 57 academic subjects. HELM provides holistic evaluation "
            "across accuracy, calibration, robustness, and fairness metrics. BIG-bench probes "
            "capabilities beyond standard benchmarks."
        ),
        "topics": ["LLM evaluation", "benchmarks"],
    },
    {
        "id": "doc-eval-2",
        "text": (
            "SuperGLUE and GLUE are classic natural language understanding benchmarks. "
            "More recent benchmarks like MMLU and BIG-bench are designed to test reasoning "
            "and world knowledge in large language models. Benchmark design requires careful "
            "consideration of contamination and task diversity."
        ),
        "topics": ["LLM evaluation", "benchmarks"],
    },
    {
        "id": "doc-scale-1",
        "text": (
            "Scaling laws describe how model performance improves as a power law function of "
            "compute, parameters, and dataset size. The Chinchilla scaling laws showed that "
            "both model size and training tokens should scale together for compute-optimal "
            "training. Emergent abilities appear at certain scale thresholds."
        ),
        "topics": ["scaling laws", "compute optimal"],
    },
    {
        "id": "doc-scale-2",
        "text": (
            "Neural network scaling follows power law relationships between loss and compute "
            "budget. Larger parameter counts combined with proportionally larger datasets "
            "yield the best compute-optimal models. Emergent capabilities such as in-context "
            "learning appear unpredictably at scale."
        ),
        "topics": ["scaling laws", "compute optimal"],
    },
]


def load_fixture(fixture_path: str) -> List[Dict[str, Any]]:
    with open(fixture_path) as f:
        data = json.load(f)
    return data.get("queries", [])


async def build_in_memory_collection(
    dense_model: TextEmbedding,
    dense_dim: int,
) -> AsyncQdrantClient:
    """
    Create an in-memory Qdrant collection and index CORPUS into it.
    Returns the populated client.
    """
    client = AsyncQdrantClient(location=":memory:")

    await client.create_collection(
        "eval_collection",
        vectors_config={"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)},
    )

    # Embed all corpus texts
    texts = [doc["text"] for doc in CORPUS]
    dense_vecs = list(dense_model.embed(texts))

    points = []
    for i, (doc, dense_vec) in enumerate(zip(CORPUS, dense_vecs)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": dense_vec.tolist()},
                payload={
                    "text": doc["text"],
                    "doc_id": doc["id"],
                    "topics": doc["topics"],
                },
            )
        )

    await client.upsert("eval_collection", points=points, wait=True)
    return client


async def dense_search(
    client: AsyncQdrantClient,
    dense_model: TextEmbedding,
    query_text: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Run dense-only search and return list of result dicts."""
    query_vec = list(dense_model.embed([query_text]))[0].tolist()
    hits = await client.search(
        "eval_collection",
        query_vector=("dense", query_vec),
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "score": h.score,
            "text": h.payload.get("text", ""),
            "doc_id": h.payload.get("doc_id", ""),
            "topics": h.payload.get("topics", []),
        }
        for h in hits
    ]


def compute_keyword_recall(retrieved_texts: List[str], expected_keywords: List[str]) -> float:
    """Fraction of expected keywords present in combined retrieved text."""
    if not expected_keywords:
        return 0.0
    combined = " ".join(retrieved_texts).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in combined)
    return hits / len(expected_keywords)


async def run_eval(fixture_path: str, top_k: int) -> Dict[str, Any]:
    queries = load_fixture(fixture_path)
    if not queries:
        print(f"ERROR: no queries in {fixture_path}")
        sys.exit(1)

    print(f"Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    t0 = time.perf_counter()
    dense_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Determine dim from a test embed
    dim = len(list(dense_model.embed(["dim probe"]))[0])
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Model loaded in {load_ms:.0f}ms — dense dim={dim}")

    print(f"Building in-memory Qdrant collection ({len(CORPUS)} documents)...")
    t0 = time.perf_counter()
    client = await build_in_memory_collection(dense_model, dim)
    index_ms = (time.perf_counter() - t0) * 1000
    print(f"  Indexed {len(CORPUS)} docs in {index_ms:.0f}ms")

    print(f"\nRunning retrieval eval: {len(queries)} queries, top_k={top_k}")
    print("-" * 60)

    per_query = []
    for q in queries:
        qid = q["id"]
        query_text = q["query"]
        expected_keywords = q.get("expected_keywords", [])

        t0 = time.perf_counter()
        results = await dense_search(client, dense_model, query_text, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        retrieved_texts = [r["text"] for r in results]
        recall = compute_keyword_recall(retrieved_texts, expected_keywords)

        per_query.append(
            {
                "query_id": qid,
                "query": query_text,
                "top_k": top_k,
                "retrieved_count": len(results),
                "recall": round(recall, 3),
                "retrieval_ms": round(retrieval_ms, 1),
                "expected_keywords": expected_keywords,
            }
        )
        kw_hits = sum(1 for kw in expected_keywords if kw.lower() in " ".join(retrieved_texts).lower())
        print(
            f"  {qid}: recall={recall:.3f}  "
            f"({kw_hits}/{len(expected_keywords)} keywords)  "
            f"latency={retrieval_ms:.1f}ms"
        )

    avg_recall = sum(r["recall"] for r in per_query) / len(per_query)
    avg_latency = sum(r["retrieval_ms"] for r in per_query) / len(per_query)

    print("-" * 60)
    print(f"  avg recall@{top_k}: {avg_recall:.3f}")
    print(f"  avg latency: {avg_latency:.1f}ms")
    print(f"  corpus size: {len(CORPUS)} docs")

    return {
        "avg_recall": round(avg_recall, 3),
        "avg_latency_ms": round(avg_latency, 1),
        "top_k": top_k,
        "corpus_size": len(CORPUS),
        "per_query": per_query,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone retrieval eval with embedded Qdrant")
    p.add_argument(
        "--fixture",
        default="tests/fixtures/rag_queries.json",
        help="Path to eval fixtures JSON (default: tests/fixtures/rag_queries.json)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve per query (default: 5)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fixture_path = Path(args.fixture)
    if not fixture_path.is_absolute():
        fixture_path = Path(__file__).parent.parent / fixture_path
    if not fixture_path.exists():
        print(f"ERROR: fixture not found: {fixture_path}")
        sys.exit(1)

    summary = asyncio.run(run_eval(str(fixture_path), top_k=args.top_k))

    avg_recall = summary["avg_recall"]
    if avg_recall >= 0.5:
        print(f"\nSMOKE OK: retrieval plumbing works (synthetic corpus — not a quality metric)")
        sys.exit(0)
    else:
        print(f"\nWARN: avg recall@{args.top_k} = {avg_recall:.3f} (< 0.5 threshold)")
        sys.exit(1)


if __name__ == "__main__":
    main()
