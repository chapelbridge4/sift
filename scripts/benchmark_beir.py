#!/usr/bin/env python3
"""Real retrieval benchmark on BEIR/SciFact using true recall@k vs qrels.

Downloads SciFact from the public BEIR URL (once; cached under ./datasets/scifact),
embeds the corpus with fastembed MiniLM-L6, indexes into an in-memory Qdrant
collection, then measures recall@k against ground-truth relevance judgments.

Usage:
    .venv/bin/python scripts/benchmark_beir.py
    .venv/bin/python scripts/benchmark_beir.py --top-k 10 --max-queries 100
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Dict, Set

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
CACHE_DIR = Path(__file__).parent.parent / "datasets"
SCIFACT_DIR = CACHE_DIR / "scifact"
COLLECTION_NAME = "scifact_dense"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Pure function — tested in isolation
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_ids: list[str], relevant_ids: Set[str], k: int) -> float:
    """Return |top-k ∩ relevant| / |relevant|.  Returns 0.0 if relevant is empty."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


# ---------------------------------------------------------------------------
# Download & parse
# ---------------------------------------------------------------------------


def download_scifact() -> None:
    """Download the SciFact zip to CACHE_DIR and unzip, if not already cached."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "scifact.zip"

    if SCIFACT_DIR.exists() and (SCIFACT_DIR / "corpus.jsonl").exists():
        print(f"  SciFact already cached at {SCIFACT_DIR}")
        return

    print(f"  Downloading SciFact from {SCIFACT_URL} ...")
    t0 = time.perf_counter()
    urllib.request.urlretrieve(SCIFACT_URL, zip_path)
    elapsed = time.perf_counter() - t0
    size_mb = zip_path.stat().st_size / 1_000_000
    print(f"  Downloaded {size_mb:.1f} MB in {elapsed:.1f}s")

    print(f"  Unzipping to {CACHE_DIR} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(CACHE_DIR)
    print("  Done.")


def parse_corpus() -> Dict[str, str]:
    """Parse corpus.jsonl → {doc_id: title + ' ' + text}."""
    corpus: Dict[str, str] = {}
    with open(SCIFACT_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = str(obj["_id"])
            title = obj.get("title", "")
            text = obj.get("text", "")
            corpus[doc_id] = (title + " " + text).strip()
    return corpus


def parse_queries() -> Dict[str, str]:
    """Parse queries.jsonl → {qid: text}."""
    queries: Dict[str, str] = {}
    with open(SCIFACT_DIR / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            queries[str(obj["_id"])] = obj["text"]
    return queries


def parse_qrels() -> Dict[str, Set[str]]:
    """Parse qrels/test.tsv → {qid: set(doc_id)} where score > 0."""
    qrels: Dict[str, Set[str]] = {}
    tsv_path = SCIFACT_DIR / "qrels" / "test.tsv"
    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        # Skip header
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            qid, corpus_id, score_str = row[0], row[1], row[2]
            try:
                score = int(score_str)
            except ValueError:
                continue
            if score > 0:
                qrels.setdefault(qid, set()).add(corpus_id)
    return qrels


# ---------------------------------------------------------------------------
# Async benchmark
# ---------------------------------------------------------------------------


async def run(top_k: int = 10, max_queries: int = 100) -> None:
    print("=== BEIR/SciFact Dense Retrieval Benchmark ===")

    # 1. Download / verify cache
    print("\n[1/4] Data")
    download_scifact()
    corpus = parse_corpus()
    queries = parse_queries()
    qrels = parse_qrels()
    print(f"  corpus={len(corpus)} docs, queries={len(queries)}, qrels qids={len(qrels)}")

    # 2. Load embedding model
    print(f"\n[2/4] Embedding model ({MODEL_NAME})")
    t0 = time.perf_counter()
    model = TextEmbedding(model_name=MODEL_NAME)
    # Probe dimension
    dim = len(list(model.embed(["probe"]))[0])
    print(f"  Loaded in {(time.perf_counter() - t0)*1000:.0f}ms — dim={dim}")

    # 3. Index corpus into in-memory Qdrant
    print(f"\n[3/4] Indexing {len(corpus)} documents into in-memory Qdrant")
    t0 = time.perf_counter()
    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
    )

    # Build ordered lists for batch embedding
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]

    # Embed in one pass (fastembed handles batching internally)
    dense_vecs = list(model.embed(doc_texts))

    points = []
    for did, vec in zip(doc_ids, dense_vecs):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": vec.tolist()},
                payload={"doc_id": did},
            )
        )

    # Upsert in batches of 512 to keep memory bounded
    batch_size = 512
    for i in range(0, len(points), batch_size):
        await client.upsert(COLLECTION_NAME, points=points[i : i + batch_size], wait=True)

    index_ms = (time.perf_counter() - t0) * 1000
    print(f"  Indexed in {index_ms:.0f}ms")

    # 4. Evaluate recall@k against qrels
    print(f"\n[4/4] Evaluating recall@{top_k} (max_queries={max_queries})")

    # Only evaluate queries that have qrels entries
    eval_qids = [qid for qid in queries if qid in qrels][:max_queries]
    print(f"  Queries with qrels: {len([qid for qid in queries if qid in qrels])}, using {len(eval_qids)}")

    recalls: list[float] = []
    latencies: list[float] = []

    for qid in eval_qids:
        query_text = queries[qid]
        relevant = qrels[qid]

        t0 = time.perf_counter()
        query_vec = list(model.embed([query_text]))[0].tolist()
        hits = (await client.query_points(
            COLLECTION_NAME,
            query=query_vec,
            using="dense",
            limit=top_k,
            with_payload=True,
        )).points
        query_ms = (time.perf_counter() - t0) * 1000
        latencies.append(query_ms)

        retrieved_ids = [h.payload["doc_id"] for h in hits]
        r = recall_at_k(retrieved_ids, relevant, k=top_k)
        recalls.append(r)

    avg_recall = sum(recalls) / len(recalls)
    avg_ms = sum(latencies) / len(latencies)

    headline = (
        f"SciFact dense (MiniLM-L6) recall@{top_k} = {avg_recall:.3f} "
        f"over {len(recalls)} queries, {avg_ms:.0f}ms/query, corpus={len(corpus)}"
    )
    print(f"\n{headline}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BEIR/SciFact real retrieval benchmark")
    p.add_argument("--top-k", type=int, default=10, help="Retrieve top-k docs per query")
    p.add_argument(
        "--max-queries", type=int, default=100, help="Max queries to evaluate (default 100)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(top_k=args.top_k, max_queries=args.max_queries))
