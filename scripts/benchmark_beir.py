#!/usr/bin/env python3
"""Real retrieval benchmark on BEIR datasets using recall@k, nDCG@10, and MRR.

Downloads BEIR datasets from the public UKP URL (once; cached under ./datasets/<dataset>),
embeds the corpus with fastembed MiniLM-L6, indexes into an in-memory Qdrant
collection, then measures recall@k, nDCG@10, and MRR against ground-truth relevance
judgments.

Usage:
    .venv/bin/python scripts/benchmark_beir.py
    .venv/bin/python scripts/benchmark_beir.py --dataset nfcorpus --full --out reports/retrieval/nfcorpus.json
    .venv/bin/python scripts/benchmark_beir.py --top-k 10 --max-queries 100
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
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

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
CACHE_DIR = Path(__file__).parent.parent / "datasets"
COLLECTION_NAME = "beir_dense"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Pure metrics — tested in isolation
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_ids: list[str], relevant_ids: Set[str], k: int) -> float:
    """Return |top-k ∩ relevant| / |relevant|.  Returns 0.0 if relevant is empty."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: Set[str], k: int) -> float:
    """Discounted Cumulative Gain at k with binary relevance, normalised by ideal DCG."""
    if not relevant_ids:
        return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: Set[str]) -> float:
    """Return 1/(rank of first relevant doc), or 0.0 if none found."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Download & parse (dataset-agnostic)
# ---------------------------------------------------------------------------


def _dataset_dir(dataset: str) -> Path:
    return CACHE_DIR / dataset


def download_dataset(dataset: str) -> None:
    """Download a BEIR dataset zip to CACHE_DIR and unzip, if not already cached."""
    dataset_dir = _dataset_dir(dataset)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / f"{dataset}.zip"

    if dataset_dir.exists() and (dataset_dir / "corpus.jsonl").exists():
        print(f"  {dataset} already cached at {dataset_dir}")
        return

    url = f"{BEIR_BASE_URL}/{dataset}.zip"
    print(f"  Downloading {dataset} from {url} ...")
    t0 = time.perf_counter()
    urllib.request.urlretrieve(url, zip_path)
    elapsed = time.perf_counter() - t0
    size_mb = zip_path.stat().st_size / 1_000_000
    print(f"  Downloaded {size_mb:.1f} MB in {elapsed:.1f}s")

    print(f"  Unzipping to {CACHE_DIR} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(CACHE_DIR)
    zip_path.unlink(missing_ok=True)
    print("  Done.")


def parse_corpus(dataset: str = "scifact") -> Dict[str, str]:
    """Parse corpus.jsonl → {doc_id: title + ' ' + text}."""
    corpus: Dict[str, str] = {}
    corpus_path = _dataset_dir(dataset) / "corpus.jsonl"
    with open(corpus_path, encoding="utf-8") as f:
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


def parse_queries(dataset: str = "scifact") -> Dict[str, str]:
    """Parse queries.jsonl → {qid: text}."""
    queries: Dict[str, str] = {}
    queries_path = _dataset_dir(dataset) / "queries.jsonl"
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            queries[str(obj["_id"])] = obj["text"]
    return queries


def parse_qrels(dataset: str = "scifact") -> Dict[str, Set[str]]:
    """Parse qrels/test.tsv → {qid: set(doc_id)} where score > 0."""
    qrels: Dict[str, Set[str]] = {}
    tsv_path = _dataset_dir(dataset) / "qrels" / "test.tsv"
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


async def run(
    top_k: int = 10,
    max_queries: int = 100,
    dataset: str = "scifact",
    full: bool = False,
    assert_recall_gte: float | None = None,
    out: Path | None = None,
) -> None:
    print(f"=== BEIR/{dataset} Dense Retrieval Benchmark ===")

    # 1. Download / verify cache
    print("\n[1/4] Data")
    download_dataset(dataset)
    corpus = parse_corpus(dataset)
    queries = parse_queries(dataset)
    qrels = parse_qrels(dataset)
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

    # 4. Evaluate metrics against qrels
    # Only evaluate queries that have qrels entries
    eval_qids_all = [qid for qid in queries if qid in qrels]
    if full:
        eval_qids = eval_qids_all
        print(f"\n[4/4] Evaluating recall@{top_k}, nDCG@{top_k}, MRR (FULL test set: {len(eval_qids)} queries)")
    else:
        eval_qids = eval_qids_all[:max_queries]
        print(
            f"\n[4/4] Evaluating recall@{top_k}, nDCG@{top_k}, MRR "
            f"(max_queries={max_queries}, using {len(eval_qids)} of {len(eval_qids_all)} queries with qrels)"
        )

    recalls: list[float] = []
    ndcgs: list[float] = []
    rrs: list[float] = []
    latencies: list[float] = []

    for qid in eval_qids:
        query_text = queries[qid]
        relevant = qrels[qid]

        t0 = time.perf_counter()
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
        query_ms = (time.perf_counter() - t0) * 1000
        latencies.append(query_ms)

        retrieved_ids = [h.payload["doc_id"] for h in hits]
        recalls.append(recall_at_k(retrieved_ids, relevant, k=top_k))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant, k=top_k))
        rrs.append(reciprocal_rank(retrieved_ids, relevant))

    n_queries = len(recalls)
    if n_queries == 0:
        sys.exit("No queries to evaluate")

    avg_recall = sum(recalls) / n_queries
    avg_ndcg = sum(ndcgs) / n_queries
    avg_mrr = sum(rrs) / n_queries
    avg_ms = sum(latencies) / n_queries

    headline = (
        f"{dataset} dense(MiniLM-L6): recall@{top_k}={avg_recall:.3f} "
        f"nDCG@{top_k}={avg_ndcg:.3f} MRR={avg_mrr:.3f} over {n_queries} queries"
    )
    print(f"\n{headline}")
    print(f"  corpus={len(corpus)} docs, {avg_ms:.0f}ms/query")

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "dataset": dataset,
            "model": MODEL_NAME,
            "n_queries": n_queries,
            f"recall_at_{top_k}": round(avg_recall, 6),
            f"ndcg_at_{top_k}": round(avg_ndcg, 6),
            "mrr": round(avg_mrr, 6),
            "top_k": top_k,
            "avg_latency_ms": round(avg_ms, 2),
            "corpus_size": len(corpus),
        }
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"  Report written to {out}")

    if assert_recall_gte is not None:
        if avg_recall < assert_recall_gte:
            print(
                f"\nASSERTION FAILED: recall@{top_k}={avg_recall:.3f} < {assert_recall_gte}",
                file=sys.stderr,
            )
            sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BEIR dense retrieval benchmark (recall@k, nDCG@k, MRR)")
    p.add_argument(
        "--dataset",
        default="scifact",
        choices=["scifact", "nfcorpus"],
        help="BEIR dataset to benchmark (default: scifact)",
    )
    p.add_argument("--top-k", type=int, default=10, help="Retrieve top-k docs per query")
    p.add_argument(
        "--max-queries",
        type=int,
        default=100,
        help="Max queries to evaluate; ignored when --full is set (default: 100)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Use ALL qrel test queries (ignores --max-queries)",
    )
    p.add_argument(
        "--assert-recall-gte",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Exit with code 1 if avg recall@k is below this threshold",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write JSON report to this path (parent dirs created automatically)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run(
            top_k=args.top_k,
            max_queries=args.max_queries,
            dataset=args.dataset,
            full=args.full,
            assert_recall_gte=args.assert_recall_gte,
            out=args.out,
        )
    )
