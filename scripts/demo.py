#!/usr/bin/env python3
"""Zero-config local demo for sift — retrieval + per-query failure triage.

No Docker, no LLM download, no cloud account.  Runs in seconds.
The ~90 MB fastembed embedder downloads once on first use; no multi-GB GGUF/LLM is ever pulled.

What it does
------------
1. Loads the three sample documents from ``examples/corpus/``.
2. Embeds and indexes them into an in-memory Qdrant collection using
   fastembed MiniLM-L6 (~90 MB, fetched once on first use; no LLM/GGUF pulled).
3. Runs a retrieval query and prints the top matching chunk.
4. Constructs a ``QueryTrace`` for a deliberately FAILING query — one whose
   gold document is intentionally absent from this small corpus — and prints
   the full triage verdict so the recruiter can see the classifier in action.

Usage
-----
    .venv/bin/python scripts/demo.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# Make project root importable when running as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

# Lazy imports: nothing heavy at module load — embedder loads on first call.
from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.triage.classifier import classify
from app.triage.signals import QueryTrace

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CORPUS_DIR = Path(__file__).parent.parent / "examples" / "corpus"
_COLLECTION = "demo"
_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_TOP_K = 3

# The "gold" ID for the failing-query demo is a document that does NOT exist
# in this small corpus, so triage fires RELEVANT_NOT_RETRIEVED.
_FAILING_QUERY = "What is speculative decoding and how does it speed up autoregressive generation?"
_FAILING_GOLD_ID = "speculative_decoding"  # absent from examples/corpus/


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------


def _load_corpus(corpus_dir: Path) -> dict[str, str]:
    """Return {stem: full_text} for every .txt file under corpus_dir."""
    docs: dict[str, str] = {}
    for path in sorted(corpus_dir.glob("*.txt")):
        docs[path.stem] = path.read_text(encoding="utf-8").strip()
    return docs


# ---------------------------------------------------------------------------
# Async demo body
# ---------------------------------------------------------------------------


async def _run_demo() -> None:
    print("=" * 60)
    print("sift — local-first RAG with per-query failure triage")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load corpus
    # ------------------------------------------------------------------
    print("\n[1/4] Loading sample corpus from examples/corpus/ ...")
    corpus = _load_corpus(_CORPUS_DIR)
    if not corpus:
        print(f"  ERROR: no .txt files found under {_CORPUS_DIR}")
        sys.exit(1)
    for name, text in corpus.items():
        n_lines = text.count("\n") + 1
        print(f"  {name}.txt  ({n_lines} lines)")

    # ------------------------------------------------------------------
    # 2. Embed corpus (lazy load — no download when model is cached)
    # ------------------------------------------------------------------
    print(f"\n[2/4] Embedding with {_MODEL} ...")
    t0 = time.perf_counter()
    model = TextEmbedding(model_name=_MODEL)
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]
    vecs = list(model.embed(doc_texts))
    dim = len(vecs[0])
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  dim={dim}  docs={len(doc_ids)}  elapsed={elapsed:.0f}ms")

    # ------------------------------------------------------------------
    # 3. Index into in-memory Qdrant (no Docker, no persistence)
    # ------------------------------------------------------------------
    print("\n[3/4] Indexing into in-memory Qdrant ...")
    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        _COLLECTION,
        vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
    )
    points = [
        PointStruct(
            id=i,
            vector={"dense": vec.tolist()},
            payload={"doc_id": did, "text": corpus[did]},
        )
        for i, (did, vec) in enumerate(zip(doc_ids, vecs))
    ]
    await client.upsert(_COLLECTION, points=points, wait=True)
    print(f"  indexed {len(points)} document(s)")

    # ------------------------------------------------------------------
    # 4a. Retrieval — a query that SHOULD match
    # ------------------------------------------------------------------
    success_query = "How does dense retrieval use embeddings to find relevant documents?"
    print("\n[4/4] Retrieval demo")
    print(f"\n  Query: \"{success_query}\"")

    q_vec = list(model.embed([success_query]))[0].tolist()
    hits = (
        await client.query_points(
            _COLLECTION,
            query=q_vec,
            using="dense",
            limit=_TOP_K,
            with_payload=True,
        )
    ).points

    print(f"\n  Top-{_TOP_K} retrieved chunks:")
    for rank, hit in enumerate(hits, 1):
        doc_id = hit.payload["doc_id"]
        score = hit.score
        # Print first sentence of the document text as a preview.
        preview = hit.payload["text"].split(".")[0].strip() + "."
        print(f"    #{rank}  [{doc_id}]  score={score:.4f}")
        print(f"        {preview}")

    # ------------------------------------------------------------------
    # 4b. Triage — deliberately FAILING query (gold not in corpus)
    # ------------------------------------------------------------------
    print("\n  Triage demo (intentionally failing query)")
    print(f"  Query:   \"{_FAILING_QUERY}\"")
    print(f"  Gold ID: \"{_FAILING_GOLD_ID}\"  (absent from corpus — retrieval must miss)")

    fq_vec = list(model.embed([_FAILING_QUERY]))[0].tolist()
    fq_hits = (
        await client.query_points(
            _COLLECTION,
            query=fq_vec,
            using="dense",
            limit=_TOP_K,
            with_payload=True,
        )
    ).points

    retrieved = [
        {"doc_id": h.payload["doc_id"], "score": h.score, "text": h.payload["text"]}
        for h in fq_hits
    ]

    trace = QueryTrace(
        query=_FAILING_QUERY,
        retrieved=retrieved,
        gold_ids={_FAILING_GOLD_ID},
        reranked=None,
        answer=None,
        top_k=_TOP_K,
    )

    verdict = classify(trace)

    print("\n  Triage verdict:")
    if verdict.failure_types:
        for ft, conf in verdict.failure_types:
            print(f"    failure_type  : {ft.name}")
            print(f"    stage         : {ft.stage}")
            print(f"    confidence    : {conf:.2f}")
            print(f"    fix_hint      : {ft.fix_hint}")
    else:
        print("    (no failure detected)")
    print(f"    primary_stage : {verdict.primary_stage}")
    print(f"    evidence      : {verdict.evidence}")

    print("\n" + "=" * 60)
    print("Done — no LLM download, no Docker, no cloud.  Time to add your corpus.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    asyncio.run(_run_demo())


if __name__ == "__main__":
    main()
