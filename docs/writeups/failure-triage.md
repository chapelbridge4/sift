<!-- FOR THE AUTHOR — NOT FOR AUTO-POSTING
     This is a DRAFT for publication on a dev blog (e.g. dev.to or a personal site),
     with cross-posts to r/LocalLLaMA and Show HN alongside a demo GIF.
     This is the traction lever for sift v1.1. DO NOT auto-post. Review the
     numbers, add the GIF link, then publish manually when you are ready.
-->

# Why Your RAG Pipeline Failed (and Which Stage to Blame)

Most RAG observability tools tell you *that* a query failed — the score was low, the answer was wrong — but not *why*. Was the relevant document never retrieved? Did the reranker bury it? Did the generator ignore it entirely? Without stage-level attribution, debugging a RAG system is a game of whack-a-mole.

**sift** is a local-first RAG toolkit built around a per-query failure-triage classifier. The core idea: every query execution generates a `QueryTrace`; a lightweight classifier turns that trace into a typed `RAGFailureType` verdict, pinpointing the earliest pipeline stage that broke.

## The Problem with "the score was low"

A single quality score hides the structure of the failure. Consider two queries that both produce a wrong answer:

- Query A: the gold document simply was not in the top-10 retrieved candidates.
- Query B: the gold document was retrieved at rank 3, but the cross-encoder reranker pushed it to rank 12, out of the generator's context window.

The remediation for A is "improve retrieval" — try hybrid dense+sparse, query expansion, larger top-k. The remediation for B is "audit the reranker" — add a score floor, domain-fine-tune, or check for diversity collapse. Treating them as the same problem wastes engineering time.

## How sift's Triage Classifier Works

The classifier operates on four ordered stages: `chunking`, `retrieval`, `reranking`, and `generation`. At each stage, deterministic boolean signals are extracted from the `QueryTrace`:

- **`recall_hit`** — did any gold document appear in the top-k retrieved candidates?
- **`reranker_moved_gold_down`** — did the cross-encoder reranker worsen the gold document's rank compared to retrieval order?
- **`answer_present`** — did the generator produce any non-empty output?
- **`answer_grounded`** — do the answer's content-word tokens have sufficient lexical overlap with the retrieved texts (threshold: 30%)?

Rules run in priority order — retrieval miss before reranking demotion before generation — so the earliest broken stage is reported as root cause. The taxonomy covers 16 `RAGFailureType` members across the four stages. The two types measurable with retrieval and reranking signals alone are:

- **`RELEVANT_NOT_RETRIEVED`** (retrieval stage): `recall_hit is False`. The relevant document exists in the corpus but never entered the candidate set. Fix hint: increase top-k, try hybrid retrieval, or use query expansion to bridge the vocabulary gap.
- **`RELEVANT_DEMOTED`** (reranking stage): `reranker_moved_gold_down is True`. Retrieval found the gold document; the reranker buried it below the context window cutoff. Fix hint: audit reranker scores on known-good pairs; consider a score floor preventing high-recall candidates from being dropped.

Generation-stage subtypes (`UNFAITHFUL`, `INCOMPLETE`, `CONTEXT_IGNORED`) are detected deterministically when answers are available, or by an optional local LLM judge for finer disambiguation. The judge is disabled by default and only loads a model when `TRIAGE_USE_LLM_JUDGE=true` is set — it never runs at import time or during benchmarks.

## Real Numbers

Benchmarks are full-test-set runs against true BEIR qrels (not sampled), using `sentence-transformers/all-MiniLM-L6-v2` via fastembed and an in-process Qdrant instance. No Docker required.

**SciFact** (300 queries, 5,183-document corpus):

| Metric | Value |
|---|---|
| Recall@10 | 0.774 |
| nDCG@10 | 0.624 |
| MRR | 0.578 |
| Avg latency | ~16 ms/query |

**NFCorpus** (323 queries, 3,633-document corpus — hard medical domain):

| Metric | Value |
|---|---|
| Recall@10 | 0.154 |
| nDCG@10 | 0.317 |
| MRR | 0.511 |

NFCorpus is structurally difficult: many relevant documents per query, so recall@10 is low across all dense retrievers on this dataset — expected and documented in the BEIR paper. MRR of 0.511 means a relevant document still appears near the top; the problem is coverage, not ordering.

**Triage distribution** (100-query SciFact run, retrieval + reranking signals active, generation not exercised):

| Outcome | Count |
|---|---|
| Passed | 74 (74%) |
| Failed | 26 (26%) |

| Failure type | Stage | Count | % of failures |
|---|---|---|---|
| `RELEVANT_NOT_RETRIEVED` | retrieval | 18 | 69.2% |
| `RELEVANT_DEMOTED` | reranking | 8 | 30.8% |

Both stages show up in the distribution — that is the point. Eight queries that retrieval *did* handle correctly were then broken by the cross-encoder reranker. Those would be invisible to a retrieval-only evaluation.

## Honest Limits

- **Local-first on 8 GB M1.** Qdrant (in-process), fastembed, and the cross-encoder reranker all fit in unified memory. Generation-stage triage requires a downloaded GGUF model — not exercised in the committed sample. With a local model, `--with-answers` enables generation signals and the optional LLM judge.
- **Single dense embedder.** All results use MiniLM-L6-v2. Hybrid retrieval (BM25 + dense) is not yet wired; the numbers reflect this single configuration.
- **Signals scope the distribution.** Chunking-stage failures (`CHUNK_TOO_LARGE`, `SEMANTIC_SPLIT`, etc.) are in the taxonomy but require chunk-level metadata not yet collected. The committed distribution covers only what the live signals can see: retrieval recall and reranker rank movement.
- **No sampling, no projections.** Benchmark numbers are full test-set runs with deterministic qrels. The triage run is 100 queries, not the full 300.

## What This Is For

sift is a diagnostic layer for iterative RAG development on a laptop: ingest a corpus, benchmark, read the failure distribution, fix the right stage, repeat. The triage classifier converts "my recall is low" into "18 of 26 failures are retrieval misses — start there."

Taxonomy, signal definitions, and classifier rules are in `app/triage/`. Benchmark runner in `scripts/`. No cloud account needed.
