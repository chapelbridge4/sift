# BENCHMARKS.md — sift flagship v0.1

Real numbers only. No projected or aspirational values.

---

## Hardware

Apple M1, 8 GB unified memory; llama.cpp via llama-cpp-python (Metal); embedded Qdrant.

- Model: `Qwen3-4B-Q4_K_M.gguf` (unsloth/Qwen3-4B-GGUF, ~2.5 GB on disk)
- Backend: llama-cpp-python with `n_gpu_layers=-1` (full Metal offload), `flash_attn=True`
- Retrieval: embedded Qdrant (in-process, no server)

---

## Retrieval Quality

Method: dense vector search (cosine similarity, fastembed MiniLM-L6), embedded Qdrant.
Metrics: recall@10, nDCG@10, MRR against BEIR true qrels (not surrogate scores).
Coverage: full test set, deterministic (no subsampling).

| Encoder                                 | Dataset   | recall@10 | nDCG@10 | MRR   | #queries | Corpus | Latency      | Report |
|-----------------------------------------|-----------|----------:|--------:|------:|---------:|-------:|-------------:|--------|
| sentence-transformers/all-MiniLM-L6-v2  | SciFact   | 0.774     | 0.624   | 0.578 | 300      | 5 183  | 16 ms/query  | [scifact.json](reports/retrieval/scifact.json) |
| sentence-transformers/all-MiniLM-L6-v2  | NFCorpus  | 0.154     | 0.317   | 0.511 | 323      | 3 633  | 16 ms/query  | [nfcorpus.json](reports/retrieval/nfcorpus.json) |

> NFCorpus has many relevant documents per query (medical literature), so recall@10 is
> structurally low — MRR better reflects ranking quality on this dataset.
>
> recall@10 is a retrieval-only metric. It is independent of the generation KV-cache
> settings and does not change across the combos below.

Measured: 2026-06-16, feat/sift-v1.1-hardening branch, full test sets.

---

## Generation — KV-Cache Quantization Matrix

Measured on M1 8 GB. Each combo ran in a fresh subprocess (model fully reloaded) to prevent
cross-run memory contamination. Workload: fixed 64-token generation from a single prompt,
1-token Metal warmup before timing.

| KV-K  | KV-V  | tok/s | Peak RSS | n_ctx | Runs? |
|-------|-------|------:|--------:|------:|-------|
| f16   | f16   | 19.49 |  2.913 GB | 4096  | yes   |
| q8_0  | q4_0  | 10.01 |  2.699 GB | 4096  | yes   |
| q4_0  | q4_0  | 19.13 |  2.630 GB | 4096  | yes   |

**Observations:**

- All three combos fit comfortably in 8 GB at n_ctx=4096 (~2.6–2.9 GB RSS).
- `q4_0/q4_0` matches `f16/f16` throughput (~19 tok/s) while saving ~280 MB RSS —
  best choice when context headroom matters.
- `q8_0/q4_0` is slower (~10 tok/s) with intermediate RSS; likely a Metal dispatch
  overhead artifact at this context length; favourable for longer contexts where
  key precision matters more.
- KV quantization does not affect retrieval recall (retrieval uses the embedding
  model, not the GGUF KV cache).

---

## CAVEATS

**Gemma 4 12B does NOT fit 8 GB at long context.**

- Q4_K_M weights: ~6.3 GB
- KV cache at 32K context: f16 ≈ 14 GB / q4 ≈ 3.5 GB
- Total (q4 KV): ~9.8 GB — exceeds 8 GB unified memory
- For 8 GB, a 4–8B model at Q4_K_M is the right fit
- KV quantization buys ~2x context headroom but cannot make a 12B model fit 8 GB
- Do NOT claim "12B on 6 GB" or throughput figures above ~20 tok/s for 4B on M1 8 GB

**Single-run measurements.** Each combo was timed once (warmup + one generation pass).
Variance across runs can be ±10–15% due to Metal pipeline state and OS memory pressure.
For publication-grade numbers, average over ≥5 runs.

**Peak RSS, not peak GPU.** `psutil.Process().memory_info().rss` captures the process
resident set size on macOS unified memory. The Metal GPU and CPU share the same physical
pool, so RSS is the correct proxy for total memory pressure on Apple Silicon.

---

## Reproducibility

```bash
# Run the full matrix (takes ~5–10 min: 3 model loads + generations)
.venv/bin/python scripts/benchmark_matrix.py --stamp <your_stamp>

# Results land in (gitignored):
# app/tuning/results/matrix_<stamp>.jsonl
```

Generated: 2026-06-14, sift-flagship-v0.1 branch.

---

## make_knowledge (papers corpus)

Corpus-intelligence ingest vs raw PDF chunk baseline on the `papers/` corpus (30 PDFs, gitignored).
Metric: keyword-recall@10 proxy over 8 probes (`data/evaluation/papers_probes.json`) — same family as the 0.254 baseline.

| Mode            | Chunks | recall@10 | Notes                                           |
|-----------------|-------:|----------:|-------------------------------------------------|
| raw ingest      | 66,941 | 0.254     | Baseline (keyword-recall proxy, same corpus)      |
| make_knowledge  | 887    | 0.750     | 30 papers → 19 topics, 149 links; 75× fewer chunks |

Measured: 2026-06-18, `feat/sift-v1.1-rename`, M1 8 GB. Build ~2.3 h (30 papers, Qwen3-4B-Instruct-2507-Q4_K_M).
Phase 0 smoke: 3/3 valid JSON in 466 s. Report: [acceptance.json](reports/knowledge/acceptance.json).

**Reproducibility:**

```bash
# Phase 0: lock knowledge LLM on 3 papers (~2.5 GB model download first)
huggingface-cli download unsloth/Qwen3-4B-Instruct-2507-GGUF \
  Qwen3-4B-Instruct-2507-Q4_K_M.gguf --local-dir ~/.cache/gguf
./scripts/hardware_guard.sh
.venv/bin/python scripts/knowledge_phase0_smoke.py --papers 3 --output reports/knowledge/phase0_smoke.json

# Full acceptance: build + keyword-recall eval
.venv/bin/python scripts/knowledge_acceptance.py --build \
  --input papers/ --collection ai_papers_knowledge --profile papers \
  --probes data/evaluation/papers_probes.json \
  --output reports/knowledge/acceptance.json

# Eval only (artifacts + index already built):
.venv/bin/python scripts/knowledge_acceptance.py \
  --collection ai_papers_knowledge \
  --artifact-dir data/corpus/.knowledge/ai_papers_knowledge \
  --skip-hardware-guard \
  --output reports/knowledge/acceptance.json
```
