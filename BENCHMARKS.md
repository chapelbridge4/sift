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

Method: dense vector search (cosine similarity) over BEIR SciFact corpus.
Metric: true recall@k against BEIR qrels (not surrogate scores).

| Encoder                              | Dataset  | Metric     | Value | Queries | Corpus | Latency     |
|--------------------------------------|----------|------------|-------|---------|--------|-------------|
| sentence-transformers/all-MiniLM-L6-v2 | SciFact | recall@10  | 0.816 | 100     | 5 183  | 20 ms/query |

> recall@10 is a retrieval-only metric. It is independent of the generation KV-cache settings
> and does not change across the combos below.

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
