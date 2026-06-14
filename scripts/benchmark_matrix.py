#!/usr/bin/env python3
"""
Quant × hardware × quality matrix benchmark.

Measures generation cost (tok/s, peak RSS) for each KV-cache quantization
combo on Qwen3-4B-Q4_K_M running on Apple M1 8GB via llama-cpp-python/Metal.

Each combo runs in a FRESH SUBPROCESS so model memory is fully released
between runs — essential on 8 GB unified memory.

Retrieval metrics (recall@10, latency) are reported as a constant row
from Task 4 measurements and are NOT re-run (KV-cache is generation-only,
it does not affect retrieval).

Usage:
    .venv/bin/python scripts/benchmark_matrix.py --stamp m1_8gb
    .venv/bin/python scripts/benchmark_matrix.py --stamp m1_8gb --n-ctx 2048
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Subprocess worker code — embedded as a string so the parent can pass it
# via `python -c "..."` without needing an additional file on disk.
# ---------------------------------------------------------------------------
_WORKER_CODE = r"""
import json
import os
import sys
import time

import psutil

# Read config from environment
model_path = os.environ["GGUF_MODEL_PATH"]
n_ctx      = int(os.environ.get("GGUF_N_CTX", "4096"))
kv_k       = os.environ.get("GGUF_CACHE_TYPE_K", "f16")
kv_v       = os.environ.get("GGUF_CACHE_TYPE_V", "f16")

# Fixed benchmark prompt — kept identical across all combos for comparability
PROMPT = (
    "Explain the key differences between transformer self-attention and "
    "cross-attention mechanisms in neural networks, including how each "
    "mechanism processes queries, keys, and values."
)
MAX_TOKENS = 64

result = {
    "kv_k": kv_k,
    "kv_v": kv_v,
    "n_ctx": n_ctx,
    "ok": False,
    "gen_tok_s": None,
    "peak_rss_gb": None,
    "error": None,
}

try:
    from llama_cpp import Llama, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0

    _TYPE_MAP = {
        "f16":  GGML_TYPE_F16,
        "q8_0": GGML_TYPE_Q8_0,
        "q4_0": GGML_TYPE_Q4_0,
    }

    type_k = _TYPE_MAP[kv_k]
    type_v = _TYPE_MAP[kv_v]

    proc = psutil.Process()

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        flash_attn=True,
        type_k=type_k,
        type_v=type_v,
        verbose=False,
    )

    # Warmup: single-token pass to prime Metal pipeline
    llm("warm", max_tokens=1, echo=False)

    # Timed generation
    t0 = time.perf_counter()
    out = llm(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0, echo=False)
    elapsed = time.perf_counter() - t0

    tokens_generated = out["usage"]["completion_tokens"]
    tok_s = tokens_generated / elapsed if elapsed > 0 else 0.0

    rss_bytes = proc.memory_info().rss
    rss_gb = rss_bytes / (1024 ** 3)

    result.update({
        "ok": True,
        "gen_tok_s": round(tok_s, 2),
        "peak_rss_gb": round(rss_gb, 3),
    })

except MemoryError as e:
    result["error"] = f"MemoryError: {e}"
except Exception as e:
    result["error"] = str(e)

print(json.dumps(result))
"""

# ---------------------------------------------------------------------------
# KV-cache combos to benchmark
# ---------------------------------------------------------------------------
KV_COMBOS = [
    ("f16",  "f16"),
    ("q8_0", "q4_0"),
    ("q4_0", "q4_0"),
]

# ---------------------------------------------------------------------------
# Retrieval constants from Task 4 (dense MiniLM-L6 on SciFact, BEIR qrels)
# These do NOT change with KV-cache settings.
# ---------------------------------------------------------------------------
RETRIEVAL_ROW = {
    "type": "retrieval",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dataset": "SciFact",
    "metric": "recall@10",
    "value": 0.816,
    "n_queries": 100,
    "corpus_size": 5183,
    "latency_ms_per_query": 20,
    "note": "dense retrieval; true recall@k vs BEIR qrels",
}


def run_combo(kv_k: str, kv_v: str, model_path: str, n_ctx: int, python_bin: str) -> dict:
    """Run one KV-cache combo in a fresh subprocess; return the parsed JSON row."""
    env = {
        **os.environ,
        "GGUF_MODEL_PATH":    model_path,
        "GGUF_N_CTX":         str(n_ctx),
        "GGUF_CACHE_TYPE_K":  kv_k,
        "GGUF_CACHE_TYPE_V":  kv_v,
    }

    print(f"  Running combo kv_k={kv_k} kv_v={kv_v} ...", flush=True)

    try:
        proc = subprocess.run(
            [python_bin, "-c", _WORKER_CODE],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10-min hard cap; model load + generation takes ~2min
        )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        # Find the last JSON line (llama_cpp may emit some non-JSON lines to stdout)
        json_line = None
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                json_line = line

        if json_line is None:
            return {
                "kv_k": kv_k,
                "kv_v": kv_v,
                "n_ctx": n_ctx,
                "ok": False,
                "gen_tok_s": None,
                "peak_rss_gb": None,
                "error": f"No JSON output. stderr: {stderr[:400]}",
            }

        row = json.loads(json_line)
        row["type"] = "generation"
        return row

    except subprocess.TimeoutExpired:
        return {
            "kv_k": kv_k,
            "kv_v": kv_v,
            "n_ctx": n_ctx,
            "type": "generation",
            "ok": False,
            "gen_tok_s": None,
            "peak_rss_gb": None,
            "error": "TimeoutExpired (600s)",
        }
    except Exception as e:
        return {
            "kv_k": kv_k,
            "kv_v": kv_v,
            "n_ctx": n_ctx,
            "type": "generation",
            "ok": False,
            "gen_tok_s": None,
            "peak_rss_gb": None,
            "error": str(e),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quant × hardware × quality matrix benchmark")
    p.add_argument(
        "--stamp",
        required=True,
        help="Run stamp used verbatim in output filename (e.g. m1_8gb)",
    )
    p.add_argument(
        "--model-path",
        default=str(Path.home() / ".cache" / "gguf" / "Qwen3-4B-Q4_K_M.gguf"),
        help="Path to GGUF model file",
    )
    p.add_argument(
        "--n-ctx",
        type=int,
        default=4096,
        help="Context window size",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_path = args.model_path
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    # Use the Python binary that invoked this script so we share the same venv
    python_bin = sys.executable

    # Output path
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "app" / "tuning" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"matrix_{args.stamp}.jsonl"

    print(f"\n=== Quant × Hardware × Quality Matrix ===")
    print(f"Model: {model_path}")
    print(f"n_ctx: {args.n_ctx}")
    print(f"Stamp: {args.stamp}")
    print(f"Output: {out_path}")
    print(f"Combos: {KV_COMBOS}")
    print()

    rows = []

    # Generation benchmarks — each in a fresh subprocess
    for idx, (kv_k, kv_v) in enumerate(KV_COMBOS, 1):
        print(f"[{idx}/{len(KV_COMBOS)}] kv_k={kv_k} kv_v={kv_v}")
        row = run_combo(kv_k, kv_v, model_path, args.n_ctx, python_bin)
        status = "OK" if row.get("ok") else f"FAIL: {row.get('error', '?')}"
        print(f"  -> {status}")
        if row.get("ok"):
            print(f"     gen_tok_s={row['gen_tok_s']}  peak_rss_gb={row['peak_rss_gb']}")
        rows.append(row)
        print()

    # Append the constant retrieval row
    rows.append(RETRIEVAL_ROW)

    # Write JSONL
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")

    # Print summary table
    print("\n--- Generation Matrix ---")
    print(f"{'KV-K':<8} {'KV-V':<8} {'tok/s':<10} {'peak_RSS_GB':<14} {'n_ctx':<8} {'ok?'}")
    print("-" * 60)
    for r in rows:
        if r.get("type") == "generation":
            tok_s = f"{r['gen_tok_s']:.2f}" if r.get("gen_tok_s") is not None else "N/A"
            rss   = f"{r['peak_rss_gb']:.3f}" if r.get("peak_rss_gb") is not None else "N/A"
            ok    = "yes" if r.get("ok") else f"no ({r.get('error','?')[:30]})"
            print(f"{r['kv_k']:<8} {r['kv_v']:<8} {tok_s:<10} {rss:<14} {r['n_ctx']:<8} {ok}")

    print("\n--- Retrieval (constant, Task 4) ---")
    r = RETRIEVAL_ROW
    print(f"  {r['model']}  {r['dataset']}  recall@10={r['value']}  "
          f"{r['n_queries']} queries  corpus={r['corpus_size']}  "
          f"{r['latency_ms_per_query']}ms/query")


if __name__ == "__main__":
    main()
