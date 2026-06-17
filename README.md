# sift — local-first RAG with failure triage

[![ci](https://github.com/chapelbridge4/sift/actions/workflows/ci.yml/badge.svg)](https://github.com/chapelbridge4/sift/actions/workflows/ci.yml)

Most RAG tooling assumes a cloud LLM and a single tenant, and tells you *that* a query failed but not *why*. **sift** runs the full loop — retrieval, generation, evaluation, and **per-query failure triage** — **offline on an 8 GB laptop**, with no cloud account and no Docker required. Inference uses GGUF/llama.cpp (Windows, Linux, macOS, CPU-capable); Apple-Silicon MLX is an optional fast-path.

On the public **BEIR/SciFact** benchmark the default stack (MiniLM-L6 dense, embedded Qdrant) retrieves at **recall@10 = 0.774** — full 300-query test set, ~16 ms/query, 5,183-doc corpus — measured against ground-truth relevance judgments, not keyword matching. Generation with Qwen3-4B (Q4_K_M) runs at **~19 tok/s, ~2.9 GB RSS** on an Apple M1 (8 GB). Full numbers and the quant × KV-cache matrix: [BENCHMARKS.md](BENCHMARKS.md).

> Research/prototype service — no built-in authentication; the ingestion endpoint accepts server-local file paths. Keep it on trusted local networks unless you add auth and deployment hardening.

## Demo

```bash
.venv/bin/python scripts/demo.py
```

First run fetches the ~90 MB fastembed embedder once; subsequent runs are fully offline. No multi-GB LLM or GGUF is ever pulled.

Expected output (real run — no fabrication):

```
============================================================
sift — local-first RAG with per-query failure triage
============================================================

[1/4] Loading sample corpus from examples/corpus/ ...
  quantization.txt  (5 lines)
  rag.txt  (5 lines)
  triage.txt  (5 lines)

[2/4] Embedding with sentence-transformers/all-MiniLM-L6-v2 ...
  dim=384  docs=3  elapsed=95ms

[3/4] Indexing into in-memory Qdrant ...
  indexed 3 document(s)

[4/4] Retrieval demo

  Query: "How does dense retrieval use embeddings to find relevant documents?"

  Top-3 retrieved chunks:
    #1  [rag]  score=0.6778
        Retrieval-Augmented Generation (RAG) grounds language model responses in retrieved documents, reducing hallucination by anchoring output to factual passages.
    #2  [triage]  score=0.4187
        Per-query failure triage classifies why a RAG query failed by decomposing the pipeline into stages: chunking, retrieval, reranking, and generation.
    #3  [quantization]  score=0.3123
        GGUF is a binary format for storing quantized large language model weights, designed for efficient CPU and Apple Silicon inference via llama.

  Triage demo (intentionally failing query)
  Query:   "What is speculative decoding and how does it speed up autoregressive generation?"
  Gold ID: "speculative_decoding"  (absent from corpus — retrieval must miss)

  Triage verdict:
    failure_type  : RELEVANT_NOT_RETRIEVED
    stage         : retrieval
    confidence    : 0.90
    fix_hint      : Increase top_k, try hybrid (dense + sparse) retrieval, or use query expansion / HyDE to bridge the vocabulary gap.
    primary_stage : retrieval
    evidence      : recall_hit is False: no gold document was retrieved in the top-k candidates.

============================================================
Done — no LLM download, no Docker, no cloud.  Time to add your corpus.
============================================================
```

![sift demo](docs/assets/demo.gif)

<!-- record: asciinema rec demo.cast -c ".venv/bin/python scripts/demo.py"; agg demo.cast docs/assets/demo.gif -->
<!-- GIF capture is a MANUAL one-time human step — the text output block above makes this README useful before the GIF lands. -->

<!-- GitHub repo metadata (run ONCE manually — do NOT execute in CI):
gh repo edit chapelbridge4/sift --description "Local-first RAG with per-query failure triage — runs on 8GB, GGUF/llama.cpp, honest BEIR evals" --add-topic rag --add-topic llm --add-topic local-first --add-topic evaluation --add-topic qdrant --add-topic llama-cpp --add-topic information-retrieval
-->

## What it does

- **Local-first RAG pipeline** — dense / sparse / hybrid retrieval over Qdrant, reranking, and working-memory conversation context, behind a FastAPI API.
- **Runs on low hardware** — GGUF/llama.cpp default backend (cross-platform, CPU-capable); embedded Qdrant means zero infrastructure to clone and run; KV-cache quantization (`q8_0`/`q4_0`, flash-attention) for context headroom.
- **Measured honestly** — true recall@k on a public benchmark (BEIR/SciFact) plus a reproducible quant × KV-cache × throughput/RAM matrix; benchmarks state where they break, not just where they shine.
- **Per-query failure triage** — classifies each failed query by a 16-type RAG error taxonomy across four pipeline stages (chunking / retrieval / reranking / generation), with per-stage attribution, a confidence, and a fix hint.

## Failure triage

When a query fails, sift says *why*. The `app/triage` package turns each query's pipeline trace into deterministic signals (was the gold doc retrieved? did the reranker bury it? is the answer grounded in the retrieved context?) and classifies the failure against a 16-type taxonomy (`app/triage/taxonomy.py`), with per-stage attribution and a fix hint. An optional local LLM judge (off by default) disambiguates generation-stage failures — no cloud judge, no API key.

```python
from app.triage.classifier import classify
from app.triage.signals import QueryTrace

verdict = classify(QueryTrace(query="...", retrieved=[...], gold_ids={"gold"},
                              reranked=None, answer="...", top_k=10))
# verdict.failure_types -> [(RAGFailureType.RELEVANT_NOT_RETRIEVED, 0.9)]
# verdict.primary_stage -> "retrieval"
# verdict.evidence      -> "recall_hit is False: no gold document was retrieved..."
```

Real run on BEIR/SciFact via `scripts/run_triage.py` (full output: [reports/triage/scifact_sample.md](reports/triage/scifact_sample.md)):

```
100 queries · 82% passed (gold retrieved) · 18% failed
of the failures: 100% RELEVANT_NOT_RETRIEVED (retrieval stage)
```

Scope note (honest): that run feeds only retrieval signals (`answer=None`), so the measured failures are retrieval-stage misses — it is **not** a retrieval-vs-generation comparison. It is the zero-dependency default (only the embedder is downloaded).

To exercise more of the pipeline, the same runner takes `--rerank` and `--with-answers`:

- `--rerank` runs the cross-encoder reranker (`app/pipeline/reranker.py`) over each query's retrieved candidates and feeds the reranked doc_id order into the classifier, so **reranking-stage** demotions (`RELEVANT_DEMOTED`) surface alongside retrieval misses. A small (~80 MB) cross-encoder downloads on first use; CPU-only.
- `--with-answers` additionally generates a real answer per query via the local backend and enables the optional LLM judge, so **generation-stage** subtypes (`UNFAITHFUL` / `INCOMPLETE` / `CONTEXT_IGNORED`) can be disambiguated. Requires a downloaded GGUF/MLX model and is slow; it skips gracefully per-query when no model is available.

The multi-stage report ([reports/triage/scifact_full_sample.md](reports/triage/scifact_full_sample.md)) opens with an **active-signals line** (retrieved / reranked / answers / judge) so the distribution is honestly scoped — a stage with no live signal cannot fail there. With `--rerank` (and optionally `--with-answers`) the full 16-type taxonomy is genuinely exercised, not just asserted.

## kbforge — retrieval-first KB builder

Build optimized knowledge bases **before** indexing (no LLM at build time, M1 8GB profile):

```bash
.venv/bin/python -m app.kbforge build \
  --input ./examples/corpus \
  --output ./data/kb_bundles/v1 \
  --profile profiles/m1_8gb.toml \
  --skip-embed --skip-eval   # safe when RAM tight / Claude parallel

# Full pipeline (downloads MiniLM once):
.venv/bin/python -m app.kbforge build \
  --input ./examples/corpus \
  --output ./data/kb_bundles/v1 \
  --profile profiles/m1_8gb.toml \
  --probes ./data/evaluation/probes.json
```

Then copy `data/kb_bundles/v1/ingest/corpus/` → `data/corpus/` and `POST /upload_files`.

Agent handoff log: [`grok_sal.md`](grok_sal.md)

## Roadmap

- RAG-aware KV-cache management (per-chunk precision, cross-query reuse of retrieved-passage KV).

## Privacy And Safety

- Do not commit `.env`, logs, Qdrant storage, SQLite runtime databases, imported document corpora, MLX caches, or benchmark result dumps. The `.gitignore` is configured for these local artifacts.
- Ingestion is sandboxed to `ALLOWED_CORPUS_DIR` (default `./data/corpus`); paths outside it — including absolute paths, `..` traversal, and symlink escapes — are rejected with HTTP 400. The API still has no built-in authentication; keep it on trusted local networks unless you add auth.
- Retrieved document payloads include source filenames and chunk text. Treat Qdrant storage as private data.
- Application logs avoid raw query/prompt text by default, but deployment logging should still be reviewed before public hosting.

## Architecture

```text
FastAPI API
  -> RagOrchestrator: orchestrates retrieval, ranking, conversation memory, generation
  -> DocumentStore: collection management, document parsing, Qdrant indexing/search
  -> Reranker: importance scoring, optional reranking/diversification
  -> ConversationMemory: in-memory conversation history
  -> Inference backend: MLX (default for `/query` RAG path on Apple Silicon) or GGUF/llama.cpp (direct/experimental)
  -> Qdrant: vector and sparse payload storage (embedded by default; server optional)
```

Key directories:

- `app/main.py`: FastAPI app and HTTP endpoints.
- `app/models/schemas.py`: Pydantic request/response contracts.
- `app/pipeline/`: RAG orchestration modules (orchestrator, document store, reranker, conversation memory).
- `app/services/`: Qdrant, embedding, parsing, and generation services.
- `app/kbforge/`: offline KB builder (parse → chunk → embed → ingest bundle).
- `app/tuning/`: local benchmarking and quality utilities.
- `scripts/run_benchmark.py`: reproducible local benchmark harness.
- `tests/`: focused unit tests.

## Requirements

- Python 3.11+.
- No Docker required — Qdrant runs embedded by default. (Docker only if you opt into a Qdrant server.)
- macOS Apple Silicon for the default MLX RAG path (`INFERENCE_BACKEND=mlx`). GGUF/llama.cpp works cross-platform (Windows / Linux / macOS, CPU-capable) for direct generation and planned offline pipelines (e.g. `make_knowledge`).
- Disk space for one GGUF model (~2.5 GB for Qwen3-4B Q4_K_M) and embedding caches.

## Setup

```bash
git clone https://github.com/chapelbridge4/sift.git
cd sift

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # cross-platform core (GGUF backend)
# optional, Apple Silicon only:
# pip install -r requirements-mlx.txt

cp .env.example .env
```

Qdrant runs **embedded** by default (`QDRANT_MODE=embedded`) — no server needed. Fetch a GGUF model once:

```bash
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='unsloth/Qwen3-4B-GGUF', filename='Qwen3-4B-Q4_K_M.gguf', local_dir='~/.cache/gguf')"
```

Start the API:

```bash
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open the interactive docs at `http://127.0.0.1:8000/docs`.

Optional — run Qdrant as a server instead of embedded (set `QDRANT_MODE=server`):

```bash
docker run -p 127.0.0.1:6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:v1.18.0
```

## Configuration

Copy `.env.example` to `.env` and adjust local values. Common settings:

| Variable | Default | Notes |
| --- | --- | --- |
| `QDRANT_HOST` | `localhost` | Local Qdrant host |
| `QDRANT_PORT` | `6333` | HTTP port |
| `QDRANT_API_KEY` | empty | Set only for protected remote Qdrant |
| `INFERENCE_BACKEND` | `gguf` | `gguf` (cross-platform) or `mlx` (Apple fast-path) |
| `QDRANT_MODE` | `embedded` | `embedded` (no server) or `server` (host/port) |
| `QDRANT_PATH` | `./qdrant_data` | Embedded storage path (or `:memory:`) |
| `GGUF_MODEL_PATH` | `~/.cache/gguf/Qwen3-4B-Q4_K_M.gguf` | Path to the GGUF model file |
| `GGUF_CACHE_TYPE_K` / `GGUF_CACHE_TYPE_V` | `q8_0` / `q4_0` | KV-cache quant (requires flash-attention) |
| `CORS_ALLOW_ORIGINS` | local frontend origins | Comma-separated origins |
| `MODEL_PROFILE` | `fast` | `fast`, `balanced`, or `quality` |
| `DENSE_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding model |
| `SPARSE_STRATEGY` | `bm25` | `bm25` or experimental `bm25plus` |
| `CHUNK_SIZE` | `512` | Character chunk target |
| `CHUNK_OVERLAP` | `128` | Overlap for large chunks |
| `MAX_FILE_SIZE_MB` | `50` | Per-file parser limit |

The default RAG backend is **MLX** (`INFERENCE_BACKEND=mlx`, Apple Silicon). Profiles use `mlx-community/Qwen3.5-4B-MLX-4bit` and differ by token cap and thinking mode. For **GGUF/llama.cpp** (`INFERENCE_BACKEND=gguf`, model `Qwen3-4B-Q4_K_M.gguf`), use direct generation APIs — the full profile-based `/query` path requires MLX unless the backend implements the full `RagBackend` contract:

| Profile | Max Tokens | Thinking | Intended Use |
| --- | ---: | --- | --- |
| `fast` | 400 | off | Default local latency profile |
| `balanced` | 600 | off | Longer answers when memory allows |
| `quality` | 800 | on | Deeper reasoning experiments |

Models download to the MLX/Hugging Face cache on first use. To prefetch manually:

```bash
.venv/bin/python -m mlx_lm.download --model mlx-community/Qwen3.5-4B-MLX-4bit
```

## API Examples

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Create a collection:

```bash
curl -X POST http://127.0.0.1:8000/build_collection \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "my_documents"}'
```

Index server-local files:

```bash
curl -X POST http://127.0.0.1:8000/upload_files \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_documents",
    "file_paths": ["/absolute/path/to/document.pdf"],
    "batch_size": 32
  }'
```

Query with retrieval and generation:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_documents",
    "query": "What is the main topic?",
    "top_k": 10,
    "fusion_method": "rrf",
    "model_profile": "fast",
    "use_llm": true
  }'
```

Retrieval-only query:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_documents",
    "query": "Find relevant passages",
    "use_llm": false,
    "include_metadata": false
  }'
```

List model profiles:

```bash
curl http://127.0.0.1:8000/models/profiles
```

## Supported Documents

The parser reliably targets:

- PDF via PyMuPDF with PyPDF2 fallback
- TXT with encoding detection
- DOCX via `python-docx`
- XLSX via `openpyxl`
- Markdown chunking for `.md` paths when routed through the parser

Legacy `.doc` and `.xls` are still accepted by the schema but are not guaranteed because the current parser routes them through DOCX/XLSX libraries. Convert legacy Office files before indexing for best results.

## Development

Run focused tests:

```bash
.venv/bin/python -m pytest tests
```

Compile app modules:

```bash
.venv/bin/python -m compileall app
```

Run the real retrieval benchmark (BEIR/SciFact, true recall@k — no server, no GPU needed):

```bash
.venv/bin/python scripts/benchmark_beir.py --top-k 10 --max-queries 100
```

Reproduce the quant × KV-cache throughput/RAM matrix (needs the GGUF model):

```bash
.venv/bin/python scripts/benchmark_matrix.py --stamp local
```

## Troubleshooting

- Qdrant: embedded mode (default) needs no server. Only if `QDRANT_MODE=server`, start one (see Setup) and check `curl http://127.0.0.1:6333/health`.
- Slow first answer: the model may be downloading or compiling Metal kernels.
- Memory pressure on 8 GB machines: keep `MODEL_PROFILE=fast`, lower `BATCH_SIZE`, and avoid concurrent model-profile switches.
- Empty retrieval: confirm the collection exists, files were indexed, and Qdrant storage has not been deleted.

## License

MIT License
