# sift — local-first RAG with honest evaluation

[![ci](https://github.com/chapelbridge4/sift/actions/workflows/ci.yml/badge.svg)](https://github.com/chapelbridge4/sift/actions/workflows/ci.yml)

Most RAG tooling assumes a cloud LLM and a single tenant. **sift** runs the full loop — retrieval, generation, and evaluation — **offline on an 8 GB laptop**, with no cloud account and no Docker required. Inference uses GGUF/llama.cpp (Windows, Linux, macOS, CPU-capable); Apple-Silicon MLX is an optional fast-path.

On the public **BEIR/SciFact** benchmark the default stack (MiniLM-L6 dense, embedded Qdrant) retrieves at **recall@10 = 0.816** — 100 queries, ~20 ms/query, 5,183-doc corpus — measured against ground-truth relevance judgments, not keyword matching. Generation with Qwen3-4B (Q4_K_M) runs at **~19 tok/s, ~2.9 GB RSS** on an Apple M1 (8 GB). Full numbers and the quant × KV-cache matrix: [BENCHMARKS.md](BENCHMARKS.md).

> Research/prototype service — no built-in authentication; the ingestion endpoint accepts server-local file paths. Keep it on trusted local networks unless you add auth and deployment hardening.

## What it does

- **Local-first RAG pipeline** — dense / sparse / hybrid retrieval over Qdrant, reranking, and working-memory conversation context, behind a FastAPI API.
- **Runs on low hardware** — GGUF/llama.cpp default backend (cross-platform, CPU-capable); embedded Qdrant means zero infrastructure to clone and run; KV-cache quantization (`q8_0`/`q4_0`, flash-attention) for context headroom.
- **Measured honestly** — true recall@k on a public benchmark (BEIR/SciFact) plus a reproducible quant × KV-cache × throughput/RAM matrix; benchmarks state where they break, not just where they shine.

## Roadmap

- **Per-query failure triage** — classifying each failed query by a RAG error taxonomy (chunking / retrieval / reranking / generation), per-tenant and per-pipeline-span. This is the headline direction; today the repo ships the pipeline + the honest evaluation it builds on.
- RAG-aware KV-cache management (per-chunk precision, cross-query reuse of retrieved-passage KV).

## Privacy And Safety

- Do not commit `.env`, logs, Qdrant storage, SQLite runtime databases, imported document corpora, MLX caches, or benchmark result dumps. The `.gitignore` is configured for these local artifacts.
- The `/upload_files` endpoint indexes files by server-local path. If the API is exposed to untrusted clients, a caller could ask it to read local files the service process can access.
- Retrieved document payloads include source filenames and chunk text. Treat Qdrant storage as private data.
- Application logs avoid raw query/prompt text by default, but deployment logging should still be reviewed before public hosting.

## Architecture

```text
FastAPI API
  -> PrefrontalCortex: orchestrates retrieval, ranking, working memory, generation
  -> Hippocampus: collection management, document parsing, Qdrant indexing/search
  -> Amygdala: importance scoring, optional reranking/diversification
  -> WorkingMemory: in-memory conversation history
  -> Inference backend: GGUF/llama.cpp (default, cross-platform) or MLX (optional Apple fast-path)
  -> Qdrant: vector and sparse payload storage (embedded by default; server optional)
```

> Design rationale: the module names (PrefrontalCortex/Hippocampus/Amygdala/WorkingMemory) are a memory-systems analogy for the orchestration roles — an internal design choice, not the product claim. The product claim is the engineering: local-first, low-hardware, honestly measured.

Key directories:

- `app/main.py`: FastAPI app and HTTP endpoints.
- `app/models/schemas.py`: Pydantic request/response contracts.
- `app/brain/`: orchestration modules.
- `app/services/`: Qdrant, embedding, parsing, and generation services.
- `app/tuning/`: local benchmarking and quality utilities.
- `scripts/run_benchmark.py`: reproducible local benchmark harness.
- `tests/`: focused unit tests.

## Requirements

- Python 3.11+.
- No Docker required — Qdrant runs embedded by default. (Docker only if you opt into a Qdrant server.)
- Any OS for the default GGUF/llama.cpp backend (Windows / Linux / macOS, CPU works). macOS Apple Silicon only for the optional MLX fast-path.
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

The default backend is **GGUF/llama.cpp** (`INFERENCE_BACKEND=gguf`, model `Qwen3-4B-Q4_K_M.gguf`). The MLX profiles below apply only when `INFERENCE_BACKEND=mlx` (Apple Silicon); they use `mlx-community/Qwen3.5-4B-MLX-4bit` and differ by token cap and thinking mode:

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
