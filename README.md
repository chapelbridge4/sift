# Brain-Inspired RAG

Brain-Inspired RAG is a local-first Retrieval-Augmented Generation service built with FastAPI, Qdrant, FastEmbed, and Apple Silicon MLX inference. It indexes local documents into Qdrant and answers questions through a small set of brain-inspired orchestration modules.

This repository is a research/prototype service, not a production-hosted API. It has no built-in authentication, and the ingestion endpoint accepts server-local file paths. Keep it bound to trusted local networks unless you add auth, authorization, and deployment hardening.

## Current Status

- Local FastAPI API for collection management, document indexing, retrieval, and RAG responses.
- Qdrant-backed dense, sparse, and hybrid retrieval paths.
- MLX/MLX-VLM generation profiles tuned for an 8 GB Apple Silicon machine.
- Focused unit tests for API contracts, parser behavior, startup boundaries, tuning helpers, and benchmark scaffolding.
- Public-push hygiene is documented in [docs/audits/public-release-readiness-2026-06-10.md](docs/audits/public-release-readiness-2026-06-10.md).

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
  -> LLMService: local MLX/MLX-VLM generation
  -> Qdrant: vector and sparse payload storage
```

Key directories:

- `app/main.py`: FastAPI app and HTTP endpoints.
- `app/models/schemas.py`: Pydantic request/response contracts.
- `app/brain/`: orchestration modules.
- `app/services/`: Qdrant, embedding, parsing, and generation services.
- `app/tuning/`: local benchmarking and quality utilities.
- `scripts/run_benchmark.py`: reproducible local benchmark harness.
- `tests/`: focused unit tests.

## Requirements

- Python 3.12 recommended.
- Docker or Docker Desktop for Qdrant.
- macOS on Apple Silicon for MLX generation.
- Enough disk space for Hugging Face / MLX model caches.

## Setup

```bash
git clone <repository-url>
cd Brain_rag

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Start Qdrant:

```bash
docker compose up -d qdrant
```

Or without Compose:

```bash
docker run -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.18.0
```

Start the API:

```bash
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open the interactive docs at `http://127.0.0.1:8000/docs`.

## Configuration

Copy `.env.example` to `.env` and adjust local values. Common settings:

| Variable | Default | Notes |
| --- | --- | --- |
| `QDRANT_HOST` | `localhost` | Local Qdrant host |
| `QDRANT_PORT` | `6333` | HTTP port |
| `QDRANT_API_KEY` | empty | Set only for protected remote Qdrant |
| `CORS_ALLOW_ORIGINS` | local frontend origins | Comma-separated origins |
| `MODEL_PROFILE` | `fast` | `fast`, `balanced`, or `quality` |
| `DENSE_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding model |
| `SPARSE_STRATEGY` | `bm25` | `bm25` or experimental `bm25plus` |
| `CHUNK_SIZE` | `512` | Character chunk target |
| `CHUNK_OVERLAP` | `128` | Overlap for large chunks |
| `MAX_FILE_SIZE_MB` | `50` | Per-file parser limit |

Current MLX profiles all use `mlx-community/Qwen3.5-4B-MLX-4bit` and differ by token cap and thinking mode:

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

Run the benchmark harness when Qdrant and the local MLX environment are available:

```bash
.venv/bin/python scripts/run_benchmark.py --model-profile fast --fusion-method rrf
```

## Troubleshooting

- Qdrant unavailable: run `docker compose up -d qdrant` and check `curl http://127.0.0.1:6333/health`.
- Slow first answer: the model may be downloading or compiling Metal kernels.
- Memory pressure on 8 GB machines: keep `MODEL_PROFILE=fast`, lower `BATCH_SIZE`, and avoid concurrent model-profile switches.
- Empty retrieval: confirm the collection exists, files were indexed, and Qdrant storage has not been deleted.

## License

MIT License
