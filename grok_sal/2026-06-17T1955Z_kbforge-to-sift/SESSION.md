# kbforge migrated into sift — 2026-06-17T1955Z

## Why the move

User clarified: **RAG = Brain_rag = sift** (one public repo). kbforge was wrongly placed in legacy `RAG/rag_system/`. Consolidated into sift to avoid portfolio junk.

## What changed

| From | To |
|------|-----|
| `RAG/rag_system/kbforge/` | `Brain_rag/app/kbforge/` |
| `RAG/rag_system/profiles/` | `Brain_rag/profiles/` |
| `RAG/rag_system/kbforge/tests/` | `Brain_rag/tests/kbforge/` |
| `RAG/rag_system/scripts/hardware_guard.sh` | `Brain_rag/scripts/hardware_guard.sh` |

## Adaptations for in-repo use

- Imports: `app.kbforge.*`
- Embed model in `m1_8gb.toml`: `all-MiniLM-L6-v2` (matches sift `DENSE_MODEL_NAME`)
- Export dir: `ingest/` (was `sift_ingest/`)
- README + CI: `pytest -m "not slow"` includes kbforge tests
- Entry: `python -m app.kbforge` or `scripts/kbforge_build.py`

## Legacy RAG/

kbforge **removed** from RAG (deprecation pointer only). Do not extend RAG for sift work.

## Verification (Grok ran)

```text
hardware: ~227MB free, Claude ~428MB — skip slow/embed
pytest tests/kbforge -m "not slow" → 5 passed
CLI: examples/corpus → 3 files, 6 chunks, bundle OK
ruff check app/kbforge → clean (after --fix import sort)
```