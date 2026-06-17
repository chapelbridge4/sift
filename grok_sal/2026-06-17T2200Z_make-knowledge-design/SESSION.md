# make_knowledge design + model research — 2026-06-17T2200Z

## Status for Claude

**Design phase complete. Implementation NOT started.** User must approve spec before coding (brainstorming skill gate).

## What the user wants

Corpus intelligence inside **sift** (`Brain_rag/`), not a separate repo. Problem is **not** chunk-and-embed — it's:

1. Ingest n documents (v1 test corpus: `papers/` — 30 arXiv PDFs, gitignored)
2. Find repeated patterns across papers
3. Remake KB into canonical docs optimized for vector retrieval
4. **Option C confirmed:** topic sheets (merged concepts) + slim per-paper summaries (citations)
5. Hierarchical retrieval: query hits topics first → drill-down to paper via `links_to`

`Comune.oristano.it` was illustrative only — do NOT optimize for municipal sites.

## Baseline to beat

| Metric | Current (`ai_papers` raw ingest) |
|--------|----------------------------------|
| Chunks | ~66,941 |
| Keyword recall proxy | ~0.254 |

Target v1: **<1,000 chunks**, recall **>0.50**.

## Artifacts produced this session

| File | Purpose |
|------|---------|
| `docs/superpowers/specs/2026-06-17-make-knowledge-design.md` | Full design spec (Option C, pipeline tiers, API, retrieval, tests) |
| §6.0 in spec | Evidence-based model research (forums + benchmarks) — user called out Grok for skipping this initially; now documented |

## Architecture summary (for quick Claude onboarding)

```
make_knowledge=true on POST /upload_files
  → Tier 0: embed + cluster claim spans (no LLM)
  → Tier 1: LLM per paper (~30 calls) → papers/{id}.md
  → Tier 2: LLM per topic cluster (~15 calls) → topics/{slug}.md
  → Tier 3: chunk canonical artifacts only → Qdrant hybrid index
  → Retrieval: topic boost + optional drill_down to paper_summary chunks
```

**Module plan:**
- Keep `app/kbforge/` as primitives (parse/chunk/embed/export) — v0.1 scope was wrong for corpus intelligence
- Add `app/knowledge/` for LLM restructuring pipeline
- Extend `UploadFilesRequest`: `make_knowledge`, `knowledge_profile`, `knowledge_model`

## Model research conclusion (do NOT default to base Qwen3-4B)

| Priority | Model | Size | Why |
|----------|-------|------|-----|
| **1 Default** | `Qwen3-4B-Instruct-2507` Q4_K_M | 2.5 GB | #1 extraction benchmark (distil labs); 5.7% hallucination (Vectara); non-thinking |
| **2 Alt** | `mlx-community/Qwen3.5-4B-MLX-4bit` | ~2–3 GB | sift RAG profiles + community Apr 2026 pick |
| Reject | Phi-4-mini | ~3 GB | 23.5% summarization hallucination |
| Reject | Gemma-3-4B | ~3 GB | 67% answer rate (batch killer) |
| Reject | Gemma-4-12B on disk | 5.1 GB | Won't fit M1 8GB with IDE agents |

**On disk today:**
```
~/.cache/gguf/Qwen3-4B-Q4_K_M.gguf           2.5 GB  (base — upgrade for knowledge tasks)
~/.cache/gguf/gemma-4-12b-it-Q3_K_S.gguf     5.1 GB  (too large)
Ollama: gemma4-obliterated                   5.3 GB  (too large)
```

**Download recommended default:**
```bash
huggingface-cli download unsloth/Qwen3-4B-Instruct-2507-GGUF \
  Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --local-dir ~/.cache/gguf
```

New env: `KNOWLEDGE_GGUF_MODEL_PATH` (separate from `GGUF_MODEL_PATH` for RAG).

**Not done yet:** 3-paper smoke test comparing Instruct-2507 vs Qwen3.5 MLX vs base Qwen3-4B.

## API sketch (spec — not implemented)

```python
make_knowledge: bool = False
knowledge_profile: Optional[str] = "papers"
knowledge_model: Optional[str] = None
```

Query extension: `drill_down: bool = False` on `/query`.

## Implementation phases (after spec approval)

1. P1 — `app/knowledge/` models, profiles, Tier 0 cluster
2. P2 — Tier 1/2 LLM + GGUF + hardware guard
3. P3 — `form_memories()` branch + index metadata
4. P4 — retrieval boost + drill-down in orchestrator
5. P5 — tests + `papers/` acceptance + BENCHMARKS.md

Next skill after approval: **writing-plans** → `docs/superpowers/plans/2026-06-17-make-knowledge.md`

## Known sift context (from SESSION_STATE.md)

- Branch work on `feat/sift-v1.1-hardening` / rename — not pushed
- SciFact recall@10 = 0.774 (real, measured)
- **Bug B:** `RagOrchestrator` hardcodes MLX; `INFERENCE_BACKEND=gguf` dead for RAG path — decide before unifying `make_knowledge` backend
- MLX profiles already point to `Qwen3.5-4B-MLX-4bit`

## Safe commands (Claude)

```bash
cd Brain_rag
./scripts/hardware_guard.sh || true
python -m pytest tests/kbforge -m "not slow" -v
# Do NOT run full papers/ LLM batch without hardware guard + user OK
```

## Open gates

- [ ] User approves `docs/superpowers/specs/2026-06-17-make-knowledge-design.md`
- [ ] Optional: 3-paper model smoke test before implementation
- [ ] writing-plans skill → implementation plan

## Do NOT

- Implement before spec approval
- Extend legacy `RAG/` repo
- Use Gemma-12B or 5GB+ models on 8GB M1 during parallel sessions
- LLM per chunk (66k scale) — only per-paper + per-topic-cluster