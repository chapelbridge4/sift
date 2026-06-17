# Grok — make_knowledge review fixes — 2026-06-18T1200Z

**For:** Claude Code review of uncommitted Grok changes on `feat/sift-v1.1-rename`.  
**Plan:** `docs/superpowers/plans/2026-06-17-make-knowledge.md`  
**Spec:** `docs/superpowers/specs/2026-06-17-make-knowledge-design.md`

## Context

Phases 1–5 of `make_knowledge` were largely complete (local commits through `433b1c5`). User asked for per-step review against plan + best practices (DRY/KISS/YAGNI/SOLID, config-driven, DI, TDD, M1 8GB). This session closes the review gaps — **no new features**, only plan/spec alignment fixes.

## Review fixes applied (uncommitted)

| Gap | Fix | Plan/spec ref |
|-----|-----|---------------|
| Hardware guard after Tier 0 embed | Guard runs **before** embed; new test proves embedder not called on guard fail | Plan §3.1b, spec §6.0 |
| One model at a time | `_release_embedder()` after cluster, before Tier 1 LLM (`EmbeddingService.cleanup`) | Plan Task 3.1 |
| No exponential backoff on JSON retry | `KnowledgeLLM` sleeps `retry_backoff_base * 2^attempt`; wired via `build_knowledge_llm(profile)` | Spec §8 |
| Magic numbers in code | Moved to TOML/settings: `max_sentences_per_claim`, `max_claims_per_paper`, `max_retries`, `retry_backoff_base_seconds`, `[parse].extensions`; `KNOWLEDGE_DRILL_DOWN_TOP_K`; orchestrator uses `RERANK_TOP_K` | Plan §6.1, bible |
| `degraded.py` imported private `_contributing_papers` | Renamed to public `contributing_papers()` in `tier2_merge.py` | SOLID / coupling |
| `index_artifacts` chunked full file incl. YAML | New `artifact_body_text()`; index chunks body only | Plan Task 3.2 |
| CLI hardcoded extensions | `scan_input_files(..., profile)` uses `profile.parse.extensions` | Config-driven |

## Files touched (this session only)

**Modified (uncommitted):**

- `app/config.py` — `KNOWLEDGE_DRILL_DOWN_TOP_K`
- `app/knowledge/{artifacts,backend,cli,config,degraded,index,pipeline,retrieval,tier0_cluster,tier2_merge}.py`
- `app/knowledge/profiles/knowledge_papers.toml`
- `app/pipeline/{document_store,orchestrator}.py`
- `tests/knowledge/{test_backend,test_cli,test_config,test_degraded,test_indexing,test_knowledge_config_settings}.py`

**Not modified (Grok reuse boundary):** `app/kbforge/`, `profiles/`, `scripts/hardware_guard.sh`

## Verification

```bash
cd Brain_rag
./.venv/bin/python -m pytest tests/knowledge/ tests/test_query_contract.py -q -m "not slow"
# 76 passed, 1 deselected (slow e2e)
```

New tests: `test_pipeline_guard_runs_before_embed`, `test_extract_backoff_between_retries`, indexing frontmatter exclusion assertions, profile field assertions.

## Prior committed work (for review scope)

Local commits on branch (not pushed): `2f8fd82` … `433b1c5` — full `app/knowledge/` package, `form_memories` branch, drill-down, citations, CLI, mocked slow e2e, README/BENCHMARKS placeholder.

## Still open (not this session)

| Item | Owner | Notes |
|------|-------|-------|
| Phase 0 model smoke-test | User-gated | Download `Qwen3-4B-Instruct-2507-Q4_K_M`, 3-paper JSON validity run |
| Acceptance benchmark row | User-gated | Full `papers/` run → `BENCHMARKS.md` knowledge row (still TBD) |
| Commit + push | Human | Grok did **not** commit these review fixes; Claude/user should review first |
| `app/kbforge/`, `grok_sal/`, `profiles/` | Untracked | Grok-owned; separate from knowledge commits |

## Claude review checklist

1. Guard-before-embed ordering in `pipeline.py` — matches spec §6.0?
2. `artifact_body_text` — paper summaries chunk rendered body, not frontmatter duplicate?
3. `build_knowledge_llm` — correct DI path in `document_store` + CLI?
4. `contributing_papers` public API — acceptable vs shared util module?
5. `orchestrator` `RERANK_TOP_K` for context limit — OK vs knowledge-specific setting?
6. Any conflict with Claude's `e9f9c9c` Bug B / rename work on `document_store` / `orchestrator`?

## Safe commands

```bash
./scripts/hardware_guard.sh || true
./.venv/bin/python -m pytest tests/knowledge/ -q -m "not slow"
```

**Avoid without user OK:** full `papers/` LLM batch, real GGUF download on low RAM.