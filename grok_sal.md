# Grok Session Activity Log (SAL) — sift / Brain_rag

Cross-agent handoff for **Grok ↔ Claude Code** on shared M1 8GB.

## Canonical repo

**One public repo:** `Brain_rag` = **sift** = `chapelbridge4/sift`  
Do **not** duplicate features in legacy `RAG/` — that folder is junk/deprecated for this workstream.

## Rules

1. Hardware + process check before embed/tests (`scripts/hardware_guard.sh`).
2. Self-fix loop: classify → diagnose → fix → re-verify until green.
3. Session logs: `grok_sal/<datetime>_<slug>/SESSION.md`
4. **Brainstorm before implement** — spec approval gate for `make_knowledge`.
5. **Cross-agent:** read latest handoff in index before editing shared files (`README.md`, `app/pipeline/*`).

## Session index

| Datetime (UTC) | Agent | Slug | Summary |
|----------------|-------|------|---------|
| 2026-06-17T1836Z | Claude | claude-bugb-fixed-handoff | Bug B fixed (`e9f9c9c`); rename branch 259 passed; README conflict flagged |
| 2026-06-17T1955Z | Grok | kbforge-to-sift | Migrated kbforge from RAG → `app/kbforge/` |
| 2026-06-17T2200Z | Grok | make-knowledge-design | Option C spec + model research; no code yet |
| 2026-06-17T2230Z | Grok | grok-ack-claude-handoff | Ack Bug B; README+CI committed; spec updated |
| 2026-06-17T2300Z | Grok | make-knowledge-p1 | Plan gaps patched; Phase 1 complete (models/config/artifacts/tier0) |
| 2026-06-18T1200Z | Grok | make-knowledge-review-fixes | Phases 1–5 done; review-gap fixes |
| 2026-06-17T2337Z | Claude | claude-reviews-make-knowledge | **APPROVED** — no Bug B regression; 324 passed |
| 2026-06-18T1230Z | Grok | grok-claude-review-ack | Applied ruff + observability + CLI profile dedup; committed |

## Latest (read both for full picture)

| Agent | Handoff |
|-------|---------|
| Claude | [`grok_sal/2026-06-17T2337Z_claude-reviews-make-knowledge/SESSION.md`](grok_sal/2026-06-17T2337Z_claude-reviews-make-knowledge/SESSION.md) |
| Grok | [`grok_sal/2026-06-18T1230Z_grok-claude-review-ack/SESSION.md`](grok_sal/2026-06-18T1230Z_grok-claude-review-ack/SESSION.md) |

## Current repo state

| Item | Value |
|------|-------|
| Branch | `feat/sift-v1.1-rename` — make_knowledge stack + review fixes **committed locally** |
| Bug B | **Resolved** — Claude verified no regression (37 Bug-B tests pass) |
| make_knowledge | **APPROVED** by Claude; Grok applied action items 1–3 |
| Tests (fast) | **76 passed** (Grok); Claude full suite **324 passed / 1 skipped** |
| Push | **Not pushed** — user-gated; remote still at Claude `d8fed16` |

## Active workstream (Grok)

**`make_knowledge`** — [`docs/superpowers/plans/2026-06-17-make-knowledge.md`](docs/superpowers/plans/2026-06-17-make-knowledge.md)

| Item | Value |
|------|-------|
| v1 corpus | `papers/` (30 PDFs, gitignored) |
| Baseline | 66,941 chunks, ~0.254 recall |
| Design | Option C: topic sheets + paper summaries + drill-down |
| Knowledge LLM | `Qwen3-4B-Instruct-2507` Q4_K_M via `KNOWLEDGE_GGUF_MODEL_PATH` |
| RAG LLM (default) | MLX `Qwen3.5-4B-MLX-4bit` (`INFERENCE_BACKEND=mlx`) |
| Acceptance | `BENCHMARKS.md` knowledge row **TBD**; Phase 0 smoke-test **not run** |

## Untracked (Grok, separate from knowledge commits)

`app/kbforge/`, `profiles/`, `scripts/hardware_guard.sh`, `tests/kbforge/`, `grok_sal/` (SAL logs not always committed)

## Safe test commands

```bash
cd Brain_rag
./scripts/hardware_guard.sh || true
./.venv/bin/python -m pytest tests/knowledge/ tests/test_query_contract.py -q -m "not slow"
python -m app.kbforge build --input examples/corpus --output /tmp/kb --skip-embed --skip-eval
```

**Avoid without user OK:** full `papers/` LLM batch, embed on low RAM, Gemma-12B / 5GB+ models.