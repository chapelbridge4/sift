#!/usr/bin/env bash
# Wait for model + hardware_guard, then Phase 0 smoke + full acceptance.
# Logs to reports/knowledge/auto_run.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"
GUARD="${ROOT}/scripts/hardware_guard.sh"
MODEL="${HOME}/.cache/gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
LOG_DIR="${ROOT}/reports/knowledge"
LOG="${LOG_DIR}/auto_run.log"
POLL_SEC="${POLL_SEC:-30}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-6}"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"
}

wait_for_model() {
  local deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
  while [[ $(date +%s) -lt $deadline ]]; do
    if [[ -f "$MODEL" ]]; then
      log "model OK: $MODEL ($(du -h "$MODEL" | awk '{print $1}'))"
      return 0
    fi
    if pgrep -f "huggingface-cli download.*Qwen3-4B-Instruct-2507" >/dev/null 2>&1; then
      log "model download in progress — waiting ${POLL_SEC}s"
    else
      log "model missing — starting huggingface-cli download"
      huggingface-cli download unsloth/Qwen3-4B-Instruct-2507-GGUF \
        Qwen3-4B-Instruct-2507-Q4_K_M.gguf --local-dir "${HOME}/.cache/gguf" \
        2>&1 | tee -a "$LOG" || true
    fi
    sleep "$POLL_SEC"
  done
  log "TIMEOUT: model not available at $MODEL within ${MAX_WAIT_HOURS}h"
  return 1
}

wait_for_ram() {
  local deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
  while [[ $(date +%s) -lt $deadline ]]; do
    if bash "$GUARD" >>"$LOG" 2>&1; then
      log "hardware_guard passed"
      return 0
    fi
    log "hardware_guard failed — retry in ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
  log "TIMEOUT: hardware_guard never passed within ${MAX_WAIT_HOURS}h"
  return 1
}

run_phase0() {
  log "=== Phase 0 smoke (3 papers) ==="
  "$PYTHON" scripts/knowledge_phase0_smoke.py --papers 3 \
    --output reports/knowledge/phase0_smoke.json 2>&1 | tee -a "$LOG"
}

run_acceptance() {
  log "=== Full acceptance build + eval ==="
  "$PYTHON" scripts/knowledge_acceptance.py --build \
    --input papers/ --collection ai_papers_knowledge --profile papers \
    --probes data/evaluation/papers_probes.json \
    --output reports/knowledge/acceptance.json 2>&1 | tee -a "$LOG"
}

acceptance_already_passed() {
  local report="${LOG_DIR}/acceptance.json"
  [[ -f "$report" ]] || return 1
  "$PYTHON" - <<'PY' "$report"
import json, sys
report = json.loads(open(sys.argv[1]).read())
sys.exit(0 if report.get("acceptance", {}).get("passed") else 1)
PY
}

update_benchmarks() {
  log "=== Updating BENCHMARKS.md from acceptance.json ==="
  "$PYTHON" - <<'PY' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path

root = Path(".")
report_path = root / "reports/knowledge/acceptance.json"
bench_path = root / "BENCHMARKS.md"
if not report_path.is_file():
    raise SystemExit(f"missing {report_path}")

report = json.loads(report_path.read_text())
chunks = report["eval"]["chunk_count"]
recall = report["eval"]["avg_keyword_recall"]
papers = report["artifacts"]["papers"]
topics = report["artifacts"]["topics"]
measured = __import__("datetime").datetime.utcnow().strftime("%Y-%m-%d")

text = bench_path.read_text()
if "| make_knowledge  | TBD" in text:
    old_row = "| make_knowledge  | TBD     | TBD       | Target: <1,000 chunks, >0.50 recall        |"
    new_row = (
        f"| make_knowledge  | {chunks:,}  | {recall:.3f}     | "
        f"{papers} papers, {topics} topics — measured {measured} |"
    )
    text = text.replace(old_row, new_row)
elif f"| make_knowledge  | {chunks:,}" in text or f"| make_knowledge  | {chunks} " in text:
    print(f"BENCHMARKS.md already has make_knowledge row (chunks={chunks})")
    raise SystemExit(0)
else:
    raise SystemExit("BENCHMARKS.md make_knowledge row not found — update manually")

bench_path.write_text(text)
print(f"Updated BENCHMARKS.md: chunks={chunks} recall={recall:.3f}")
PY
}

main() {
  log "=== knowledge_wait_and_run start (poll=${POLL_SEC}s max=${MAX_WAIT_HOURS}h) ==="
  if acceptance_already_passed; then
    log "acceptance already passed — skipping build (see reports/knowledge/acceptance.json)"
    update_benchmarks || true
    log "=== DONE (cached acceptance) ==="
    return 0
  fi
  wait_for_model
  wait_for_ram
  run_phase0
  run_acceptance
  update_benchmarks
  log "=== DONE — see reports/knowledge/{phase0_smoke,acceptance}.json ==="
}

main "$@"