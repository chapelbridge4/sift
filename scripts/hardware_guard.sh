#!/usr/bin/env bash
# Pre-flight check before kbforge embed tests or heavy pytest.
# Exit 0 = safe for light tests; exit 1 = skip embed/slow tests.
set -euo pipefail

MIN_FREE_PAGES=100000  # ~1.6 GB (pages are 16KB on Apple Silicon)
CLAUDE_MAX_RSS_KB=524288  # 512 MB — if Claude exceeds, warn

pages_free=$(vm_stat | awk '/Pages free/ {gsub(/\./,"",$3); print $3}')
pages_spec=$(vm_stat | awk '/Pages speculative/ {gsub(/\./,"",$3); print $3}')
effective_free=$((pages_free + pages_spec))
free_mb=$((effective_free * 16 / 1024))

echo "[hardware_guard] effective_free≈${free_mb}MB (pages=${effective_free})"

if pgrep -x claude >/dev/null 2>&1; then
  claude_rss=$(ps -o rss= -p "$(pgrep -x claude | head -1)" 2>/dev/null | tr -d ' ')
  claude_mb=$((claude_rss / 1024))
  echo "[hardware_guard] claude RSS=${claude_mb}MB"
  if [[ "${claude_rss:-0}" -gt "${CLAUDE_MAX_RSS_KB}" ]]; then
    echo "[hardware_guard] WARN: Claude active — skip embed/slow tests"
    exit 1
  fi
fi

if [[ "${effective_free}" -lt "${MIN_FREE_PAGES}" ]]; then
  echo "[hardware_guard] WARN: low free memory — skip embed/slow tests"
  exit 1
fi

echo "[hardware_guard] OK for embed/slow tests"
exit 0