#!/usr/bin/env bash
# Disk space pre-flight for heavy model/embed workloads on M1 8GB.
# Exit 0 = enough free space; exit 1 = critically low (run with --free).
set -euo pipefail

MIN_FREE_GB="${MIN_FREE_GB:-5}"
DATA_VOL="${DISK_GUARD_VOLUME:-/System/Volumes/Data}"

free_gb() {
  df -g "$DATA_VOL" | awk 'NR==2 {print $4}'
}

report_usage() {
  echo "[disk_guard] volume=$DATA_VOL free≈$(free_gb)GB (min=${MIN_FREE_GB}GB)"
  echo "[disk_guard] largest caches (informational):"
  du -sh \
    "${HOME}/.cache/huggingface" \
    "${HOME}/.cache/gguf" \
    "${HOME}/.ollama" \
    2>/dev/null | sed 's/^/[disk_guard]   /' || true
}

free_safe_caches() {
  echo "[disk_guard] freeing unused model caches (keeps sift/Brain_rag defaults)..."

  # GGUF duplicates (keep Instruct-2507 Q4_K_M symlink → HF blob)
  rm -f "${HOME}/.cache/gguf/gemma-4-12b-it-Q3_K_S.gguf" 2>/dev/null || true
  rm -f "${HOME}/.cache/gguf/Qwen3-4B-Q4_K_M.gguf" 2>/dev/null || true
  rm -f "${HOME}/.cache/gguf/Qwen3-4B-Instruct-2507-Q3_K_M.gguf" 2>/dev/null || true
  rm -f "${HOME}/.cache/gguf/Qwen3-4B-Instruct-2507-Q2_K.gguf" 2>/dev/null || true
  rm -f "${HOME}/.cache/gguf/Qwen3-0.6B-Q4_K_M.gguf" 2>/dev/null || true

  # HF models not used by Brain_rag defaults
  rm -rf \
    "${HOME}/.cache/huggingface/hub/models--OBLITERATUS--gemma-4-E4B-it-OBLITERATED" \
    "${HOME}/.cache/huggingface/hub/models--mlx-community--LFM2-8B-A1B-4bit" \
    "${HOME}/.cache/huggingface/hub/models--mlx-community--Granite-4.0-H-Tiny-4bit-DWQ" \
    "${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit" \
    "${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-8bit" \
    2>/dev/null || true
  find "${HOME}/.cache/huggingface" -name "*.incomplete" -delete 2>/dev/null || true

  # Ollama weights (if present)
  rm -rf "${HOME}/.ollama/models" "${HOME}/Library/Caches/ollama" 2>/dev/null || true

  # Project-local caches
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  rm -rf "${ROOT}/.pytest_cache" "${ROOT}/.ruff_cache" "${ROOT}/qdrant_storage" 2>/dev/null || true
  if [[ -x "${ROOT}/.venv/bin/pip" ]]; then
    "${ROOT}/.venv/bin/pip" cache purge >/dev/null 2>&1 || true
  fi

  echo "[disk_guard] after cleanup free≈$(free_gb)GB"
}

main() {
  report_usage
  avail="$(free_gb)"
  if [[ "${avail}" -ge "${MIN_FREE_GB}" ]]; then
    echo "[disk_guard] OK"
    exit 0
  fi

  echo "[disk_guard] WARN: low disk space (${avail}GB < ${MIN_FREE_GB}GB)"
  if [[ "${1:-}" == "--free" ]]; then
    free_safe_caches
    avail="$(free_gb)"
    if [[ "${avail}" -ge "${MIN_FREE_GB}" ]]; then
      echo "[disk_guard] OK after cleanup"
      exit 0
    fi
    echo "[disk_guard] still low after cleanup — free space manually"
    exit 1
  fi

  echo "[disk_guard] run: ./scripts/disk_guard.sh --free"
  exit 1
}

main "$@"