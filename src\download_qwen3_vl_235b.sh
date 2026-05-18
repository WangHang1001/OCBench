#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen3-VL-235B-A22B-Thinking}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/model/Qwen3-VL-235B-A22B-Thinking}"

mkdir -p "${MODEL_DIR}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "ERROR: huggingface-cli not found. Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

echo "Downloading ${MODEL_ID}"
echo "Target: ${MODEL_DIR}"

HF_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_ARGS+=(--token "${HF_TOKEN}")
fi

huggingface-cli download "${MODEL_ID}" \
  --local-dir "${MODEL_DIR}" \
  --resume-download \
  "${HF_ARGS[@]}"

echo "Done. Model files are in ${MODEL_DIR}"
