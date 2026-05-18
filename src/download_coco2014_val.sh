#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/dataset/coco2014}"
VAL_ZIP="${DATA_ROOT}/val2014.zip"

VAL_URL="http://images.cocodataset.org/zips/val2014.zip"

mkdir -p "${DATA_ROOT}"

download_file() {
  local url="$1"
  local output="$2"
  if [[ -s "${output}" ]]; then
    echo "Found existing file: ${output}"
    return
  fi

  echo "Downloading ${url}"
  echo "Target: ${output}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 8 -s 8 -o "$(basename "${output}")" -d "$(dirname "${output}")" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "${output}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -C - --fail --retry 5 --retry-delay 10 -o "${output}" "${url}"
  else
    echo "ERROR: need aria2c, wget, or curl to download files." >&2
    exit 1
  fi
}

download_file "${VAL_URL}" "${VAL_ZIP}"

echo "Extracting val2014..."
unzip -n -q "${VAL_ZIP}" -d "${DATA_ROOT}"

image_count="$(find "${DATA_ROOT}/val2014" -maxdepth 1 -type f -name '*.jpg' | wc -l | tr -d ' ')"
echo "val2014 image count: ${image_count}"

if [[ "${image_count}" != "40504" ]]; then
  echo "WARNING: expected 40504 val2014 images, got ${image_count}." >&2
fi

echo "Done. COCO2014 val data is in ${DATA_ROOT}"
