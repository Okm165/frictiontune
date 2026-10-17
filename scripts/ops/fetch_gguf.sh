#!/usr/bin/env bash
# Fetch a GGUF for the container `llm` image (a llama.cpp server built for HIP).
#
# The image takes a GGUF at run time rather than baking one in. This downloads
# the files the pipeline's two local models need into the git-ignored
# `.toolchain/models/`, which the pod mounts at the same path via the repository
# bind mount.
#
#   scripts/ops/fetch_gguf.sh qwen       # Qwen3-8B Q4_K_M chat model (~5 GB)
#   scripts/ops/fetch_gguf.sh bge-m3     # bge-m3 Q8_0 embedder (~0.6 GB)
#   scripts/ops/fetch_gguf.sh all        # both
#
# Any preset can be overridden for a variant without editing the script:
#   FT_GGUF_PRESET=<name> FT_GGUF_REPO=<org>/<repo> FT_GGUF_FILE=<file.gguf> \
#     scripts/ops/fetch_gguf.sh
#
# Run the two servers with (see container/README.md):
#   FRICTIONTUNE_LLM_MODEL_PATH=/work/.toolchain/models/Qwen3-8B-Q4_K_M.gguf
#   FRICTIONTUNE_LLM_MODEL_PATH=/work/.toolchain/models/bge-m3-Q8_0.gguf \
#     FRICTIONTUNE_LLM_MODE=embed
set -euo pipefail

REPO="${FT_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DEST="${FT_GGUF_DIR:-${REPO}/.toolchain/models}"
PRESET="${1:-${FT_GGUF_PRESET:-qwen}}"

PYTHON="${FT_PYTHON:-${REPO}/.venv/bin/python}"
[ -x "${PYTHON}" ] || PYTHON="python3"

fetch() {
  local repo="$1" file="$2"
  printf '[fetch] %s/%s -> %s\n' "${repo}" "${file}" "${DEST}"
  "${PYTHON}" - "${repo}" "${file}" "${DEST}" <<'PY'
"""Download one GGUF file from the Hugging Face Hub into a local directory."""
import sys

from huggingface_hub import hf_hub_download

repo_id, filename, local_dir = sys.argv[1:4]
path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
print(f"[fetch] saved {path}")
PY
}

mkdir -p "${DEST}"
case "${PRESET}" in
  qwen)
    fetch "${FT_GGUF_REPO:-Qwen/Qwen3-8B-GGUF}" "${FT_GGUF_FILE:-Qwen3-8B-Q4_K_M.gguf}"
    ;;
  bge-m3|embed)
    fetch "${FT_GGUF_REPO:-gpustack/bge-m3-GGUF}" "${FT_GGUF_FILE:-bge-m3-Q8_0.gguf}"
    ;;
  all)
    fetch "Qwen/Qwen3-8B-GGUF" "Qwen3-8B-Q4_K_M.gguf"
    fetch "gpustack/bge-m3-GGUF" "bge-m3-Q8_0.gguf"
    ;;
  *)
    # Unknown preset: treat it as a raw `<org>/<repo> <file>` override pair.
    fetch "${FT_GGUF_REPO:?set FT_GGUF_REPO for a custom preset}" \
          "${FT_GGUF_FILE:?set FT_GGUF_FILE for a custom preset}"
    ;;
esac