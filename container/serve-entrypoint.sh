#!/usr/bin/env bash
# Entrypoint for the `serve` image: one OpenAI-compatible chat server per
# process, standing in for vLLM/SGLang during the replay stage.
#
# Usage (inside a shared pod network namespace so the pipeline reaches it):
#
#   # base model on one GPU
#   ft-serve --model "$FRICTIONTUNE_BASE_WEIGHTS" --host 0.0.0.0 \
#            --served-name qwen3-8b --port 8100
#
#   # LoRA adapter produced by train_sft, on another GPU
#   ft-serve --model "$FRICTIONTUNE_BASE_WEIGHTS" \
#            --adapter-glob 'artifacts/<pipeline>/train_sft/*/adapter' \
#            --host 0.0.0.0 --served-name qwen3-8b-lora --port 8101
#
# Bind 0.0.0.0 inside a pod so the published port (host → pod netns) routes.
# Pin a GPU per server with ROCR_VISIBLE_DEVICES alone — setting it together
# with HIP_VISIBLE_DEVICES makes the HIP runtime enumerate zero devices.
set -euo pipefail

# Prefer the mounted (live) server script, falling back to the baked copy.
_script="/work/scripts/ops/serve_openai.py"
if [ ! -f "${_script}" ]; then
  _script="/opt/frictiontune/serve_openai.py"
fi

if [ -d /work ]; then
  cd /work
fi

exec python3 "${_script}" "$@"