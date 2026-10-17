#!/usr/bin/env bash
# Entrypoint for the `llm` image: a llama.cpp server exposing the
# OpenAI-compatible endpoint the pipeline's roles expect. One image serves two
# modes, so the chat model and the embedding model share the same build, the
# same ROCm runtime, and the same device plumbing:
#
#   FRICTIONTUNE_LLM_MODE=chat   (default) /v1/chat/completions  (Qwen, etc.)
#   FRICTIONTUNE_LLM_MODE=embed            /v1/embeddings        (bge-m3, etc.)
#
# Chat mode — point the pipeline at it by setting, in the repo .env:
#
#   FRICTIONTUNE_LLM_BASE_URL=http://127.0.0.1:8000/v1
#   FRICTIONTUNE_LLM_API_KEY=sk-local
#   FRICTIONTUNE_MODEL=<the --alias below>
#
# Embedding mode — point the *embedder* at it with:
#
#   FRICTIONTUNE_EMBED_BASE_URL=http://127.0.0.1:8001/v1
#   FRICTIONTUNE_EMBED_MODEL=<the --alias below>
#   FRICTIONTUNE_EMBED_API_KEY=sk-local
#
# and dropping a GGUF on disk in both cases:
#
#   FRICTIONTUNE_LLM_MODEL_PATH=/work/.toolchain/models/Qwen3-8B-Q4_K_M.gguf
#   FRICTIONTUNE_LLM_MODEL_PATH=/work/.toolchain/models/bge-m3-Q8_0.gguf
#
# Every knob is env-overridable; extra arguments are forwarded to llama-server.
#
# Hardware knobs (all optional):
#   FRICTIONTUNE_LLM_CTX          total context; slots get ctx/parallel (default 65536)
#   FRICTIONTUNE_LLM_PARALLEL     concurrent slots (default 4)
#   FRICTIONTUNE_LLM_NGL          layers on GPU, 999 = all (default 999)
#   FRICTIONTUNE_LLM_SPLIT_MODE   layer|row (default layer; right for no-NVLink)
#   FRICTIONTUNE_LLM_TENSOR_SPLIT e.g. "0.30,0.35,0.35" to divide across three GPUs
#   FRICTIONTUNE_LLM_THREADS      CPU threads (default: nproc)
#   FRICTIONTUNE_LLM_BATCH        logical (prefill) batch (default 4096)
#   FRICTIONTUNE_LLM_UBATCH       physical (prefill) batch (default 2048)
#   FRICTIONTUNE_LLM_KV_TYPE     KV cache type, e.g. q8_0 to halve KV and double ctx
#   FRICTIONTUNE_LLM_REASONING    on|off (default off)
#   FRICTIONTUNE_LLM_POOLING      embeddings pooling {none,mean,cls,last,rank}
#
# To fan one server across every card, give it all devices (do NOT set a single
# ROCR_VISIBLE_DEVICES) and leave split-mode at layer:
#   podman run ... --device /dev/kfd --device /dev/dri frictiontune:llm
set -euo pipefail

MODEL="${FRICTIONTUNE_LLM_MODEL_PATH:?set FRICTIONTUNE_LLM_MODEL_PATH to a .gguf file}"
ALIAS="${FRICTIONTUNE_LLM_ALIAS:-$(basename "${MODEL}" .gguf)}"
MODE="${FRICTIONTUNE_LLM_MODE:-chat}"
PORT="${FRICTIONTUNE_LLM_PORT:-8000}"
HOST="${FRICTIONTUNE_LLM_HOST:-0.0.0.0}"
# `--ctx-size` is the *total* server context; each of the `--parallel` slots gets
# ctx/parallel. Defaults give 4 slots × 16384 tokens of usable window, which
# covers the pipeline's largest prompts (analyst episodes) plus an 8k generation.
CTX="${FRICTIONTUNE_LLM_CTX:-65536}"
NGL="${FRICTIONTUNE_LLM_NGL:-999}"          # offload every layer to the GPU
PARALLEL="${FRICTIONTUNE_LLM_PARALLEL:-4}"  # concurrent slots = ctx / this per slot
# Spread across every visible GPU. `layer` split is right for cards without
# NVLink (our RX 7900 XTXs); `--tensor-split` defaults to a VRAM-proportional
# split when unset. THREADS sizes the CPU thread pool used for tokenisation and
# sampling; default to all cores.
SPLIT_MODE="${FRICTIONTUNE_LLM_SPLIT_MODE:-layer}"
TENSOR_SPLIT="${FRICTIONTUNE_LLM_TENSOR_SPLIT:-}"
THREADS="${FRICTIONTUNE_LLM_THREADS:-$(nproc)}"
# Prefill is the throughput bottleneck on a big prompt (long compaction inputs,
# multi-thousand-token analyst episodes): a large physical batch keeps the GPUs
# fed instead of serialising token by token.
BATCH="${FRICTIONTUNE_LLM_BATCH:-4096}"
UBATCH="${FRICTIONTUNE_LLM_UBATCH:-2048}"
# Optional KV-cache quantisation (`q8_0` halves the KV, so the same VRAM holds
# twice the context). Empty keeps the f16 default.
KV_TYPE="${FRICTIONTUNE_LLM_KV_TYPE:-}"

common=(
  --model "${MODEL}"
  --alias "${ALIAS}"
  --host "${HOST}"
  --port "${PORT}"
  --ctx-size "${CTX}"
  --n-gpu-layers "${NGL}"
  --parallel "${PARALLEL}"
  --split-mode "${SPLIT_MODE}"
  --threads "${THREADS}"
  --batch-size "${BATCH}"
  --ubatch-size "${UBATCH}"
)
[ -n "${TENSOR_SPLIT}" ] && common+=(--tensor-split "${TENSOR_SPLIT}")
[ -n "${KV_TYPE}" ] && common+=(--cache-type-k "${KV_TYPE}" --cache-type-v "${KV_TYPE}")

if [ "${MODE}" = "embed" ]; then
  # Embeddings are prompt-only, so the physical batch (`-ub`) is the throughput
  # knob: a whole retrieval batch is packed into one prompt-eval pass. Pooling
  # defaults to the model's own metadata (bge-m3 declares CLS).
  POOLING="${FRICTIONTUNE_LLM_POOLING:-}"
  args=(
    "${common[@]}"
    --embeddings
  )
  [ -n "${POOLING}" ] && args+=(--pooling "${POOLING}")
else
  # Reasoning is off by default: the analyst/judge/teacher roles want the JSON
  # answer, not a Qwen3 thinking trace, and suppressing it is faster and keeps
  # `message.content` clean. It is also the difference between a valid reply and
  # a truncation when a slot is nearly full. Set FRICTIONTUNE_LLM_REASONING=on to
  # re-enable it; thoughts then land in `message.reasoning_content`, never in
  # `content`.
  REASONING="${FRICTIONTUNE_LLM_REASONING:-off}"
  args=(
    "${common[@]}"
    --reasoning "${REASONING}"
    --reasoning-format deepseek
    --jinja
  )
fi

exec llama-server "${args[@]}" "$@"