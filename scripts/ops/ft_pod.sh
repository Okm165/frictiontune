#!/usr/bin/env bash
# Run frictiontune as a podman *pod*: the pipeline together with a monitor
# sidecar and, when needed, the replay servers and a local llama.cpp server, all
# sharing one network namespace and the bind-mounted repository.
#
#   scripts/ops/ft_pod.sh build            # build the pipeline images
#   scripts/ops/ft_pod.sh data             # local LLM (all GPUs) + pipeline → verify_rules
#   scripts/ops/ft_pod.sh train            # pipeline → train_sft on a free GPU
#   scripts/ops/ft_pod.sh replay           # base+adapter servers + pipeline → export
#   scripts/ops/ft_pod.sh all              # data → train → replay
#   scripts/ops/ft_pod.sh serve            # servers + monitor only
#   scripts/ops/ft_pod.sh status           # pod + container state
#   scripts/ops/ft_pod.sh report           # latest replay dashboard + run analysis
#   scripts/ops/ft_pod.sh logs [name]      # follow one container (default: monitor)
#   scripts/ops/ft_pod.sh shell [name]     # interactive shell in a container
#   scripts/ops/ft_pod.sh down             # tear the pod down
#
# The pipeline is run in *phases* because its GPU workloads do not all fit on the
# machine at once: the data phase wants every GPU for local inference, training
# wants every GPU for a torchrun data-parallel group (topology read from the
# hardware preset, `devices: 3`), and replay wants one GPU per served model. Each
# phase therefore hands the whole machine to the resource that is the bottleneck
# for that phase, and the idempotent orchestrator reuses the artifacts the
# previous phase wrote.
#
# This is the *imperative* sibling of container/pod.yaml. kube play cannot grant
# /dev/kfd under rootless podman, so GPU work goes through `podman run --device`.
#
# Everything is overridable by environment variable, and nothing is hard-coded
# to one vendor: AMD (ROCm) and NVIDIA (CUDA) are detected, the card count is
# read from the hardware preset, and the phase layout adapts to however many
# cards exist. The defaults below are a working 3-GPU layout: base server on
# GPU0:8100, adapter server on GPU1:8101, training across every card (DDP), and
# the replay-phase LLM on GPU2. With FT_LOCAL_LLM=1 the data phase serves the
# chat model locally: one layer-split server across all cards by default, or one
# data-parallel replica per card (set FT_LLM_SERVERS) when the model fits a card,
# in which case the client fans calls across them. A small bge-m3 embed server
# shares GPU0:8001, and the roles and embedder point at these servers — the whole
# data phase is then on-box.
set -euo pipefail

REPO="${FT_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
POD="${FT_POD:-frictiontune}"
PREFIX="${FT_IMAGE_PREFIX:-localhost/frictiontune}"
ENGINE="${FT_ENGINE:-podman}"

GPU_BASE="${FT_GPU_BASE:-0}"
GPU_ADAPTER="${FT_GPU_ADAPTER:-1}"
GPU_TRAIN="${FT_GPU_TRAIN:-2}"
GPU_LLM="${FT_GPU_LLM:-2}"
GPU_EMBED="${FT_GPU_EMBED:-0}"
BASE_PORT="${FT_BASE_PORT:-8100}"
ADAPTER_PORT="${FT_ADAPTER_PORT:-8101}"
LLM_PORT="${FT_LLM_PORT:-8000}"
EMBED_PORT="${FT_EMBED_PORT:-8001}"
BASE_SERVED_NAME="${FT_BASE_SERVED_NAME:-qwen3-8b}"
ADAPTER_SERVED_NAME="${FT_ADAPTER_SERVED_NAME:-qwen3-8b-lora}"
LLM_ALIAS="${FT_LLM_ALIAS:-qwen3-8b}"
EMBED_ALIAS="${FT_EMBED_ALIAS:-bge-m3}"

# Set FT_LOCAL_LLM=1 to run the llama.cpp `llm` image in the pod and point the
# pipeline's roles at it instead of a hosted endpoint. That selects the `local`
# environment overlay unless FT_ENVIRONMENT says otherwise.
USE_LOCAL_LLM="${FT_LOCAL_LLM:-0}"
ENVIRONMENT="${FT_ENVIRONMENT:-$([ "${USE_LOCAL_LLM}" = "1" ] && printf local || printf bitdeer)}"

# Context budget for the local llama.cpp server. `--ctx-size` is the *total*
# server context; each of the `--parallel` slots gets ctx/parallel. On three
# 24 GB cards, minus the ~5 GB of Q4 weights, the KV cache is the budget: f16 KV
# is ~144 KiB/token for this model, so the default `--cache-type-k/-v q8_0`
# roughly halves it to ~72 KiB/token and buys both more slots *and* more context
# than f16 would. The data phase spends it on 16 slots × 32k so a long compactor
# input and 16 concurrent episodes both fit and decode batches deeply; the replay
# phase, sharing a card with a served model, uses a smaller pool.
LLM_CTX="${FT_LLM_CTX:-524288}"
LLM_PARALLEL="${FT_LLM_PARALLEL:-16}"
LLM_KV_TYPE="${FT_LLM_KV_TYPE:-q8_0}"
LLM_REPLAY_CTX="${FT_LLM_REPLAY_CTX:-65536}"
LLM_REPLAY_PARALLEL="${FT_LLM_REPLAY_PARALLEL:-2}"
# The local embedder (bge-m3) is tiny and prompt-only, so it *co-resides* with
# the chat model rather than taking a phase of its own: it is pinned to one card
# and the chat model's `--tensor-split` is biased away from that card, so both
# fit with no idle VRAM. The split is *derived* from the detected card count and
# the embed card below (override with FT_LLM_TENSOR_SPLIT=a,b,c).
LLM_TENSOR_SPLIT="${FT_LLM_TENSOR_SPLIT:-}"
EMBED_CTX="${FT_EMBED_CTX:-16384}"
EMBED_PARALLEL="${FT_EMBED_PARALLEL:-4}"
# The physical prefill batch must hold a whole embedding input, not a chunk of it:
# llama.cpp rejects an input longer than `--ubatch-size` outright (500), and the
# client's char budget (8000 chars ≈ up to ~2700 code tokens) can exceed a small
# batch. Size it to the slot (ctx ÷ parallel = 4096 tokens) so any input the
# client can send fits.
EMBED_UBATCH="${FT_EMBED_UBATCH:-4096}"

# The canonical run invocation. Keep in sync with the host recipe so a container
# run and a host run resolve to the same run_id and share the artifact cache.
# Training topology (including the DDP world size) comes from the hardware
# preset, so it is not overridden here.
PIPELINE="${FT_PIPELINE:-v10}"
HARDWARE="${FT_HARDWARE:-3x24g}"
CORPUS_DB="${FT_CORPUS_DB:-.context/medium.db}"
EVAL_EXAMPLES="${FT_EVAL_EXAMPLES:-16}"
# The adapter the replay serves is the one *this* pipeline trained, so the glob is
# scoped to the pipeline id. A hardcoded `artifacts/v10/...` would serve a stale
# adapter from an earlier pipeline and the replay would measure the wrong model.
ADAPTER_GLOB="${FT_ADAPTER_GLOB:-artifacts/${PIPELINE}/train_sft/*/adapter}"
# NCCL moves gradients and parameters between ranks through POSIX shared memory
# (`/dev/shm/nccl-*`), and it allocates one segment per channel per rank: a
# 3-rank QLoRA job asks for ~10 MB segments and exhausts podman's 64 MB default
# with `No space left on device (28)` before the first optimizer step. The
# default below is declared, not consumed — /dev/shm is tmpfs, so only the pages
# NCCL actually touches are allocated.
SHM_SIZE="${FT_SHM_SIZE:-8g}"

RUN_ARGS=(
  --pipeline "${PIPELINE}" --hardware "${HARDWARE}" --environment "${ENVIRONMENT}"
  --set paths.corpus_db="${CORPUS_DB}"
  --set stages.evaluate_replay.max_examples="${EVAL_EXAMPLES}"
)

# ── accelerator abstraction ───────────────────────────────────────────────────
# The pod is vendor-neutral: the same phases run on AMD (ROCm) and NVIDIA (CUDA).
# Detection here yields three primitives every GPU container uses — the podman
# device grants, the per-card pin env, and the card count — so nothing below is
# AMD-specific. Force a choice with FT_ACCELERATOR=amd|nvidia, or replace the raw
# device grants entirely with FT_DEVICE_FLAGS.
detect_accelerator() {
  if [ -n "${FT_ACCELERATOR:-}" ]; then printf '%s' "${FT_ACCELERATOR}"; return; fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    printf 'nvidia'; return
  fi
  printf 'amd'
}
ACCELERATOR="${FT_ACCELERATOR:-$(detect_accelerator)}"

# The number of accelerators is normally the hardware preset's `devices`, so the
# declared topology and the pod's device grants agree. Fall back to the vendor
# probe when the preset is absent or says `auto`.
preset_devices() {
  local file="${REPO}/configs/hardware/${HARDWARE}.yaml"
  [ -f "${file}" ] || return 1
  awk -F': *' '/^devices:/{gsub(/[^0-9]/, "", $2); print $2; exit}' "${file}"
}
probe_devices() {
  case "${ACCELERATOR}" in
    nvidia) nvidia-smi -L 2>/dev/null | grep -c '^GPU' ;;
    *) ls /dev/dri/renderD* 2>/dev/null | wc -l ;;
  esac
}
GPU_COUNT="${FT_GPU_COUNT:-$(preset_devices 2>/dev/null || true)}"
case "${GPU_COUNT}" in ''|*[!0-9]*) GPU_COUNT="$(probe_devices)";; esac
[ "${GPU_COUNT}" -ge 1 ] 2>/dev/null || GPU_COUNT=1

# Clamp a requested card index onto the cards that exist, so a 1- or 2-card box
# still runs every phase (sharing a card) instead of passing a bogus index.
clamp_gpu() { local index="$1"; [ "${index}" -ge "${GPU_COUNT}" ] && printf '%s' "$((GPU_COUNT - 1))" || printf '%s' "${index}"; }

GPU_BASE="$(clamp_gpu "${GPU_BASE:-0}")"
GPU_ADAPTER="$(clamp_gpu "${GPU_ADAPTER:-1}")"
GPU_TRAIN="$(clamp_gpu "${GPU_TRAIN:-2}")"
GPU_LLM="$(clamp_gpu "${GPU_LLM:-2}")"
GPU_EMBED="$(clamp_gpu "${GPU_EMBED:-0}")"

# Derive the chat model's tensor split: even, except the embed card gets a little
# less so bge-m3's weights and KV fit beside the model's share of the layers.
tensor_split_for() {
  local count="$1" reserve="${2:-}"
  awk -v n="${count}" -v r="${reserve}" 'BEGIN{
    if (r == "") r = -1;
    share = (r < 0 ? 1.0 / n : 0.85 / n);
    rest  = (n > 1 ? (1.0 - share) / (n - 1) : share);
    out = "";
    for (i = 0; i < n; i++) { out = out (i ? "," : "") sprintf("%.2f", (i == r ? share : rest)) }
    print out;
  }'
}
[ -n "${LLM_TENSOR_SPLIT}" ] || LLM_TENSOR_SPLIT="$(tensor_split_for "${GPU_COUNT}" "${GPU_EMBED}")"

# ── local-serving topology ────────────────────────────────────────────────────
# The data phase serves the chat model one of two ways, chosen by FT_LLM_SERVERS:
#   1   one layer-split server across every visible card (the default; right when
#       the model does not fit a single card)
#   N   N data-parallel replicas, one per card, each on its own port. The client
#       fans calls across them (see `base_urls` in the LLM client). Right when the
#       model fits a card: the cards decode independently instead of pipelining
#       one batch through every card.
# The context and slot budgets are divided evenly, so per-slot context is
# unchanged: a replica gets ctx/N and parallel/N.
LLM_SERVERS="${FT_LLM_SERVERS:-1}"
# What the *current phase* actually started: the replica list for a data-parallel
# data phase, or the single URL for a layer-split server or a pinned replay
# server. `run_pipeline` and `wait_llm_healthy` read this, so the replay phase
# (one server on GPU_LLM) never advertises endpoints it did not start.
LLM_ACTIVE_BASE_URLS="http://127.0.0.1:${LLM_PORT}/v1"

# The chat endpoints for the configured topology, comma-separated.
llm_base_urls() {
  if [ "${LLM_SERVERS}" -le 1 ]; then printf 'http://127.0.0.1:%s/v1' "${LLM_PORT}"; return; fi
  local i out=""
  for i in $(seq 0 $((LLM_SERVERS - 1))); do
    out="${out}${out:+,}http://127.0.0.1:$((LLM_PORT + i * 10))/v1"
  done
  printf '%s' "${out}"
}

log() { printf '\033[1;36m[ft_pod]\033[0m %s\n' "$*"; }

# ── shared flags ──────────────────────────────────────────────────────────────
# The repo bind mount and the device grants are kept separate: the monitor needs
# neither the GPU nor a relabel, and `--security-opt label=disable` makes a
# plain (non-`:z`) mount of an SELinux-protected repo work under rootless podman.
pod_flags()     { printf '%s' "--pod ${POD}"; }
runtime_flags() { printf '%s' "--security-opt label=disable"; }
# Vendor device grants. AMD passes the KFD + DRM nodes; NVIDIA passes the GPUs
# through CDI. Override the whole grant with FT_DEVICE_FLAGS on an unusual host.
device_flags()  {
  if [ -n "${FT_DEVICE_FLAGS:-}" ]; then printf '%s' "${FT_DEVICE_FLAGS}"; return; fi
  case "${ACCELERATOR}" in
    nvidia) printf '%s' "--device nvidia.com/gpu=all" ;;
    *)      printf '%s' "--device /dev/kfd --device /dev/dri" ;;
  esac
}
mount_flags()   { printf '%s' "-v ${REPO}:/work -w /work"; }

# Pin one accelerator. AMD: ROCR_VISIBLE_DEVICES alone (setting it *together
# with* HIP_VISIBLE_DEVICES makes the HIP runtime enumerate zero devices).
# NVIDIA: NVIDIA_VISIBLE_DEVICES, with the same one-card semantics.
gpu_env_flag() {
  case "${ACCELERATOR}" in
    nvidia) printf '%s' "-e NVIDIA_VISIBLE_DEVICES=$1" ;;
    *)      printf '%s' "-e ROCR_VISIBLE_DEVICES=$1" ;;
  esac
}

load_env() {
  # Source the repo dotenv so the pod inherits the same endpoints as the host.
  if [ -f "${REPO}/.env" ]; then
    set -a; # shellcheck disable=SC1091
    . "${REPO}/.env"; set +a
  fi
}

base_weights_mount() {
  # If the base weights live outside the repo, bind them in at the same path.
  local w="${FRICTIONTUNE_BASE_WEIGHTS:-}"
  if [ -n "${w}" ] && [ "${w#/}" != "${w}" ] && [ -d "${w}" ] && [ "${w#"${REPO}"}" = "${w}" ]; then
    printf '%s' "-v ${w}:${w}:ro"
  fi
}

ensure_pod() {
  if ! ${ENGINE} pod exists "${POD}" >/dev/null 2>&1; then
    # Publish the chat-model ports for every configured replica, so a
    # data-parallel run (one server per card on LLM_PORT + 10·i) is reachable
    # from the host exactly like the single server is. In-pod callers reach them
    # over the shared network namespace regardless.
    local -a publish=(
      --publish "${BASE_PORT}:${BASE_PORT}"
      --publish "${ADAPTER_PORT}:${ADAPTER_PORT}"
      --publish "${EMBED_PORT}:${EMBED_PORT}"
    )
    local i
    for i in $(seq 0 $((LLM_SERVERS - 1))); do
      publish+=(--publish "$((LLM_PORT + i * 10)):$((LLM_PORT + i * 10))")
    done
    log "creating pod ${POD} (publishing base/adapter/embed + ${LLM_SERVERS} llm port(s) from ${LLM_PORT}, /dev/shm ${SHM_SIZE})"
    ${ENGINE} pod create --name "${POD}" --shm-size "${SHM_SIZE}" "${publish[@]}" >/dev/null
  fi
}

rm_container() { ${ENGINE} rm -f "$1" >/dev/null 2>&1 || true; }

# ── sidecar containers ────────────────────────────────────────────────────────
# start_serve_server <name> <gpu> <port> <served_name> [extra serve args...]
# The base and adapter servers differ only in their name, card, port, served
# name, and the adapter's extra `--adapter-glob`; everything else — the mount,
# the device grants, the weights — is identical, so it lives here once.
start_serve_server() {
  load_env
  local name="$1" gpu="$2" port="$3" served="$4"
  shift 4
  rm_container "${POD}-${name}"
  log "${name} server → GPU${gpu}:${port} (${FRICTIONTUNE_BASE_WEIGHTS:-?})"
  # shellcheck disable=SC2046
  ${ENGINE} run -d --name "${POD}-${name}" $(pod_flags) $(runtime_flags) $(device_flags) \
    $(mount_flags) \
    $(gpu_env_flag "${gpu}") \
    $(base_weights_mount) \
    "${PREFIX}:serve" \
    --model "${FRICTIONTUNE_BASE_WEIGHTS:?set FRICTIONTUNE_BASE_WEIGHTS}" \
    "$@" \
    --host 0.0.0.0 --served-name "${served}" --port "${port}" >/dev/null
}

start_base_server() {
  start_serve_server base "${GPU_BASE}" "${BASE_PORT}" "${BASE_SERVED_NAME}"
}

start_adapter_server() {
  start_serve_server adapter "${GPU_ADAPTER}" "${ADAPTER_PORT}" "${ADAPTER_SERVED_NAME}" \
    --adapter-glob "${ADAPTER_GLOB}"
}

start_monitor() {
  rm_container "${POD}-monitor"
  log "monitor → journal/runs (analyze --watch)"
  # --entrypoint bash: the image entrypoint is ft-cli, which would otherwise
  # swallow `bash -lc …` as a frictiontune subcommand.
  ${ENGINE} run -d --name "${POD}-monitor" $(pod_flags) $(runtime_flags) \
    $(mount_flags) \
    --entrypoint bash "${PREFIX}:data" \
    -lc 'exec python3 scripts/ops/analyze.py --root journal/runs --watch --interval 20' >/dev/null
}

# start_llm_server <ctx> <parallel> [gpu|all] [name_suffix] [port]
start_llm_server() {
  load_env
  local ctx="$1" parallel="$2" gpu="${3:-all}" suffix="${4:-}" port="${5:-${LLM_PORT}}"
  local name="${POD}-llm${suffix}"
  rm_container "${name}"
  local model="${FRICTIONTUNE_LLM_MODEL_PATH:?set FRICTIONTUNE_LLM_MODEL_PATH for FT_LOCAL_LLM=1}"
  # The value may be repo-relative (host view); inside the container the repo is
  # at /work, so anchor it there.
  case "${model}" in /*) ;; *) model="/work/${model}" ;; esac
  local pin=""
  # `all` deliberately leaves ROCR_VISIBLE_DEVICES unset so llama.cpp layer-splits
  # the model across every visible card.
  local split=""
  if [ "${gpu}" != "all" ]; then
    pin="$(gpu_env_flag "${gpu}")"
  else
    # Only a multi-card server takes a tensor split; with one card it is noise.
    split="-e FRICTIONTUNE_LLM_TENSOR_SPLIT=${LLM_TENSOR_SPLIT}"
  fi
  log "llm server ${name} → ${gpu}:${port} (ctx ${ctx}, ${parallel} slots × $((ctx / parallel)))"
  # shellcheck disable=SC2046
  ${ENGINE} run -d --name "${name}" $(pod_flags) $(runtime_flags) $(device_flags) \
    $(mount_flags) \
    ${pin} \
    -e FRICTIONTUNE_LLM_MODEL_PATH="${model}" \
    -e FRICTIONTUNE_LLM_ALIAS="${LLM_ALIAS}" \
    -e FRICTIONTUNE_LLM_PORT="${port}" \
    -e FRICTIONTUNE_LLM_CTX="${ctx}" \
    -e FRICTIONTUNE_LLM_PARALLEL="${parallel}" \
    -e FRICTIONTUNE_LLM_SPLIT_MODE="${FT_LLM_SPLIT_MODE:-layer}" \
    -e FRICTIONTUNE_LLM_KV_TYPE="${LLM_KV_TYPE}" \
    ${split} \
    "${PREFIX}:llm" >/dev/null
}

# start_chat_servers <ctx> <parallel> [gpu|all]
# The data-phase entry point: one replica per card when the model fits (so each
# card decodes its own stream) or a single layer-split server otherwise. A
# non-`all` gpu request (the replay phase, which shares cards with the served
# base/adapter models) is always a single server on that card.
start_chat_servers() {
  local ctx="$1" parallel="$2" gpu="${3:-all}"
  # Clear whichever topology ran last (the single server or N replicas) so a
  # switch from layer-split to data-parallel does not leave a stale container.
  stop_llm
  if [ "${gpu}" != "all" ] || [ "${LLM_SERVERS}" -le 1 ]; then
    # A pinned replay server, or a single layer-split server: one URL.
    LLM_ACTIVE_BASE_URLS="http://127.0.0.1:${LLM_PORT}/v1"
    start_llm_server "${ctx}" "${parallel}" "${gpu}"
    return
  fi
  # Data-parallel: identical model on each card, own port, own KV pool. Split the
  # context and slot budgets so each replica keeps the same per-slot window.
  LLM_ACTIVE_BASE_URLS="$(llm_base_urls)"
  local slots=$(( (parallel + LLM_SERVERS - 1) / LLM_SERVERS ))
  local rctx=$(( ctx / LLM_SERVERS ))
  log "data-parallel llm: ${LLM_SERVERS} replicas × ${slots} slots (ctx ${rctx} each)"
  local i
  for i in $(seq 0 $((LLM_SERVERS - 1))); do
    start_llm_server "${rctx}" "${slots}" "${i}" "-${i}" "$((LLM_PORT + i * 10))"
  done
}

# Wait until every chat endpoint in LLM_ACTIVE_BASE_URLS answers /health.
wait_llm_healthy() {
  local url port rc=0
  for url in ${LLM_ACTIVE_BASE_URLS//,/ }; do
    port="${url#http://127.0.0.1:}"; port="${port%%/*}"
    wait_healthy "${port}" llm || rc=1
  done
  return "${rc}"
}

# start_embed_server <ctx> <parallel> [gpu|all]
# Same image, MODE=embed: /v1/embeddings for the `dedup`/`decontaminate` tiers.
# Pinned to one card by default so it can share the machine with the chat model.
start_embed_server() {
  load_env
  local ctx="$1" parallel="$2" gpu="${3:-${GPU_EMBED}}"
  rm_container "${POD}-embed"
  local model="${FRICTIONTUNE_EMBED_MODEL_PATH:?set FRICTIONTUNE_EMBED_MODEL_PATH for a local embedder}"
  case "${model}" in /*) ;; *) model="/work/${model}" ;; esac
  log "embed server → GPU${gpu}:${EMBED_PORT} (ctx ${ctx}, ${parallel} slots × $((ctx / parallel)))"
  # shellcheck disable=SC2046
  ${ENGINE} run -d --name "${POD}-embed" $(pod_flags) $(runtime_flags) $(device_flags) \
    $(mount_flags) \
    $(gpu_env_flag "${gpu}") \
    -e FRICTIONTUNE_LLM_MODE=embed \
    -e FRICTIONTUNE_LLM_MODEL_PATH="${model}" \
    -e FRICTIONTUNE_LLM_ALIAS="${EMBED_ALIAS}" \
    -e FRICTIONTUNE_LLM_PORT="${EMBED_PORT}" \
    -e FRICTIONTUNE_LLM_CTX="${ctx}" \
    -e FRICTIONTUNE_LLM_PARALLEL="${parallel}" \
    -e FRICTIONTUNE_LLM_UBATCH="${EMBED_UBATCH}" \
    "${PREFIX}:llm" >/dev/null
}

stop_llm() {
  # Data-parallel runs several containers (`<pod>-llm-0`, `-1`, …); a layer-split
  # run just one (`<pod>-llm`). Remove every match.
  local name
  while read -r name; do
    [ -n "${name}" ] && rm_container "${name}"
  done < <(${ENGINE} ps -a --filter "name=${POD}-llm" --format '{{.Names}}' 2>/dev/null || true)
}
stop_embed()   { rm_container "${POD}-embed"; }
stop_servers() { rm_container "${POD}-base"; rm_container "${POD}-adapter"; }

wait_healthy() {
  local port="$1" name="$2" tries="${3:-90}"
  for _ in $(seq "${tries}"); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      log "${name} healthy on ${port}"; return 0
    fi
    sleep 2
  done
  log "WARNING: ${name} not healthy on ${port} after ${tries} tries"
  return 1
}

# ── pipeline runner ───────────────────────────────────────────────────────────
# run_pipeline <phase> <stage[,stage...]> <gpu|none> [extra run flags...]
run_pipeline() {
  local phase="$1" stages="$2" gpu="$3"; shift 3
  load_env
  local stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local out="${REPO}/journal/partial/pod_${PIPELINE}_${phase}_${stamp}.txt"
  local -a stage_flags=() extra=("$@")
  local name
  for name in ${stages//,/ }; do
    stage_flags+=(--stage "${name}")
  done

  local -a env_flags=()
  # Point the pipeline's roles at the in-pod llama.cpp. The `local` overlay reads
  # these names; the server is reachable at 127.0.0.1 because the pod shares a
  # network namespace. `LLM_ACTIVE_BASE_URLS` is a comma-separated replica list
  # when the model runs data-parallel, which the client round-robins; a single
  # URL otherwise.
  if [ "${USE_LOCAL_LLM}" = "1" ] && [ "${phase}" != "train" ]; then
    env_flags+=(
      -e FRICTIONTUNE_LOCAL_LLM_BASE_URL="${LLM_ACTIVE_BASE_URLS}"
      -e FRICTIONTUNE_LOCAL_LLM_MODEL="${LLM_ALIAS}"
      -e FRICTIONTUNE_LOCAL_LLM_API_KEY=sk-local
      -e FRICTIONTUNE_LOCAL_LLM_CONCURRENCY="${LLM_PARALLEL}"
    )
  fi
  # The embedder's *identity* is part of the pipeline's configuration and must be
  # the same in every phase, or the stage cache re-keys at each phase boundary
  # and a phase rebuilds the pipeline it was supposed to consume. Its *endpoint*
  # is only advertised where a server actually runs: the data phase, where
  # retrieval and dedup live. In replay the base model owns GPU0, so no embed
  # server is started.
  if [ "${USE_LOCAL_LLM}" = "1" ]; then
    env_flags+=(
      -e FRICTIONTUNE_EMBED_MODEL="${EMBED_ALIAS}"
      -e FRICTIONTUNE_EMBED_API_KEY=sk-local
    )
    if [ "${phase}" = "data" ]; then
      env_flags+=(
        -e FRICTIONTUNE_EMBED_BASE_URL="http://127.0.0.1:${EMBED_PORT}/v1"
      )
    fi
  fi
  # The phases that run a torch job (training, and export's adapter merge) hold a
  # large, churning working set: 4-bit weights, the vocabulary-wide logits, and
  # the backward buffers. `expandable_segments` would help the default caching
  # allocator fragment that set, but the ROCm build does not implement it —
  # asking for it only prints `expandable_segments not supported on this
  # platform`, and the trainer already offers it on CUDA builds alone
  # (frictiontune_train.training._stabilise_allocator). The window in the
  # hardware preset is what has to fit; see configs/hardware/3x24g.yaml.
  # Point the replay stage at the in-pod base/adapter servers, so `evaluate_replay`
  # compares the trained adapter against the model it came from — not two
  # unrelated hosted endpoints.
  if [ "${phase}" = "replay" ]; then
    env_flags+=(
      -e FRICTIONTUNE_TARGET_BASE_URL="http://127.0.0.1:${BASE_PORT}/v1"
      -e FRICTIONTUNE_TARGET_MODEL="${BASE_SERVED_NAME}"
      -e FRICTIONTUNE_TARGET_API_KEY=local
      -e FRICTIONTUNE_ADAPTER_BASE_URL="http://127.0.0.1:${ADAPTER_PORT}/v1"
      -e FRICTIONTUNE_ADAPTER_MODEL="${ADAPTER_SERVED_NAME}"
      -e FRICTIONTUNE_ADAPTER_API_KEY=local
    )
  fi

  local -a gpu_flag=()
  [ "${gpu}" != "none" ] && gpu_flag=($(gpu_env_flag "${gpu}"))

  log "pipeline[${phase}] → ${stages} (gpu=${gpu}); stream → ${out#"${REPO}"/}"
  # shellcheck disable=SC2046,SC2086
  ${ENGINE} run --rm --name "${POD}-pipeline" $(pod_flags) $(runtime_flags) $(device_flags) \
    $(mount_flags) \
    "${gpu_flag[@]}" \
    --env-file "${REPO}/.env" \
    "${env_flags[@]}" \
    $(base_weights_mount) \
    "${PREFIX}:train" \
    run "${stage_flags[@]}" --include-deps "${extra[@]}" "${RUN_ARGS[@]}" 2>&1 | tee "${out}"
  local rc="${PIPESTATUS[0]}"
  log "pipeline[${phase}] exited rc=${rc}; registering logs"
  python3 "${REPO}/scripts/ops/log_registry.py" register \
    --pipeline "${PIPELINE}" --stamp "pod_${phase}_${stamp}" --log "${out}" || true
  python3 "${REPO}/scripts/ops/analyze.py" --root "${REPO}/journal/runs" || true
  return "${rc}"
}

# ── commands ──────────────────────────────────────────────────────────────────
cmd_build() {
  log "building images from container/Containerfile (variant ${FT_VARIANT:-rocm})"
  make -C "${REPO}/container" all \
    ENGINE="${ENGINE}" IMAGE_PREFIX="${PREFIX}" VARIANT="${FT_VARIANT:-rocm}" "$@"
}

# Phase 1 — the LLM-bound data pipeline. Hand the whole machine to llama.cpp.
# Two invocations cover *every* stage that needs a model: `decontaminate` walks
# the friction→mine→judge→build_sft→curate→decontaminate branch, `verify_rules`
# walks the compile_rules→verify_rules branch. Both are cached afterwards, so
# the later phases never need an LLM.
cmd_data() {
  ensure_pod
  stop_servers
  if [ "${USE_LOCAL_LLM}" = "1" ]; then
    start_chat_servers "${LLM_CTX}" "${LLM_PARALLEL}" all
    wait_llm_healthy || true
    start_embed_server "${EMBED_CTX}" "${EMBED_PARALLEL}"
    wait_healthy "${EMBED_PORT}" embed || true
  fi
  start_monitor
  run_pipeline data decontaminate none
  run_pipeline data verify_rules none
}

# Phase 2 — training. Evict every server so the training group owns the machine;
# the backend reads the hardware preset's `devices` and launches one torchrun
# rank per card, so no single GPU is pinned here.
cmd_train() {
  ensure_pod
  stop_llm
  stop_embed
  stop_servers
  start_monitor
  # Training stops every llama.cpp server to give the job the whole machine, so
  # the phase has no model endpoint at all. `--cached-deps` makes that explicit:
  # the phase may run only the stages named here. `fetch_base` probes the base
  # model and needs no endpoint, and `train_sft` reads the dataset
  # `decontaminate` already wrote; if any *other* upstream stage is missing the
  # run stops naming it instead of re-running an LLM stage against an endpoint
  # that is not there.
  run_pipeline train fetch_base,train_sft none --cached-deps
}

cmd_curve() {
  # The learning-curve instrument (GAP-093, Tier C gated): re-train on growing
  # subsets of decontaminate's training set with train_sft's hyperparameters and
  # tabulate the best val loss per size. Needs GPUs (it is four training runs),
  # needs no endpoint (it calls no LLM), and needs decontaminate + fetch_base
  # cached — the same preconditions as the train phase.
  ensure_pod
  stop_llm
  stop_embed
  stop_servers
  start_monitor
  run_pipeline curve fetch_base,learning_curve none --cached-deps --allow-gated
}

# Phase 3 — replay + export. Two served models on GPU0/1; the judge (if local)
# and nothing else on GPU2.
cmd_replay() {
  ensure_pod
  stop_llm
  stop_embed
  start_base_server
  start_adapter_server
  if [ "${USE_LOCAL_LLM}" = "1" ]; then
    start_chat_servers "${LLM_REPLAY_CTX}" "${LLM_REPLAY_PARALLEL}" "${GPU_LLM}"
  fi
  start_monitor
  wait_healthy "${BASE_PORT}" base || true
  wait_healthy "${ADAPTER_PORT}" adapter || true
  if [ "${USE_LOCAL_LLM}" = "1" ]; then
    wait_llm_healthy || true
  fi
  # The replay phase means "measure and package the adapter that training just
  # produced", so `evaluate_replay` and `export` are named explicitly and
  # everything below them must already be cached: a replay must not silently
  # rebuild the data pipeline it is supposed to be measuring.
  run_pipeline replay evaluate_replay none --cached-deps
  # `export` merges the LoRA back into the base weights on the GPU, and the base
  # and adapter servers the replay just measured still hold it: the merge died
  # with `HIP out of memory. Tried to allocate 192.00 MiB` while only 2.21 GiB
  # was allocated by PyTorch, because llama.cpp owned the rest of the card. The
  # merge needs no server, so hand it the machine.
  stop_servers
  run_pipeline replay export none --cached-deps
}

cmd_all() {
  # Each phase hands the whole machine to its own bottleneck and persists its
  # artifacts before the next begins. A failed phase stops the chain on purpose:
  # training on a half-written data phase, or replaying a missing adapter, would
  # produce an artifact that looks valid and is not.
  cmd_data || return 1
  cmd_train || return 1
  cmd_replay || return 1
}

cmd_serve() {
  ensure_pod
  start_base_server
  start_adapter_server
  if [ "${USE_LOCAL_LLM}" = "1" ]; then
    start_chat_servers "${LLM_CTX}" "${LLM_PARALLEL}" all
    start_embed_server "${EMBED_CTX}" "${EMBED_PARALLEL}"
  fi
  start_monitor
  wait_healthy "${BASE_PORT}" base || true
  wait_healthy "${ADAPTER_PORT}" adapter || true
  log "servers up; tail with: scripts/ops/ft_pod.sh logs base|adapter|llm|embed|monitor"
}

cmd_status() {
  ${ENGINE} pod ps --filter "name=${POD}" || true
  ${ENGINE} ps --pod --filter "pod=${POD}" || true
}

# Print the latest replay result for this pipeline and re-run the anomaly/gap
# inference. This is the "is it improving?" read-out after a cycle: the friction
# drop from evaluate_replay, then the operational analysis across all runs.
cmd_report() {
  local dir
  dir="$(ls -dt "${REPO}/artifacts/${PIPELINE}"/evaluate_replay/*/ 2>/dev/null | head -1)"
  if [ -n "${dir}" ]; then
    log "replay ${PIPELINE}: ${dir#"${REPO}"/}"
    cat "${dir}report.md" 2>/dev/null || true
  else
    log "no evaluate_replay artifact for ${PIPELINE}"
  fi
  # Regenerate the analysis over the run registry, then show its head.
  python3 "${REPO}/scripts/ops/analyze.py" --root "${REPO}/journal/runs" >/dev/null 2>&1 || true
  [ -f "${REPO}/journal/runs/ANALYSIS.md" ] && sed -n '1,40p' "${REPO}/journal/runs/ANALYSIS.md"
}

cmd_logs() {
  local name="${1:-monitor}"
  ${ENGINE} logs -f "${POD}-${name}"
}

cmd_shell() {
  local name="${1:-data}"
  ${ENGINE} exec -it "${POD}-${name}" bash
}

cmd_down() {
  log "tearing down pod ${POD}"
  ${ENGINE} pod rm -f "${POD}" >/dev/null 2>&1 || true
}

action="${1:-}"; shift || true
case "${action}" in
  build)  cmd_build "$@" ;;
  data)   cmd_data "$@" ;;
  train)  cmd_train "$@" ;;
  curve)  cmd_curve "$@" ;;
  replay) cmd_replay "$@" ;;
  all)    cmd_all "$@" ;;
  serve)  cmd_serve "$@" ;;
  status) cmd_status "$@" ;;
  report) cmd_report "$@" ;;
  logs)   cmd_logs "$@" ;;
  shell)  cmd_shell "$@" ;;
  down)   cmd_down "$@" ;;
  *)
    echo "usage: $0 {build|data|train|replay|all|serve|status|report|logs [name]|shell [name]|down}" >&2
    exit 2
    ;;
esac