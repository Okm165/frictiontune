# frictiontune containers

One `Containerfile` builds a **cascade** of images — one per pipeline — so the
heavy GPU-toolchain layer is created once and inherited. A shared OS layer forks
into the Python pipeline chain and the llama.cpp inference server:

```
os ──▶ base ──▶ data ──▶ train ──▶ serve
                     └──▶ dev
   └──▶ llm
```

The cascade is **vendor-neutral**: the default parent is AMD ROCm, and swapping
three build args builds the identical cascade on NVIDIA CUDA. See
[AMD or NVIDIA](#amd-or-nvidia) below.

| pipeline image | `--target` | what it runs | GPU |
| --- | --- | --- | --- |
| **data** | `data` | `fetch_base → ingest → normalize → segment → friction_detect → mine → judge → build_sft → curate → decontaminate → compile_rules → verify_rules` | no |
| **train** | `train` | the above **+** `train_sft`, `evaluate_replay`, `export` | yes |
| **serve** | `serve` | the OpenAI-compatible `target`/`adapter` replay servers | yes |
| **llm** | `llm` | a llama.cpp server for the `analyst`/`judge`/`teacher` roles (local Qwen) | yes |
| **dev** | `dev` | the four quality gates (`ruff`, `format`, `mypy`, `pytest`) | no |

## Why the cascade is laid out this way

* `os` is the **only** stage that touches `apt`; it holds the vendor GPU
  userspace (ROCm or CUDA), the build tools and git. Everything forks from it, so
  no other stage pays for OS setup twice.
* `base` adds `uv` and an empty `/opt/venv`. Every heavy Python wheel lives in a
  later layer, so changing a pipeline's dependencies never invalidates the OS.
* `data` installs the light `frictiontune` package (pydantic, structlog, openai,
  …). It is the image for the CPU/LLM half of the pipeline and needs no GPU.
* `train` **inherits from `data`** and adds the vendor PyTorch stack from the GPU
  wheel index plus `frictiontune-train`. Because it cascades, the `data` layers
  are reused rather than rebuilt.
* `serve` **inherits from `train`** (it needs the same torch/transformers/peft)
  and adds only `aiohttp` and the replay server.
* `dev` also inherits from `train` and adds the lint/type/test tooling.
* `llm` branches straight off `os` — a C++ inference server shares the OS and GPU
  userspace but none of the Python pipeline weight. It is independent of the
  `base → train` chain, so building it never rebuilds the Python layers.

Every version is an `ARG` (see below). The resolved values are written into the
image at `/opt/frictiontune/build-args.txt` **and** into the OCI labels, so an
image documents its own build:

```bash
podman run --rm frictiontune:train cat /opt/frictiontune/build-args.txt
```

## Build

```bash
# from the repository root; the build context is the repo
podman build -f container/Containerfile --target data  -t frictiontune:data  .
podman build -f container/Containerfile --target train -t frictiontune:train .
podman build -f container/Containerfile --target serve -t frictiontune:serve .
podman build -f container/Containerfile --target dev   -t frictiontune:dev   .
podman build -f container/Containerfile --target llm   -t frictiontune:llm   .
```

`make -C container all` builds every pipeline image, and
`make -C container <target>` builds one. Docker works too:
`docker build -f container/Containerfile --target train -t frictiontune:train .`

### Quality gates in the container

`dev` exists so the four gates run against the *same* toolchain the pipeline
uses, without a host virtualenv. Because the repo is bind-mounted at `/work` and
`packages/*/src` shadows the baked install, this is the live tree, not a copy:

```bash
podman run --rm --security-opt label=disable -v "$PWD":/work -w /work \
  --entrypoint bash frictiontune:dev -lc '
    ruff check packages tests &&
    ruff format --check packages tests &&
    mypy packages/frictiontune/src/frictiontune \
         packages/frictiontune-train/src/frictiontune_train tests &&
    pytest -q'
```

### AMD or NVIDIA

The `os` parent, the PyTorch wheel index and the llama.cpp backend are the only
vendor-specific inputs, and all three are build args. `make` bundles the NVIDIA
set behind one variable:

```bash
make -C container all VARIANT=cuda            # NVIDIA (default CUDA_ARCH=89)
make -C container all VARIANT=cuda CUDA_ARCH=90
make -C container all                          # AMD ROCm (default)
```

or, with raw `podman build`:

```bash
# AMD (ROCm) — the defaults, spelled out
podman build -f container/Containerfile --target train -t frictiontune:train \
  --build-arg OS_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.0-complete \
  --build-arg VARIANT=rocm \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/rocm7.0 \
  --build-arg TORCH_VERSION=2.10.0+rocm7.0 \
  --build-arg TORCH_EXTRA_PACKAGES=triton-rocm==3.6.0 \
  --build-arg GGML_BACKEND=HIP \
  --build-arg 'GGML_EXTRA_ARGS=-DAMDGPU_TARGETS=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm' .

# NVIDIA (CUDA)
podman build -f container/Containerfile --target train -t frictiontune:train \
  --build-arg OS_IMAGE=nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04 \
  --build-arg VARIANT=cuda \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 \
  --build-arg TORCH_VERSION=2.10.0 \
  --build-arg TORCH_EXTRA_PACKAGES= \
  --build-arg GGML_BACKEND=CUDA \
  --build-arg 'GGML_EXTRA_ARGS=-DCMAKE_CUDA_ARCHITECTURES=89' .
```

`TORCH_EXTRA_PACKAGES` is empty for CUDA (torch bundles Triton); ROCm needs the
separate `triton-rocm`. `AMDGPU_TARGETS` only applies to the HIP backend;
`CUDA_ARCH` only to the CUDA backend. The resolved set is recorded in
`/opt/frictiontune/build-args.txt` and the `com.frictiontune.variant` label, so
`podman inspect` tells you which vendor an image was built for.

At *run* time nothing is vendor-specific either: `ft_pod.sh` detects the
accelerator and passes the right device grants (`/dev/kfd`+`/dev/dri` for AMD,
CDI `nvidia.com/gpu=all` for NVIDIA) and the right per-card pin env
(`ROCR_VISIBLE_DEVICES` vs `NVIDIA_VISIBLE_DEVICES`). Force it on a mixed host
with `FT_ACCELERATOR=amd|nvidia`, or replace the grants entirely with
`FT_DEVICE_FLAGS`.

### Build arguments

Override any of these with `--build-arg NAME=value` (or `make … ARG=value`).

| arg | default | purpose |
| --- | --- | --- |
| `VARIANT` | `rocm` | vendor default selector (`rocm`/`cuda`); recorded as a label |
| `OS_IMAGE` | `docker.io/rocm/dev-ubuntu-24.04:7.0-complete` | vendor OS parent |
| `ROCM_IMAGE` / `CUDA_IMAGE` | ROCm 7.0 / `nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04` | canned parents for `OS_IMAGE` |
| `PYTHON_VERSION` | `3.12` | interpreter for `/opt/venv` |
| `UV_VERSION` | `0.13.0` | pinned `uv` |
| `UV_IMAGE` | `ghcr.io/astral-sh/uv:0.13.0` | source of the uv static binary |
| `TORCH_INDEX` | `https://download.pytorch.org/whl/rocm7.0` | vendor wheel index |
| `TORCH_VERSION` | `2.10.0+rocm7.0` | vendor PyTorch build |
| `TORCH_EXTRA_PACKAGES` | `triton-rocm==3.6.0` | vendor extras (empty for CUDA) |
| `TRANSFORMERS_VERSION` | `5.19.0` | HF transformers |
| `TRL_VERSION` | `1.15.0` | TRL (SFT trainer) |
| `PEFT_VERSION` | `0.21.2` | LoRA/PEFT |
| `DATASETS_VERSION` | `5.1.0` | HF datasets |
| `ACCELERATE_VERSION` | `1.15.0` | accelerate |
| `BITSANDBYTES_VERSION` | `0.50.2` | 4-bit QLoRA |
| `SENTENCEPIECE_VERSION` | `0.2.2` | tokeniser |
| `PYPI_INDEX` | `https://pypi.org/simple` | everything else |
| `AMDGPU_TARGETS` | `gfx1100` | GPU arch for the llama.cpp HIP build (RX 7900 XTX = `gfx1100`) |
| `GGML_BACKEND` | `HIP` | llama.cpp backend (`HIP`/`CUDA`) |
| `GGML_EXTRA_ARGS` | `-DAMDGPU_TARGETS=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm` | vendor CMake flags for the llama.cpp build |
| `LLAMA_CPP_REF` | `master` | llama.cpp git ref to build the `llm` image from |
| `HOST_UID` / `HOST_GID` | `1000` | recorded for reference |

## Run

The image supplies the toolchain; the **repository is bind-mounted at `/work`**
and its `packages/*/src` are placed ahead of the baked install on `PYTHONPATH`
(see `entrypoint.sh`). Two consequences:

1. Pipeline **code stays live**: edit the tree, re-run — no rebuild.
2. The **run-id contract is preserved**: `code_revision()` hashes the mounted
   repo's git state, so a container run and a host run of the same tree resolve
   to the same `run_id` and share the artifact cache.

A one-shot run of the whole pipeline, with the GPU and the model env exposed:

```bash
podman run --rm --device /dev/kfd --device /dev/dri \
  --security-opt label=disable \
  -v "$PWD":/work -w /work \
  --env-file .env \
  frictiontune:train \
  run --stage export --include-deps \
    --pipeline v10 --hardware 3x24g --environment bitdeer \
    --set paths.corpus_db=.context/medium.db
```

> Rootless podman on Fedora needs `--security-opt label=disable` (or a `:z`
> relabel) for the SELinux-protected bind mount. GPU access is vendor-specific:
> AMD needs `/dev/kfd` **and** `/dev/dri` (pin a card with `ROCR_VISIBLE_DEVICES`),
> NVIDIA passes the GPUs through CDI (`--device nvidia.com/gpu=all`) and pins with
> `NVIDIA_VISIBLE_DEVICES`. `ft_pod.sh` chooses the right grants automatically.

## A fully-local LLM (llama.cpp + Qwen)

The pipeline's roles all speak OpenAI. By default they talk to a hosted endpoint
(e.g. Bitdeer); the `llm` image replaces that with a local llama.cpp server, so
nothing leaves the box. The `local` environment overlay
(`configs/environments/local.yaml`) points every judgement, generation, and
compaction role at it and raises role concurrency to match the server's slots.

```bash
# 1. fetch the GGUFs into the repo (bind-mounted at /work, so they are visible)
scripts/ops/fetch_gguf.sh qwen       # Qwen3-8B Q4_K_M chat model (~5 GB)
scripts/ops/fetch_gguf.sh bge-m3     # bge-m3 Q8_0 embedder     (~0.6 GB)
```

### Fan one server across every GPU

llama.cpp layer-splits a model across all visible cards with
`--split-mode layer` and a VRAM-proportional `--tensor-split`. Give the
container **all** devices and do **not** pin `ROCR_VISIBLE_DEVICES`:

```bash
podman run --rm --device /dev/kfd --device /dev/dri \
  --security-opt label=disable \
  -v "$PWD":/work -w /work \
  -e FRICTIONTUNE_LLM_MODEL_PATH=/work/.toolchain/models/Qwen3-8B-Q4_K_M.gguf \
  -e FRICTIONTUNE_LLM_ALIAS=qwen3-8b \
  -e FRICTIONTUNE_LLM_CTX=262144 \
  -e FRICTIONTUNE_LLM_PARALLEL=8 \
  -p 8000:8000 \
  frictiontune:llm
```

`--ctx-size` is the **total** server context and is divided across `--parallel`
slots, so `CTX=262144 PARALLEL=8` gives 8 concurrent requests of 32768 tokens
each. The KV budget is what caps this: Qwen3-8B costs ≈144 KiB per token, so
262144 tokens ≈ 36 GiB of KV — which is why the model must be split across
several cards. `ft-llm` (the entrypoint) reads `FRICTIONTUNE_LLM_MODEL_PATH`,
`_ALIAS`, `_PORT`, `_CTX`, `_NGL` (GPU layers), `_PARALLEL` (slots),
`_SPLIT_MODE`, `_TENSOR_SPLIT`, `_THREADS`, `_MODE`, and `_REASONING`; anything
else is forwarded to `llama-server`.

`ft_pod.sh` sizes this for you (see below); with three 24 GB cards it launches
all three GPUs at `CTX=262144 PARALLEL=8` for the data phase and a smaller pool
for replay, when a card is shared with a served model.

### The same image serves embeddings

`FRICTIONTUNE_LLM_MODE=embed` turns the identical image into a llama.cpp
`/v1/embeddings` server, so the embedder (bge-m3, used for T3/T4 retrieval and
duplicate detection) also stays on the box. It is prompt-only and tiny, so it
shares a card with the chat model instead of taking a phase of its own:

```bash
podman run --rm --device /dev/kfd --device /dev/dri \
  --security-opt label=disable \
  -v "$PWD":/work -w /work \
  -e ROCR_VISIBLE_DEVICES=0 \
  -e FRICTIONTUNE_LLM_MODE=embed \
  -e FRICTIONTUNE_LLM_MODEL_PATH=/work/.toolchain/models/bge-m3-Q8_0.gguf \
  -e FRICTIONTUNE_LLM_ALIAS=bge-m3 \
  -e FRICTIONTUNE_LLM_CTX=16384 \
  -e FRICTIONTUNE_LLM_PARALLEL=4 \
  -p 8001:8001 \
  frictiontune:llm
```

Point the pipeline at it with `FRICTIONTUNE_EMBED_BASE_URL=http://127.0.0.1:8001/v1`,
`FRICTIONTUNE_EMBED_MODEL=bge-m3`, `FRICTIONTUNE_EMBED_API_KEY=sk-local`. In embed
mode the throughput knob is the physical batch (`FRICTIONTUNE_LLM_UBATCH`), and
pooling follows the model's metadata unless `FRICTIONTUNE_LLM_POOLING` overrides
it. `ft_pod.sh data` does all of this automatically under `FT_LOCAL_LLM=1`.

## Phased, all-GPU runs

The pipeline's GPU workloads do not fit on the machine at once — local inference
wants every card, training wants every card (DDP) to itself, and replay wants one
card per served model. `ft_pod.sh` therefore runs it in **phases** and lets the
idempotent orchestrator reuse the artifacts each phase leaves behind:

| command | what runs | GPU layout |
| --- | --- | --- |
| `ft_pod.sh data` | `decontaminate` + `verify_rules` — every LLM-bound stage | llama.cpp across **all** GPUs |
| `ft_pod.sh train` | `train_sft` | **DDP across all cards**, all servers evicted |
| `ft_pod.sh replay` | `evaluate_replay` + `export` | base → GPU0, adapter → GPU1, judge → GPU2 |
| `ft_pod.sh all` | the three phases above, in order | — |

The training phase is **DDP**: `torch.distributed.run` launches one process per
card with `--nproc_per_node` taken from the hardware preset's `devices` (not a
hardcoded count), so `devices: 3` gives a 3-rank run whose effective batch is
`per_device_batch × grad_accum × 3`. The adapter is written by rank 0 only. On
consumer AMD cards RCCL needs `NCCL_P2P_DISABLE=1` (no working PCIe P2P); the
launcher sets it, plus `NCCL_IB_DISABLE=1`, automatically. See
`packages/frictiontune-train/src/frictiontune_train/distributed.py` for the single
source of the topology.

```bash
# fully local, whole machine, larger corpus
FT_LOCAL_LLM=1 FT_CORPUS_DB=.context/large.db scripts/ops/ft_pod.sh all

# hosted judgement, local replay servers only
scripts/ops/ft_pod.sh data && scripts/ops/ft_pod.sh train && scripts/ops/ft_pod.sh replay
```

The same flags (`--pipeline`, `--hardware`, `--environment`, the `--set`
overrides, `FT_CORPUS_DB`) are passed to every phase, so all phases resolve the
same `run_id` and the second and third phases skip the first phase's work.

## The pod

`pod.yaml` defines the pod with a **pipeline container and a monitor sidecar**
that share a network namespace and the `/work` volume. The sidecar streams the
run's log registry and regenerates the anomaly/gap analysis live. Its only
machine-specific value is the repo path, supplied as `${FT_REPO}`:

```bash
make -C container pod                    # render ${FT_REPO} and kube play
make -C container pod-down               # tear it down

# equivalent, without make:
FT_REPO=$PWD envsubst < container/pod.yaml | podman kube play -
FT_REPO=$PWD envsubst < container/pod.yaml | podman kube down -
```

Because the pod shares a namespace, the replay servers (`serve` image) can run
as siblings of the pipeline and be reached at `127.0.0.1:8100` / `:8101`.

## Observability

Every run registers its per-stage logs and a machine-readable index:

```bash
python3 scripts/ops/log_registry.py backfill              # register journal/partial/*.txt
python3 scripts/ops/log_registry.py show --pipeline v10   # stage table per run
python3 scripts/ops/analyze.py --root journal/runs        # anomaly + gap report
```

The registry lands under `journal/runs/<pipeline>/<stamp>/` (`run.log`,
`index.jsonl`, `summary.json`, `stages/<stage>.log`) and the analyzer writes
`journal/runs/ANALYSIS.md`, `anomalies.json` and `gaps.md`. The monitor sidecar
just loops the analyzer, so the reports are always current.