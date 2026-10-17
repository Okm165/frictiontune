# Development

How to work on `frictiontune` itself.

## Layout

```
frictiontune/
├── pyproject.toml              # uv workspace root (virtual) + shared tool config
├── configs/                    # layered configuration (no machine specifics)
├── container/                  # Containerfile cascade + pod spec (one image per pipeline)
├── scripts/                    # ops tooling: pod runner, log registry, anomaly analysis
├── docs/
├── packages/
│   ├── frictiontune/           # light data pipeline + CLI
│   │   ├── pyproject.toml
│   │   └── src/frictiontune/
│   │       ├── core/           # deterministic plumbing only
│   │       ├── config/         # models + layered loader
│   │       ├── llm/            # client, gateway, prompts/library/*.md
│   │       ├── domain/         # strict pydantic domain + response models
│   │       ├── corpus/         # SQLite readers
│   │       ├── backends/       # protocols + registry + embedding
│   │       ├── pipeline/       # spec, stage, registry, dag, context, orchestrator
│   │       ├── stages/         # core stages + _extensions
│   │       ├── sft/  reporting/
│   │       ├── tui/            # optional Textual viewer
│   │       ├── application.py  # wires everything together
│   │       └── cli.py          # Typer entry point
│   └── frictiontune-train/     # GPU side (optional install)
│       └── src/frictiontune_train/
│           ├── plugin.py       # entry point: register backends
│           ├── training.py  sft.py  grpo.py  loaders.py
│           ├── reward.py  peft_export.py  rollout.py  lazy.py
└── tests/
```

## Setup

```bash
uv sync --all-packages
```

Developer tools (ruff, mypy, pytest, pre-commit) come from the workspace `dev`
dependency group at the repository root, so a normal sync installs them.

The GPU package is only installed when you need it:

```bash
uv sync --package frictiontune-train --extra train
```

## Lint, format, types

The shared tool config lives in the root `pyproject.toml` and applies to every
member.

```bash
uv run ruff check packages tests
uv run ruff format packages tests
uv run mypy packages/frictiontune/src/frictiontune packages/frictiontune-train/src/frictiontune_train tests
```

Ruff is strict: `E,F,W,I,N,UP,B,A,C4,T20,SIM,RET,PTH,RUF,ANN,TCH,D`. Notable
settings:

* **`TCH` (typing-only imports)** is configured with
  `runtime-evaluated-base-classes` for the pydantic bases and
  `runtime-evaluated-decorators`. Without this, moving a field type into
  `TYPE_CHECKING` would break pydantic at import time. Typer's
  `typer.Option`/`typer.Argument` are `extend-immutable-calls` so `B008` stays
  quiet for the legitimate command-signature pattern.
* `E501` is **not** ignored by hand; long lines are split. `ruff format` governs
  code line length and lint governs the rest.

mypy runs in `strict` mode with the `pydantic.mypy` plugin, plus `extra_checks`
and the `redundant-expr` / `truthy-bool` / `ignore-without-code` error codes, so
every suppression must name its error code. `textual` and the torch stack ship
no stubs and are covered by `ignore_missing_imports` overrides. The test suite is
type-checked under the same policy. `disallow_subclassing_any` is relaxed only
for `frictiontune.tui.*`, where Textual's bases resolve to `Any`.

## Tests

```bash
uv run pytest
```

Tests live in `tests/` and cover deterministic plumbing: JSON extraction,
hashing, retry, artifact suffix/kind round-tripping, config expansion and empty
key pruning, prompt rendering, taxonomy validation, and dashboard direction.
They are marked so you can skip the slow or environment-dependent ones:

* `network` — would call an external LLM endpoint; skipped by default.
* `gpu` — requires a GPU.

Tests must not require a live LLM or a GPU to pass.

### Measuring prompt quality

Prompt changes should be justified by a number, not a feeling. The golden-episode
harness scores the friction analyst against hand-labelled episodes and reports
precision/recall/F1, exact-episode match, and form/cause accuracy:

```bash
uv run frictiontune eval-prompts --hardware 3x24g
```

The fixture lives at
`packages/frictiontune/src/frictiontune/evaluation/fixtures/golden_episodes.jsonl`
(one JSON object per line: `id`, `transcript`, `expected` leaves). Its labels must
match `configs/friction_taxonomy.yaml`, which `tests/test_evaluation.py` enforces.
The scoring maths is pure and unit-tested offline; only the command needs an
endpoint. Grow the fixture whenever a real failure is found.

## Adding a stage

1. Create `packages/frictiontune/src/frictiontune/stages/<name>.py`.
2. Subclass `Stage` and declare `spec = StageSpec(...)` and `settings_model`.
3. Implement `async def run(self, ctx) -> StageResult`, reading inputs via
   `ctx.catalog` and writing via `ctx.emit_*`.
4. Register it in `stages/__init__.py` (`CORE_STAGES` for Tier A, or via
   `register_extension_stages`).
5. Add edges in `pipeline/dag.py` (`CORE_EDGES` or `EXTENSION_EDGES`).
6. Add a `configs/stages/<name>.yaml` block if it takes settings.
7. Document it in `docs/pipeline.md` or `docs/extensions.md`.

The orchestrator derives order from the graph; do not encode order in the stage.

A `StageSpec` must answer: **why does this exist (ROI), what evidence justifies
it, what does it cost, and what would kill it?** If you cannot answer, the step
probably belongs behind a gate.

## Adding a prompt

1. Add `packages/frictiontune/src/frictiontune/llm/prompts/library/<name>.md`.
   Front matter declares `name`, `version`, `role`, `schema`, and `description`.
   The body has a `:::system` section (policy) and a `:::user` section (the
   dynamic `{{ variable }}` inputs).
2. Call it with `gateway.complete_structured(role, config, "<name>", values,
   Schema)`.
3. Put the response model in `domain/responses.py` and name it in the prompt's
   `schema:` field.

Write the system section as **policy only**: what to judge, the rubric, the data
boundary, and a closing "reply with a single JSON object and nothing else". Do
**not** hand-write the output JSON: `complete_structured` derives the contract
from the pydantic schema and sends it as an extra system message, so field names,
types, and enums have a single source of truth and cannot drift from the code.
Bump `version` whenever the content changes.

Boilerplate shared by several prompts lives once in
`llm/prompts/library/_partials/*.md` and is pulled in with `{{> partial-name }}`
(expanded at parse time — a missing or cyclic partial is a hard error). The two
current partials are `untrusted-data` (the data-boundary sentence) and
`json-only` (the closing line); use them instead of repeating the text.

Every placeholder must appear in the `values` dict; a missing one is a hard
render error, never an empty string. `tests/test_prompts.py` checks that the
library parses, renders, that includes expand, and that every declared `schema`
resolves; `tests/test_prompt_contracts.py` statically checks that each
`complete_structured` call site passes exactly its prompt's variables.

## Adding a backend (in `frictiontune-train`)

Implement one of the protocols in `frictiontune.backends.protocols`
(`TrainingBackend`, `ServingBackend`, `RolloutBackend`) and register it in
`frictiontune_train/plugin.py:register`. Import the heavy stack **inside** the
method, not at module scope, so `import frictiontune_train` stays light. Use
`frictiontune_train.lazy.require` for actionable errors.

## Containers and operations

The pipeline runs as containers built from one cascading `container/Containerfile`
(one image per pipeline: `data`, `train`, `serve`, `llm`, `dev`). The image owns
the toolchain; the repository is bind-mounted at `/work` and its `packages/*/src`
shadow the baked install, so code stays live and `code_revision()` sees the same
tree a host run would — the same `run_id`, the same artifact cache. See
[`container/README.md`](../container/README.md) for the build args, the pod
layout, and the local llama.cpp recipe.

Every run is registered and analysed:

```bash
python3 scripts/ops/log_registry.py backfill      # register journal/partial/*.txt
python3 scripts/ops/log_registry.py show --pipeline v10
python3 scripts/ops/analyze.py --root journal/runs # anomalies.json, ANALYSIS.md, gaps.md
```

`scripts/ops/ft_pod.sh` runs the pipeline in a podman *pod* with the replay
servers and the monitor sidecar. Because the GPU workloads do not fit on the
machine at once, it runs in **phases**, each handing the hardware to that phase's
bottleneck:

| command | stages | GPU layout |
| --- | --- | --- |
| `ft_pod.sh data` | `decontaminate`, `verify_rules` (every LLM stage) | local llama.cpp on **all** GPUs |
| `ft_pod.sh train` | `train_sft` | **DDP across all cards**, servers evicted |
| `ft_pod.sh replay` | `evaluate_replay`, `export` | base, adapter, and (local) judge on separate cards |
| `ft_pod.sh all` | all three, in order | — |

All phases pass identical run flags, so they resolve the same `run_id` and later
phases skip earlier phases' work. With `FT_LOCAL_LLM=1` the data phase serves the
`analyst`/`judge`/`teacher`/`compactor` roles from a local llama.cpp server and the
`local` environment overlay raises role concurrency to match. By default that is a
single process layer-split across every card (`CTX` is the total context, divided
across `--parallel` slots). Set `FT_LLM_SERVERS=N` to run `N` data-parallel
replicas instead — one model per card, each on its own port, with the context and
slot budget split evenly so the per-slot window is unchanged. The client
round-robins across the replica list; no proxy is involved. Use replicas when the
model fits a card (an 8B at Q4 does), layer-split when it does not. See
[`container/README.md`](../container/README.md) for the full recipe.

## Hardware generality

The project is designed to run on **any** GPU box, not just the one it was
developed on. Three layers cooperate, and all three are driven by the hardware
preset (`configs/hardware/<name>.yaml`, selected with `--hardware`):

* **Vendor.** The container cascade builds for AMD (ROCm) or NVIDIA (CUDA) by
  swapping three build args (`make -C container all VARIANT=cuda`). At run time
  `ft_pod.sh` detects the accelerator and passes the right podman device grants
  and per-card pin env. Force it with `FT_ACCELERATOR=amd|nvidia`.
* **Card count.** `hardware.devices` is the single source of the DDP topology:
  `devices: 3` → a 3-rank `torch.distributed.run`; `devices: 1` → a single pinned
  process. `"auto"` resolves from the *visible* devices. Never hardcode a count.
* **Budget.** `hardware.per_device_batch`, `hardware.grad_accum` (per card under
  DDP) and `hardware.max_seq_len` size both training and llama.cpp; the preset is
  the only place the budget is declared. The derived `--tensor-split` in
  `ft_pod.sh` is computed from the detected card count, not written by hand.

Add a preset to support a new box; do not add conditionals to stage code. The
`cloud-h100x4`, `mi300x-192g`, `3x24g`, `2x24g` and `1x24g` presets are the
worked examples, and
[`configs/hardware/README.md`](../configs/hardware/README.md) is the authoring
guide.

## The rules of the codebase

These are not style preferences; they keep the project coherent.

1. **No semantic judgment in deterministic code.** If a decision is about
   meaning, route it through the gateway. Cosine similarity may *retrieve*; it
   may not *decide*.
2. **One retry primitive.** `core.retry.with_retry` only.
3. **One LLM funnel.** Stages never construct a client.
4. **Strict models at every boundary.** Agent responses, config, and backend
   contracts are pydantic models with `extra="forbid"`.
5. **Atomic, explorable stages.** Write named artifacts; never pass objects.
6. **No machine specifics in the repo.** Everything host- or model-specific is
   an environment variable.
7. **No GPU assumptions.** Read the hardware config; never count devices.

## A note on running during development

The pipeline is designed to be run; the *extensions* and training need real
endpoints and GPUs. For unit-level work you rarely need either: the plumbing is
pure and tested, and the gateway is only entered by stages that declare
`requires_llm = True`.