# frictiontune

[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Mine real agent chat logs, detect friction, compile runtime rules, and
fine-tune an agent that needs less steering.**

`frictiontune` is a terminal-first, open-source data pipeline. It reads a corpus
of real developer↔agent conversations, uses LLM/agent evaluators to find the
moments where a person had to correct or re-steer the agent, and turns those
moments into two outputs:

1. **Runtime rules** — a compiled system prompt and JSON policy an agent can
   load today, no training required.
2. **Curated training data** — masked SFT records (plus optional synthetic and
   RL data) used to QLoRA-fine-tune an open model.

The finish line is a **multi-signal replay dashboard**: the tuned model is
replayed against the same episodes as the base model and the "friction drop" is
reported with a bootstrap interval.

> New to the project? Read [`docs/quickstart.md`](docs/quickstart.md) first. It
> assumes no machine-learning background.

## Why

An agent that needs constant steering is expensive in a way that does not show
up in a benchmark score. People re-explain, undo, and re-run. That is *friction*.
It is measurable from the logs, and it is exactly what fine-tuning should reduce.

`frictiontune` is **ROI-first**: every stage declares why it exists, what
evidence justifies it, what it costs, and the kill criterion that would retire
it. Low-ROI steps are demoted to evidence-gated extensions rather than deleted.
All semantic judgment is made by LLMs or agents; deterministic code only does
plumbing (IO, parsing, masking, hashing, validation).

## Architecture at a glance

```
corpus ─▶ ingest ─▶ normalize ─▶ segment ─▶ friction_detect ─┬─▶ mine ─▶ judge ─▶ build_sft ─▶ curate ─▶ decontaminate ─▶ train_sft ─┐
                                                              └─▶ compile_rules ─▶ verify_rules ─────────────────────────────────┴─▶ evaluate_replay ─▶ export
```

* **Core pipeline** (15 stages) is the default. Each stage is atomic,
  independently runnable, and writes explorable artifacts.
* **Extensions** (Tier C: `reduce`, `attribute`, `friction_cluster`,
  `synthesize`, `raft_srft`, `rollout`, `train_grpo`, `evaluate_sim_user`,
  `evaluate_external`) only run behind `--allow-gated`.
* Data flows through a shared **artifact catalog**; a stage's run id is a hash
  of *(stage, config, code revision, input hashes)*, so re-runs are idempotent.

See [`docs/architecture.md`](docs/architecture.md) for the full design and
[`docs/pipeline.md`](docs/pipeline.md) for the stage-by-stage contract.

## Install

Requires [uv](https://docs.astral.sh/uv/). The repository is a uv workspace:

```bash
git clone https://github.com/frictiontune/frictiontune
cd frictiontune
uv sync --all-packages          # light data pipeline (installs on a CPU box)
```

The GPU side is optional and separate:

```bash
uv sync --package frictiontune-train --extra train   # torch, transformers, peft, trl
```

## Configure

Nothing machine-specific lives in the repo; it is all environment variables:

```bash
cp .env.example .env
# edit .env, then validate:
uv run frictiontune validate --config-dir configs
```

## Run

```bash
# See every stage and its ROI contract.
uv run frictiontune stages

# Dry-run the core pipeline.
uv run frictiontune run --hardware 3x24g --dry-run

# Run it.
uv run frictiontune run --hardware 3x24g

# Explore artifacts of the last run.
uv run frictiontune inspect
uv run frictiontune report
uv run frictiontune dashboard

# Measure the friction analyst against the golden episodes (calls an LLM).
uv run frictiontune eval-prompts --hardware 3x24g
```

## Documentation

| Document | What it covers |
| --- | --- |
| [`docs/quickstart.md`](docs/quickstart.md) | Zero-to-first-run for newcomers |
| [`docs/architecture.md`](docs/architecture.md) | Stage contract, catalog, idempotency |
| [`docs/pipeline.md`](docs/pipeline.md) | Every stage: inputs, outputs, mechanics |
| [`docs/configuration.md`](docs/configuration.md) | The layered config tree and env vars |
| [`docs/extensions.md`](docs/extensions.md) | The gated Tier C stages |
| [`docs/development.md`](docs/development.md) | Layout, lint, typing, tests |

## Status

An early-stage proof of concept. The data pipeline is complete and runnable; the
training backend is real but requires a GPU box and a bf16 base checkpoint. GGUF
weights cannot be QLoRA-trained — provide a Hugging Face checkpoint.

## License

Apache-2.0. See [`LICENSE`](LICENSE).