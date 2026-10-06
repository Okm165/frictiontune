# Quickstart

This guide takes you from a fresh clone to your first artifacts. It assumes no
machine-learning background, but it does assume a terminal and
[`uv`](https://docs.astral.sh/uv/).

## What frictiontune does

You have chat logs between developers and a coding agent. In those logs the agent
sometimes gets things wrong: it misreads an instruction, edits too much, forgets a
constraint, repeats a failing command, or fails to verify. Every time that
happens, a person has to step in and correct it. We call that **friction**.

`frictiontune`:

1. reads those logs,
2. finds the friction moments using LLM/agent evaluators,
3. turns them into **runtime rules** you can drop into an agent today, and
4. builds **curated training data** to fine-tune an open model so it needs less
   steering in the future.

It then **replays** the same situations against the original model and the tuned
model and reports the difference (the "friction drop").

## 0. Install

```bash
git clone https://github.com/frictiontune/frictiontune
cd frictiontune
uv sync --all-packages
```

`uv sync` creates `.venv` and installs the light data pipeline. It does **not**
install torch. You only need the GPU packages when you actually train:

```bash
uv sync --package frictiontune-train --extra train
```

## 1. Configure

The repo contains no machine-specific values. Copy the example environment and
fill in only what you need:

```bash
cp .env.example .env
```

The two things you almost certainly need to set:

* **`FRICTIONTUNE_CORPUS_DB`** — path to the SQLite session database to mine.
* **`FRICTIONTUNE_LLM_BASE_URL`** and **`FRICTIONTUNE_MODEL`** — an
  OpenAI-compatible endpoint for the analyst/judge roles.

Anything you leave unset keeps its default. Then validate:

```bash
uv run frictiontune validate --config-dir configs
```

You should see a summary table and `configuration is valid`.

## 2. Look at the pipeline

Before running anything, see what the pipeline is made of. Each stage explains
what it is for, what it costs, and what evidence justifies it:

```bash
uv run frictiontune stages
```

Tier `A` stages are the default core. Tier `C` stages are extensions that only
run with `--allow-gated`.

## 3. Dry-run

Ask the orchestrator what it would do, without doing it:

```bash
uv run frictiontune run --hardware 3x24g --dry-run
```

`--hardware` selects a preset from `configs/hardware/`. The presets only affect
training and describe VRAM, quantisation, batch sizes, and context length. Nothing
about your GPU count is hardcoded.

## 4. Run the core pipeline

```bash
uv run frictiontune run --hardware 3x24g
```

Progress is logged in the terminal and, more importantly, written to disk. Every
stage's output is kept under `artifacts/<pipeline>/<stage>/<run_id>/`.

Running the same command again is cheap: a stage whose inputs, config, and code
have not changed is skipped and its recorded outputs are reused.

## 5. Explore the artifacts

```bash
uv run frictiontune inspect            # list stage runs and their files
uv run frictiontune inspect --stage friction_detect
uv run frictiontune report             # markdown summary of the run
```

Open any JSONL or Parquet file to see what a stage produced. The whole point is
that there is no black box: the friction events, the rules, the judged examples,
and the contamination report are all plain files.

## 6. Read the friction drop

After `train_sft` and `evaluate_replay` have run:

```bash
uv run frictiontune dashboard
```

This renders the multi-signal replay dashboard. The headline is
`friction_avoidance` (higher is better); each signal shows the base value, the
tuned value, and a bootstrap interval around the change. If the interval spans
zero, treat the change as noise.

## 7. Optional: the terminal UI

```bash
uv sync --package frictiontune --extra tui
uv run frictiontune tui
```

The TUI is a read-only viewer over the stage registry and the artifact tree.

## Where to go next

* [`architecture.md`](architecture.md) — how the pieces fit together.
* [`pipeline.md`](pipeline.md) — every stage, input by input.
* [`configuration.md`](configuration.md) — the whole config tree and env vars.
* [`extensions.md`](extensions.md) — the gated Tier C stages.
* [`development.md`](development.md) — hacking on frictiontune itself.

## A note on training data

QLoRA training needs a **Hugging Face bf16 checkpoint**. A GGUF file (the kind
served by `llama.cpp`) cannot be QLoRA-trained. `fetch_base` probes this and
`train_sft` refuses impossible jobs rather than burning GPU hours, but the
practical consequence is worth stating up front: choose a bf16 base for training
even if the model you evaluate against is served as GGUF.