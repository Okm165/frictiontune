# frictiontune-train

The GPU side of frictiontune: QLoRA SFT, GRPO, adapter export, and rollout
against an external agent runner. This package is the only place that imports
the ML stack; the data pipeline talks to it solely through the backend protocols.

This is a member of the `frictiontune` workspace. See the repository root
[`README.md`](../../README.md) and [`docs/`](../../docs).

## Install

```bash
uv sync --package frictiontune-train --extra train   # SFT
uv sync --package frictiontune-train --extra grpo    # RL
```

## Backends registered

* `LocalTrainingBackend` — `train` (QLoRA SFT) and `train_grpo`.
* `LocalServingBackend` — merge or package an adapter, with a parity probe.
* `HttpRolloutBackend` — send tasks to an external agent runner
  (`FRICTIONTUNE_ROLLOUT_URL`).

Unsloth is used when importable and `FRICTIONTUNE_USE_UNSLOTH != 0`; otherwise a
plain transformers + peft path runs.