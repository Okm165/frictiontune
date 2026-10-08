# Extensions (Tier C)

The core pipeline is deliberately small. Everything else lives in
`frictiontune.stages._extensions` as an **evidence-gated extension**: real,
runnable, and complete, but not run by default because the ROI case for running
it *on every run* is not yet made.

A Tier C stage never runs unless you pass `--allow-gated`. The gate is enforced by
the orchestrator, not by the registry, so the stage is always importable and
inspectable:

```bash
uv run frictiontune stages --lead extension
uv run frictiontune run --stage reduce --allow-gated --include-deps
```

Each extension declares the evidence that would justify promoting it and the kill
criterion that would retire it. Those live on the spec and are printed by
`frictiontune stages`.

---

## `reduce` — map-reduce compression

**In:** `episodes` → **Out:** `reduced_spans` (JSONL), `reduce_report` (JSON).
Requires LLM.

Long episodes (up to ~102 assistant turns in the measured corpus) can exceed the
analyst window. `reduce` chunks them and asks the compactor agent for a faithful
summary that **preserves failures**, producing `ReducedSpan`s the analyst can
consume. It is a genuine map-reduce: each chunk is summarised independently.

*Gate:* only needed when episodes exceed the analysis context window.
*Kill:* drop if `normalize` compaction already fits every episode in one window.

## `attribute` — root-cause attribution

**In:** `friction_events`, `episodes` → **Out:** `attributions` (JSONL),
`attribute_report` (JSON). Requires LLM.

For each friction event, an agent localises the **earliest decisive step** — the
turn after which the failure became inevitable — and returns an `Attribution`.
This can sharpen masking and rule scoping.

*Gate:* masking already works from recovery spans, so attribution is optional.
*Kill:* drop if decisive turns never change masking or rule scope in practice.

## `friction_cluster` — the preference profile

**In:** `friction_events` → **Out:** `preference_profile` (JSON),
`friction_cluster_report` (JSON). Requires LLM.

Groups a long tail of events into a handful of **behaviour dimensions** with
rationales — the Preference Profile that `synthesize` targets.

*Gate:* only useful once the event tail is long.
*Kill:* drop if clusters add nothing over the taxonomy counts in the profile.

## `synthesize` — synthetic coverage

**In:** `preference_profile`, `candidates` → **Out:** `synthetic` (JSONL),
`synthesize_report` (JSON). Requires LLM.

Generates friction-avoiding training examples per behaviour dimension, seeded by
real mined recoveries (the `candidates` that `mine` extracted). The `synthetic`
examples go through the same `judge` gate as mined ones — `judge` consumes
`synthetic` as an optional input — so nothing generated reaches training without
passing the gate.

*Gate:* mined recoveries may already cover the needed behaviours.
*Kill:* drop if synthetic examples fail the judge at a higher rate than mined ones.

## `raft_srft` — success track

**In:** `episodes`, `friction_events` → **Out:** `sft_raft` (JSONL),
`raft_report` (JSON). No API.

Builds success-track examples from **clean episodes** (those without friction
events), following RAFT/SRFT. Purely deterministic selection over the trace.
Success examples give the model a positive track alongside recovered failures and
are folded into the same `judge` gate as an optional input.

*Gate:* recovery data may be sufficient on its own.
*Kill:* drop if success-track data shifts the model away from recovery behaviour.

## `rollout` — executable sandbox

**In:** `adapter` → **Out:** `rollouts` (JSONL), `rollout_report` (JSON).
No LLM; uses the **rollout backend**.

Frictiontune does **not** reimplement an agent loop. `rollout` sends each
configured task to an external agent runner over HTTP
(`FRICTIONTUNE_ROLLOUT_URL`, optional bearer token) and records the transcript.
This is how the project reuses existing OSS agents (OpenCode, Claude Code, an ACP
bridge) instead of maintaining its own.

*Gate:* requires a maintained execution sandbox.
*Kill:* drop the sandbox if judge-scored replay already predicts friction reduction.

## `train_grpo` — reinforcement learning

**In:** `train`, `judged`, `adapter`, `base_probe` → **Out:** `grpo_result` (JSON),
`train_grpo_report` (JSON). No LLM; uses the training backend.

Runs GRPO after SFT has plateaued, with a **composed reward** rather than a single
proxy. The reward is defined in `frictiontune_train.reward`:

| Term | Kind | What it does |
| --- | --- | --- |
| `judge` | judged (optional) | Remote judge score via `FRICTIONTUNE_GRPO_JUDGE_URL`. |
| `format` | plumbing | Rejects empty / runaway / marker-imbalanced completions. |
| `tool` | plumbing | Fenced JSON tool calls must parse. |
| `anti_hack` | plumbing | Penalises prompt-copying and low self-repetition. |

Weights are configurable per run.

*Gate:* SFT must plateau first, and the reward must be calibrated.
*Kill:* drop if reward hacking appears or SFT already saturates the signals.

## `evaluate_sim_user` — user in the loop

**In:** `adapter` → **Out:** `sim_user` (JSON), `evaluate_sim_user_report` (JSON).
Requires LLM.

A simulated user (the `user_simulator` role, capped at `max_turns`) drives each
goal against both the base and tuned models. The signal is **user turns per
episode** — fewer turns means the agent needed less steering. Higher fidelity
than replay, and noisier at small scale.

*Gate:* the replay dashboard is cheaper and already informative.
*Kill:* drop if replay already ranks the two models the same way.

## `evaluate_external` — calibration

**In:** `adapter` → **Out:** `external` (JSON), `evaluate_external_report` (JSON).
Requires LLM.

Runs tasks from `evaluation.external_corpora` through both models and judges the
outputs, guarding against overfitting to the one mined corpus.

*Gate:* internal replay is the primary evaluation.
*Kill:* drop if the internal replay and sim-user signals already generalise.

---

## Why gate at all?

Because ROI is a budget. A stage that costs GPU hours or a large API bill must
earn its place on a *default* run. Gating makes that trade explicit and
reversible: the code is written, tested, and ready; turning it on is one flag,
and turning it off restores the cheap default. Nothing here is a stub.