# Pipeline reference

Every stage is atomic, independently runnable, and self-describing. This page is
the stage-by-stage contract: what each one consumes, what it produces, and the
mechanics that make it work.

Two conventions throughout:

* **Artifact names are logical and extensionless** (`friction_events`, `rules`).
  The store adds the suffix (`.jsonl`, `.json`, `.md`). Read them with
  `ctx.catalog.stream(name, Model)` or `ctx.catalog.read_json(name)`.
* **Deterministic vs judged.** Steps marked *plumbing* are pure code; steps
  marked *judged* are LLM/agent calls returning strict schemas.

The core run is 15 stages. `fetch_base` is a prerequisite utility that the others
depend on indirectly. Extensions are documented in [`extensions.md`](extensions.md).

```
ingest ─▶ normalize ─▶ segment ─▶ friction_detect ─┬─▶ mine ─▶ judge ─▶ build_sft ─▶ curate ─▶ decontaminate ─▶ train_sft ─┐
                                                    └─▶ compile_rules ─▶ verify_rules ─────────────────────────────────────┴─▶ evaluate_replay ─▶ export
```

---

## Shared shapes

Some stages operate on the same domain objects. Knowing them makes the rest
readable; full definitions live in `frictiontune.domain`.

| Object | Meaning |
| --- | --- |
| `Session`, `Message`, `Part` | The canonical trace. A `Part` is `reasoning`, `text`, `tool`, `compaction`, `system`, `file`, or `idle`. |
| `Episode` | Everything between two human messages; the unit of friction and evaluation. Carries descriptive `EpisodeFeatures`. |
| `Phase` | A plan/execute/verify sub-segment of a long episode. |
| `FrictionEvent` | A typed friction moment: taxonomy leaf + form/cause + verbatim `evidence` + `severity` + `confidence` + resolution. |
| `FrictionProfile` | Aggregate cut of all events (counts by leaf/form/cause/cost, resolution rates). |
| `TrainingCandidate` | A mined example before judging. |
| `TrainingExample` | A judged example with a `keep`/`reject` verdict and rationale. |
| `SFTRecord` | A rendered, tokenized, masked training record (`input_ids`, `labels`). |
| `Rule` / `CompiledRules` | A runtime rule: applicability, instruction, verifier, evidence. |
| `SFTRecord` `labels` | `-100` on context/environment; real token ids only on the trained completion. |

---

## Core stages

### `ingest`

**In:** the source SQLite DB (`FRICTIONTUNE_CORPUS_DB` or `--set ingest.db_path`).
**Out:** `messages` (JSONL), `sessions` (JSONL), `ingest_report` (JSON).
**Cost:** cpu · no API.

Pure plumbing. Opens the database **read-only and immutable** so a live agent
session is never disturbed, streams session-by-session, and emits canonical
`Message`/`Session` objects. Unparseable rows are skipped and counted rather than
crashing the run. Nothing is held in memory beyond one session.

*Why it is core:* nothing works without the trace, and the read-only/streaming
discipline is a correctness and privacy guarantee.

### `normalize`

**In:** `messages`.
**Out:** `messages` (rewritten), `tool_schemas` (JSONL), `redactions` (JSON),
`normalize_report` (JSON).
**Cost:** mixed · medium API.

Three transforms, each split into plumbing + judgment:

* **compaction** *(judged, gated by size)* — tool outputs beyond
  `compact_threshold_chars` are compressed by an agent that must keep error text
  verbatim. Head/tail truncation would cut errors out of the middle; the measured
  corpus has a p90 tool output of 13 KB with 33.7% above 4 KB.
* **schema inference** *(judged)* — tool JSON Schemas are inferred from observed
  calls by an agent, never by merging keys heuristically.
* **redaction** *(plumbing candidates + judged decisions)* — regex proposes
  candidates; an agent decides. Over-redaction is a real cost, so the model makes
  the call.

Streamed in batches with bounded concurrency into a JSONL sink.

### `segment`

**In:** `messages`.
**Out:** `episodes` (JSONL), `phases` (JSONL), `segment_report` (JSON).
**Cost:** mixed · medium API.

Splits sessions into user-bounded episodes *(plumbing)*. Episodes long enough to
matter are divided into coherent plan/execute/verify phases *(judged)*; short
episodes get a single phase with no LLM call. The measured corpus averages ~22
assistant turns per user turn (p90 59, max 102), so segmentation is what keeps a
single episode analysable.

### `friction_detect`

**In:** `episodes`.
**Out:** `friction_events` (JSONL), `friction_profile` (JSON).
**Cost:** mixed · high API.

The signal the whole pipeline is built around.

1. A cheap **router** *(judged)* decides whether an episode merits deep analysis.
2. The **friction analyst** *(judged)* reads the episode and returns typed
   `FrictionEvent`s with **verbatim evidence**. It runs as a small ensemble at a
   configured temperature; results are merged by vote for calibrated precision.
3. Events are validated against the taxonomy: a leaf must agree with the
   event's declared form and cause, or the event is dropped/repaired.

The aggregate is the `FrictionProfile` — the axis counts you read first.

### `compile_rules`

**In:** `friction_events`, `episodes`, `tool_schemas`.
**Out:** `rules` (JSON), `rules_report` (JSON).
**Cost:** mixed · medium API.

Turns observed friction into **atomic runtime rules**. A rule is only useful if it
can be checked at runtime, so each compiled rule carries the TRACE structure: an
**applicability** condition, an imperative **instruction**, and a **verifier** the
runtime can evaluate. Compilation is per episode *(judged)* so the rule keeps its
context; identical rules from different episodes are merged deterministically and
their evidence unioned *(plumbing)*.

### `verify_rules`

**In:** `rules`, `episodes`, `friction_events`.
**Out:** `rules_verified` (JSON), `verify_rules_report` (JSON).
**Cost:** mixed · medium API.

Agentic replay over a held-out sample: would each rule have fired? Would it have
**prevented** the targeted friction? Would it have **blocked correct behaviour**
(collateral)? Collateral is the failure mode that makes rules dangerous, so the
accept gate requires prevention and forbids collateral. Only verified rules ship.

### `mine`

**In:** `friction_events`, `episodes`, `tool_schemas`.
**Out:** `candidates` (JSONL), `mine_report` (JSON).
**Cost:** mixed · high API.

Recovery is the residual an SFT policy must learn: an imitator of success cannot
recover once it deviates. For each friction event the **recovery-finder**
*(judged)* locates the span where the assistant diagnosed and fixed the failure.
That span becomes the trained completion; everything before it is masked context.
Produces `TrainingCandidate`s.

### `judge`

**In:** `candidates`.
**Out:** `judged` (JSONL), `judge_report` (JSON).
**Cost:** mixed · high API.

The shared quality gate every candidate must pass — mined or synthetic. LIMA's
lesson is that unfiltered data actively hurts, so the gate is strict: a rubric of
five 0–5 axes *(judged)*, an explicit accept decision, and a deterministic
post-check that a model cannot over-accept a candidate it scored as
disqualifying on adherence or correctness *(plumbing)*.

### `build_sft`

**In:** `judged`.
**Out:** `sft` (JSONL), `token_stats` (JSON).
**Cost:** cpu · no API.

Renders judged examples into masked, model-ready records *(plumbing)*. Labels are
`-100` on everything that is context or environment output, so loss is computed
only over the assistant behaviour we want to teach. Whether **reasoning** tokens
are trained or masked is a first-class, configurable policy
(`include_reasoning_context`, `train_reasoning`). The tokenizer is the base
model's (HuggingFace) with a deterministic fallback so a CPU box is never blocked.

### `curate`

**In:** `judged`, `sft`.
**Out:** `curated` (JSONL), `curate_report` (JSON).
**Cost:** mixed · medium API.

Two deduplication passes: an exact byte-hash pass *(plumbing)* and a semantic
pass. Embeddings only **retrieve** candidate pairs; a **dedup agent** *(judged)*
decides whether two examples teach the same lesson. There is no shingle or cosine
threshold making the semantic call.

### `decontaminate`

**In:** `curated`, `judged`.
**Out:** `train` (JSONL), `eval` (JSONL), `contamination_report` (JSON).
**Cost:** mixed · medium API.

Makes the friction-drop number trustworthy. The split is a strict **time split**
(future work is evaluated on, past work is trained on), never random. Four tiers
then run over the boundary:

* **T1** exact n-gram overlap *(plumbing)*
* **T2** syntactic containment *(plumbing)*
* **T3** semantic leak *(embeddings retrieve, agent judges)*
* **T4** task-level leak *(agent judges)*

A "lucky pass" flag records cases the eval set only passes because of leakage.
Relaxing a tier is a deliberate, recorded decision, not a default.

### `train_sft`

**In:** `train`, `base_probe`.
**Out:** `training_result` (JSON), `adapter` (JSON).
**Cost:** gpu · no API.

Thin by design: validates the base probe, builds a typed `TrainingRequest`, and
delegates to the registered **training backend** behind one seam. The MoE trap is
handled upstream — `target_parameters` is empty by default so LoRA never attaches
to fused expert weights. If no backend is registered (no GPU package installed),
the stage fails with a clear message rather than importing torch.

### `evaluate_replay`

**In:** `eval`, `judged`, `adapter`.
**Out:** `dashboard` (JSON), `replay` (JSONL).
**Cost:** gpu/api · high API.

The finish line. Each held-out example is replayed through **both** the base model
(`target`) and the served adapter (`adapter`) from the **same** context, then
scored by the shared judge. A dashboard of independent signals is built with a
seeded bootstrap interval, so a small or noisy difference from the model change is
visible as such rather than sold as a win. Headline signal: `friction_avoidance`
(higher-better).

### `export`

**In:** `adapter`.
**Out:** `export_result` (JSON).
**Cost:** gpu/cpu · no API.

Delegates to the serving backend to merge or package the adapter, then records a
**parity** result so a silent export regression cannot pass unnoticed. A bad merge
produces a model that loads but answers differently; parity catches that.

### `fetch_base` *(prerequisite utility)*

**In:** —
**Out:** `base_probe` (JSON), `report` (markdown).
**Cost:** cpu · no API.

Locates the base weights, downloads them if `allow_download` is set, and records
architecture facts (hidden size, layers, vocab, context length, MoE detection,
quantisation, GGUF presence). `train_sft` uses the probe to reject impossible
jobs before any GPU time is spent. It is the reason a GGUF-only setup fails fast
with an explanation instead of hours later.

---

## How to run a subset

```bash
# one stage
uv run frictiontune run --stage friction_detect

# several, plus everything they depend on
uv run frictiontune run --stage evaluate_replay --include-deps

# include gated extensions
uv run frictiontune run --stage friction_cluster --allow-gated
```

Execution order is always re-derived from the graph; asking for a stage with
`--include-deps` pulls in its upstream closure automatically.