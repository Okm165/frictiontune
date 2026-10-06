# Configuration

`frictiontune` keeps every machine-, model-, and research-specific value in
configuration. The repository itself contains none. This page documents the tree,
the resolution order, and the environment variables.

## Resolution order

Configuration is loaded by `frictiontune.config.loader.load_config`. Later layers
win, shallow/deep-merging into one tree:

1. `configs/base.yaml` — defaults
2. `configs/roles.yaml` — LLM role bindings
3. `configs/friction_taxonomy.yaml` — friction axes and leaves
4. `configs/hardware/<preset>.yaml` — selected with `--hardware`
5. `configs/environments/<env>.yaml` — optional overlay, `--environment`
6. `configs/stages/*.yaml` — per-stage blocks, keyed by filename stem
7. explicit `--set a.b=c` overrides

```bash
uv run frictiontune validate --hardware 3x24g --environment example --set seed=7
```

The final tree is validated into a strict `RunConfig`. Unknown keys are
**rejected**, so a typo fails at load time, not mid-run.

## Environment expansion

Any string may reference an environment variable:

```yaml
base_url: ${FRICTIONTUNE_LLM_BASE_URL:-http://localhost:8000/v1}
```

* `${VAR}` — required; missing is an error.
* `${VAR:-default}` — optional with a fallback.

An expansion that produces an **empty string drops the key**, so a setting like
`corpus_db: ${FRICTIONTUNE_CORPUS_DB:-}` keeps its `None` default when the
variable is unset instead of collapsing to `.`.

## `RunConfig`

| Field | Type | Notes |
| --- | --- | --- |
| `project` | str | Free-form label. |
| `seed` | int | Drives sampling and bootstrap intervals. |
| `paths` | `PathsConfig` | `artifacts_dir`, `cache_dir`, `corpus_db`, `base_weights_dir`. |
| `hardware` | `HardwareConfig` | Training knobs; no GPU count is hard-coded. |
| `llm` | `dict[RoleName, LlmRoleConfig]` | One binding per role. |
| `embedding` | `EmbeddingConfig` | Retrieval/calibration/clustering embeddings. |
| `taxonomy` | `TaxonomyConfig` | Forms, causes, costs, resolutions, leaves. |
| `evaluation` | `EvaluationConfig` | Time split and T1–T4 toggles. |
| `stages` | `dict[str, dict]` | Per-stage settings blocks. |

`RunConfig.role(RoleName.X)` returns that role or raises a clear error if it is
not configured.

## Roles (`configs/roles.yaml`)

All roles speak the same **OpenAI-compatible** interface; only the endpoint,
model, temperature, and limits differ. Roles point at environment variables, so
the same tree runs against different providers.

| Role | Used by | Typical temperature |
| --- | --- | --- |
| `target` | `evaluate_replay` (base model) | 0.0 |
| `adapter` | `evaluate_replay` (tuned model) | 0.0 |
| `router` | `friction_detect` triage | 0.0 |
| `compactor` | `normalize` | 0.0 |
| `schema_inferer` | `normalize` | 0.0 |
| `friction_analyst` | `friction_detect` | 0.2 (enables the ensemble) |
| `recovery_finder` | `mine` | 0.0 |
| `attributor` | `attribute` extension | 0.0 |
| `rule_compiler` | `compile_rules` | 0.0 |
| `rule_verifier` | `verify_rules` | 0.0 |
| `judge` | `judge`, `decontaminate`, `evaluate_replay` | 0.0 |
| `dedup` | `curate` | 0.0 |
| `teacher` | `synthesize` extension | 0.7 |
| `user_simulator` | `evaluate_sim_user` extension | 0.7 |

An `LlmRoleConfig` has: `provider`, `base_url`, `model`, `api_key_env`,
`temperature`, `max_tokens`, `concurrency`, `cache`, `timeout_s`,
`response_format`, `prompt_ref`, and a `retry` policy. `cache: false` is used for
`target`/`adapter` because replay must not serve cached generations.

`response_format` is `text`, `json_object`, or `json_schema`. Structured calls
always carry the pydantic JSON Schema in the messages; setting a role to
`json_schema` additionally asks the endpoint for grammar-constrained decoding so
the reply cannot violate the schema. Endpoints that do not support it fall back
safely — leave roles on `json_object` unless you know the backend honours
`json_schema`.

## Embeddings

Embeddings are used only for **retrieval** — proposing candidate neighbours for an
agent to judge — never for a threshold decision. `EmbeddingConfig` mirrors the role
config: `provider`, `base_url`, `model`, `api_key_env`, `dimensions`, `cache`,
`timeout_s`.

## Hardware presets (`configs/hardware/`)

A preset is a named `HardwareConfig`. Select it with `--hardware`, or set fields
directly with `--set hardware.devices=2`.

| Preset | Devices | Quant | Offload | Context |
| --- | --- | --- | --- | --- |
| `1x24g` | 1 | nf4 | none | 8192 |
| `2x24g` | 2 | nf4 | none | 8192 |
| `3x24g` | 3 | nf4 | none | 8192 |
| `cpu-offload-128g` | 1 | nf4 | cpu | 4096 |
| `mi300x-192g` | 1 | nf4 | none | 16384 |
| `cloud-h100x4` | 4 | nf4 | none | 16384 |

Adding a preset is a matter of dropping a new YAML file in the directory. The
pipeline never inspects the machine to guess; it uses what you configure.

## Taxonomy (`configs/friction_taxonomy.yaml`)

The SWE-chat axes plus our leaf labels:

* **forms** — `project-reading`, `intent-interpretation`, `rule-following`,
  `action-bounding`, `implement/execute`, `reporting`
* **causes** — `context`, `capability`, `instruction-ambiguity`, `tool/env`, `spec`
* **costs** — `effort`, `trust`, `irreversible`
* **resolutions** — `user-correction`, `self-recovery`, `none`
* **leaves** — each leaf maps to exactly one `[form, cause]` pair, e.g.
  `instruction_misread: [intent-interpretation, instruction-ambiguity]`.

`friction_detect` rejects (or repairs) any event whose leaf disagrees with its
declared form/cause, so the taxonomy is self-consistent by construction.

## Evaluation policy

`EvaluationConfig` controls the trustworthiness of the headline number:

* `time_split_quantile` — where the strict time split falls (default `0.8`).
* `enforce_time_split` — must be true for a trustworthy replay.
* `t1_exact_ngram` … `t4_task_level` — which contamination tiers run.
* `ngram_size` — T1 window size.
* `seeds` — bootstrap seeds for the dashboard intervals.
* `lucky_pass_flag` — record eval items that only pass due to leakage.
* `external_corpora` — optional calibration corpora for `evaluate_external`.

## Per-stage settings (`configs/stages/`)

Each file's stem is a stage name and its body is that stage's settings block:

```yaml
# configs/stages/friction_detect.yaml
router_enabled: true
ensemble: 2
batch_size: 32
max_transcript_chars: 100000
taxonomy_enforcement: true
```

A stage reads its block with `ctx.settings(SettingsModel)`. Because settings
models are strict, an unknown key in a `stages/*.yaml` file is an error. Remove a
file to fall back entirely to the code defaults.

## Environment variables

The fastest way to see the full contract is `.env.example`. The important groups:

| Variable | Purpose |
| --- | --- |
| `FRICTIONTUNE_CORPUS_DB` | SQLite session database to mine. |
| `FRICTIONTUNE_BASE_WEIGHTS` / `_REPO` / `FRICTIONTUNE_TOKENIZER` | Base model location. |
| `FRICTIONTUNE_LLM_BASE_URL` / `_API_KEY` / `FRICTIONTUNE_MODEL` | Analyst/judge endpoint. |
| `FRICTIONTUNE_TARGET_BASE_URL` / `_MODEL` / `_API_KEY` | Base model endpoint for replay. |
| `FRICTIONTUNE_ADAPTER_BASE_URL` / `_MODEL` | Tuned model endpoint for replay. |
| `FRICTIONTUNE_EMBED_BASE_URL` / `_MODEL` / `_API_KEY` | Embeddings. |
| `FRICTIONTUNE_TEACHER_MODEL`, `FRICTIONTUNE_USER_SIM_MODEL` | Generation roles. |
| `FRICTIONTUNE_ROLLOUT_URL` / `_TOKEN` | External agent runner for rollouts. |
| `FRICTIONTUNE_GRPO_JUDGE_URL` | Optional remote judge reward term. |

`--set key=value` is resolved with `parse_set_overrides` and can override anything,
including nested keys (`hardware.devices=2`, `evaluation.time_split_quantile=0.7`).