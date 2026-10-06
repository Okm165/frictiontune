# Architecture

This document explains how `frictiontune` is put together and why. If you only
want to run it, see [`quickstart.md`](quickstart.md).

## The one idea

**Semantic judgment is made by LLMs or agents. Deterministic code only does
plumbing.** Every place where the pipeline decides *what a piece of text means* —
is this a friction event, does this completion teach the right lesson, are these
two examples the same, is a rule good — is an LLM/agent call returning a strict
schema. The deterministic code reads files, parses JSON, masks tokens, hashes
content, validates schemas, and retries. No regex decides whether an agent
misbehaved; no cosine threshold decides whether two examples are duplicates.

This is enforced by construction: deterministic primitives live in
`frictiontune.core` and never make a judgment, and every judgment flows through
one gateway (`frictiontune.llm.gateway.LlmGateway`).

## Two packages, one workspace

The repository is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/)
with a virtual root and two members:

| Package | Role | Installs on |
| --- | --- | --- |
| `packages/frictiontune` (`frictiontune`) | The light data pipeline and CLI | A CPU log box |
| `packages/frictiontune-train` (`frictiontune-train`) | Torch/peft/trl training, export, rollout | A GPU box |

The data pipeline depends on the training side **only through protocols**
(`frictiontune.backends.protocols`). The heavy package registers concrete
implementations through the `frictiontune.plugins` entry-point group, so the
light package never imports torch.

```
Application.build(config)
  └─ StageRegistry (core + extensions)      ← packages/frictiontune
  └─ BackendRegistry                         ← protocols
  └─ LlmGateway (one funnel for all LLM IO)
  └─ load_plugins(...)                       ← entry point "frictiontune.plugins"
                                               packages/frictiontune-train supplies
                                               training/serving/rollout backends
```

## The stage contract

Every stage is an atomic step subclassing `frictiontune.pipeline.stage.Stage`.
A stage:

1. declares a **`StageSpec`** — a self-describing ROI contract (tier, lead, ROI,
   justification, evidence, cost, inputs, outputs, kill criterion, whether it
   needs an LLM, and any gate);
2. declares a **`settings_model`** — a pydantic model for its config block;
3. implements **`async def run(ctx: StageContext) -> StageResult`**.

```python
class Stage(ABC):
    spec: ClassVar[StageSpec]
    settings_model: ClassVar[type[StageSettings]] = StageSettings

    async def run(self, ctx: StageContext) -> StageResult: ...
```

`StageSpec` is not decoration. `frictiontune stages` renders it, and the
orchestrator uses `tier` and the gate to decide what may run. A stage that cannot
justify itself does not belong in the core.

## Data flow: the artifact catalog

Stages do not pass Python objects to each other. Each stage reads named inputs
from a shared `ArtifactCatalog` and writes named outputs through
`StageContext.emit_*`. This is what makes stages independently runnable and the
whole pipeline explorable: every intermediate is on disk.

| Helper | Writes | Typical use |
| --- | --- | --- |
| `ctx.emit_jsonl(name, ...)` | JSONL | Nested, heterogeneous records (messages, events, examples) |
| `ctx.emit_json(name, obj)` | JSON | One structured object (a report, a probe) |
| `ctx.emit_text(name, text)` | Markdown | Human-readable report |
| `ctx.jsonl_sink` | streaming | Never hold the corpus in memory |

`emit_*` writes the file, **publishes** it to the catalog under its logical name,
and records it on the result. Sinks stream to disk so a multi-gigabyte corpus is
never resident.

A `DatasetRef` carries the logical `name`, the `path`, the `kind`, row/byte
counts, and a `sha256`. The store is suffix-aware: logical names are extensionless
(`messages`), and the on-disk file gets the right suffix (`messages.jsonl`,
`rules.json`). Because the ref records its `kind`, names round-trip correctly
through manifests.

## Idempotency and run ids

A stage invocation is identified by a **run id**: a hash of

```
(stage name, stage config block, code revision, hashes of input artifacts)
```

Before running a stage the orchestrator looks for a manifest with that run id. If
one exists and `--force` is not set, the stage is skipped and its recorded
outputs are re-published into the catalog. Re-running a pipeline therefore does
the minimum work and produces identical artifacts. Change the config, the code,
or an upstream input and the run id changes, so the stage re-runs.

The graph is declared in `frictiontune.pipeline.dag`:

* `CORE_EDGES` — the 15-stage default data path.
* `EXTENSION_EDGES` — edges that attach the Tier C extensions.
* `core_graph()` / `full_graph()` — `DependencyGraph` instances over them.

The orchestrator re-derives execution order from the graph; the order in the
source files is documentation, not control flow.

## One LLM funnel

All model access goes through a single OpenAI-compatible client:

```
LlmClient          # transport: async, cached by content hash, transport-retried
  → LlmGateway     # schema funnel: render prompt, complete, validate, repair
      → stages     # never construct their own client
```

* **Prompts** live as markdown under `frictiontune/llm/prompts/library/` and are
  addressed by name. A prompt carries *policy* — what to judge, the rubric, and
  the data boundary — plus `{{ placeholder }}` inputs. It never hand-writes the
  output JSON.
* **Structured calls** use `gateway.complete_structured(role, config, prompt_name,
  values, schema)` and return a validated pydantic instance. The gateway derives
  the output contract from `schema` (its JSON Schema), inlines nested `$ref`s so
  the contract is self-contained for models that ignore `$defs`, keeps the
  declaration order (reasoning fields first), and sends it as an extra system
  message — so the model, the prompt, and the code share one source of truth.
  Validation is **strict**: a stringified number or boolean is a malformed reply,
  not something to coerce. Only two JSON realities are allowed through — an
  integer literal for a probability (`domain.types.Probability`) and an enum
  supplied by value (`domain.types.enum_by_value`). On malformed output the
  gateway asks the model to repair the specific problem, bounded by
  `max_repairs`; the repair loop is itself a `with_retry` policy.
* **Retry is defined once.** `frictiontune.core.retry.with_retry` is the single
  retry primitive, and `frictiontune.llm.transport.TRANSPORT_RETRYABLE` is the
  single set of retryable transport errors — OS-level failures plus the OpenAI
  SDK's connection/timeout, rate-limit, and 5xx errors, which are not `OSError`
  subclasses and would otherwise never be retried. The chat client and the
  embedding backend both use them, and the gateway reuses the primitive for
  schema repair; a test asserts `tenacity` is imported only by `core/retry.py`.
  There is no second, ad-hoc retry definition anywhere.

Roles (`frictiontune.config.models.RoleName`) are typed names — `router`,
`compactor`, `friction_analyst`, `judge`, `teacher`, `target`, `adapter`, … Each
maps to an `LlmRoleConfig` (endpoint, model, temperature, limits). `target` and
`adapter` are the two endpoints replay compares; everything else is the analyst
side.

## Configuration

Configuration is layered and contains no machine specifics; every host path or
model id is an environment variable reference (`${VAR:-default}`). The tree is
resolved by `frictiontune.config.loader.load_config` — see
[`configuration.md`](configuration.md). Empty environment expansions drop the key
so optional values keep their `None` defaults.

## ROI-first design

`StageSpec.tier` is a deliberate budget:

* **Tier A (`Lead.CORE`)** — runs by default. High ROI, justified by the plans.
* **Tier C (`Lead.EXTENSION`)** — runs only with `--allow-gated`. Each carries the
  evidence that would justify promoting it and the kill criterion that would
  retire it.

Extensions are not stubs; they are implemented end to end. They are gated because
the ROI case for running them *by default* is not yet made, not because they are
incomplete. See [`extensions.md`](extensions.md).

## Reporting

`frictiontune.reporting` renders the evidence a developer reads:

* `render_run_report(manifest)` — what a run did.
* `render_friction_report(profile)` — the friction taxonomy cut.
* `render_dashboard(dashboard)` — the multi-signal "friction drop".

The dashboard is the finish line. It compares the base model (`target`) and the
tuned model (`adapter`) over the same replayed episodes with a seeded bootstrap
interval per signal. Its headline signal is `friction_avoidance`
(higher-better); a single number is easy to game, so the dashboard reports
several independent signals and shows each delta with its interval.