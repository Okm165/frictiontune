# AGENTS.md

Guidance for AI coding agents working in the `frictiontune` repository.

## What this project is

A terminal-first, open-source pipeline that mines developer↔agent chat logs,
detects **friction**, compiles runtime rules, and builds curated training data for
QLoRA fine-tuning. Two uv workspace members:

* `packages/frictiontune` — the light data pipeline and CLI.
* `packages/frictiontune-train` — the optional GPU/training side.

## Non-negotiable principles

Read these before changing anything. They are enforced by construction, not by
convention.

1. **All semantic judgment is made by LLMs/agents.** Deterministic code does
   plumbing only: IO, parsing, masking, hashing, validation, retrying. Never add
   a regex, keyword list, cosine threshold, or heuristic that decides *meaning*.
   Embeddings may *retrieve*; an agent must *decide*.
2. **One retry primitive.** `frictiontune.core.retry.with_retry`. Do not write
   another retry loop.
3. **One LLM funnel.** All model access goes through `LlmGateway`
   (`frictiontune.llm.gateway`). Stages never build a client. Structured calls
   return strict pydantic models.
4. **Strict models at every boundary.** Agent responses, config, and backend
   contracts are pydantic models with `extra="forbid"`. Validate and repair
   malformed LLM output into typed objects; never pass raw dicts around.
5. **Atomic, explorable stages.** A stage reads named artifacts from the
   `ArtifactCatalog` and writes named artifacts via `ctx.emit_*`. Never pass
   Python objects between stages.
6. **No machine specifics in the repo.** Host paths, model ids, and endpoints are
   environment variables (`${VAR:-default}`). Never hardcode a GPU count; read
   `ctx.config.hardware`.
7. **No deferrals.** Every stage and component is implemented end to end.
   Low-ROI steps are *gated* (`Lead.EXTENSION`, `--allow-gated`), not stubbed.
8. **ROI is a budget.** Every `StageSpec` states ROI, justification, evidence,
   cost, and a kill criterion.

## Repository map

See [`docs/development.md`](docs/development.md) for the full layout. Key files:

* `pipeline/spec.py` — `StageSpec` (the self-describing contract).
* `pipeline/stage.py` — the `Stage` base class.
* `pipeline/context.py` — `RunContext`, `StageContext`, `ArtifactCatalog`.
* `pipeline/orchestrator.py` — idempotent execution by `run_id`.
* `pipeline/dag.py` — `CORE_EDGES`, `EXTENSION_EDGES`.
* `llm/gateway.py` — the single LLM funnel.
* `llm/prompts/library/*.md` — prompts, addressed by name.
* `domain/` — strict domain and response models.
* `stages/` — core stages plus `_extensions.py`.

## Commands

```bash
uv sync --all-packages
uv run ruff check packages tests
uv run ruff format packages tests
uv run mypy packages/frictiontune/src/frictiontune packages/frictiontune-train/src/frictiontune_train
uv run pytest
```

After any change, run all four. The bar is: ruff clean, `ruff format --check`
clean, mypy strict clean, tests pass.

## Rules for editing

* **Match the surrounding style.** This codebase documents every public
  function, uses `from __future__ import annotations`, and keeps modules narrowly
  scoped.
* **Typing-only imports:** because pydantic evaluates field annotations at
  runtime, do not move a field's type into `TYPE_CHECKING` unless the root ruff
  config's `runtime-evaluated-base-classes` covers it. When in doubt, leave the
  import at module scope.
* **Adding a stage:** implement the class, register it in `stages/__init__.py`,
  add graph edges in `pipeline/dag.py`, add a `configs/stages/<name>.yaml` block
  if needed, and document it.
* **Adding a prompt:** create `llm/prompts/library/<name>.md`, call it via
  `complete_structured`, and add its schema to `domain/responses.py`.
* **Adding a backend:** implement the protocol in `frictiontune-train` and
  register it in `plugin.py`. Import the heavy stack inside methods, never at
  module scope.

## Things not to do

* Do not add a deterministic heuristic to make a semantic call.
* Do not construct an `openai`/`httpx` client outside `LlmClient`.
* Do not write a second retry or JSON-repair helper.
* Do not hardcode paths, model names, GPU counts, or endpoints.
* Do not make a core stage depend on a GPU package.
* Do not delete or rewrite artifacts a run produced; the artifact tree is the
  evidence trail.

## Testing expectations

Unit tests cover deterministic plumbing and must pass without a network or GPU.
Mark tests that need an LLM endpoint with `@pytest.mark.network` and tests that
need a GPU with `@pytest.mark.gpu`; both are excluded by default.