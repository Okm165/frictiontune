# frictiontune

The light data pipeline: mine agent chat logs, detect friction, compile runtime
rules, and build curated SFT data. Installs on a CPU box with no ML stack.

This is a member of the `frictiontune` workspace. See the repository root
[`README.md`](../../README.md) and [`docs/`](../../docs) for the full picture.

The public entry point is the `frictiontune` CLI:

```bash
uv run frictiontune stages
uv run frictiontune validate
uv run frictiontune run --hardware 3x24g
```

Heavy training lives in the sibling package `frictiontune-train`, which plugs in
through the `frictiontune.plugins` entry-point group.