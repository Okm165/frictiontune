"""The ``frictiontune`` terminal entry point.

Every subcommand is a thin shell over the library: load a validated config,
build the :class:`~frictiontune.application.Application`, and either inspect the
pipeline or run it. Nothing here makes a semantic decision; it only presents
what the pipeline already recorded.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from frictiontune.application import Application
from frictiontune.config.loader import load_config, parse_set_overrides
from frictiontune.core.errors import FrictionTuneError
from frictiontune.evaluation import GoldenReport, evaluate_golden, load_bundled_golden, load_golden
from frictiontune.pipeline.dag import full_graph
from frictiontune.pipeline.manifest import RunManifest, read_run_manifest
from frictiontune.pipeline.orchestrator import Orchestrator
from frictiontune.pipeline.registry import default_registry, validate_stage_settings
from frictiontune.reporting.dashboard import ReplayDashboard
from frictiontune.reporting.report import human_size, render_dashboard, render_run_report
from frictiontune.version import __version__

if TYPE_CHECKING:
    from collections.abc import Sequence

    from frictiontune.config.models import RunConfig
    from frictiontune.core.logging import LogFormat, LogLevel

app = typer.Typer(
    name="frictiontune",
    help=(
        "Mine agent chat logs for friction, compile rules, and "
        "fine-tune agents that need less steering."
    ),
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_DEFAULT_CONFIG_DIR = Path("configs")
_DEFAULT_PIPELINE = "core"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load(
    config_dir: Path,
    *,
    hardware: str | None,
    environment: str | None,
    overrides: Sequence[str],
) -> RunConfig:
    """Load a validated run configuration from the layered config tree."""
    return load_config(
        config_dir,
        hardware_preset=hardware,
        environment=environment,
        overrides=parse_set_overrides(list(overrides)),
    )


def _fail(message: str, code: int = 1) -> typer.Exit:
    console.print(f"[bold red]error:[/] {message}")
    return typer.Exit(code)


def _summary(config: RunConfig) -> Table:
    table = Table(title="resolved configuration", show_header=False)
    table.add_column("key", style="bold")
    table.add_column("value")
    table.add_row("project", config.project)
    table.add_row("seed", str(config.seed))
    table.add_row("artifacts_dir", str(config.paths.artifacts_dir))
    table.add_row("cache_dir", str(config.paths.cache_dir))
    corpus = str(config.paths.corpus_db) if config.paths.corpus_db else "(unset)"
    table.add_row("corpus_db", corpus)
    table.add_row(
        "hardware",
        f"{config.hardware.devices} dev / {config.hardware.quant} / {config.hardware.dtype}",
    )
    table.add_row("roles", ", ".join(sorted(role.value for role in config.llm)))
    table.add_row("taxonomy leaves", str(len(config.taxonomy.leaves)))
    table.add_row("stage blocks", ", ".join(sorted(config.stages)) or "(none)")
    return table


def _manifest_path(config: RunConfig, pipeline: str) -> Path:
    return config.paths.artifacts_dir / pipeline / "manifest.json"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the frictiontune version."""
    console.print(f"frictiontune {__version__}")


@app.command()
def stages(
    config_dir: Path = typer.Option(_DEFAULT_CONFIG_DIR, "--config-dir", help="Config directory."),
    hardware: str | None = typer.Option(None, "--hardware", help="Hardware preset name."),
    environment: str | None = typer.Option(None, "--environment", help="Environment overlay name."),
    set_: list[str] = typer.Option([], "--set", "-s", help="Override a config key: a.b=c."),
    lead: str | None = typer.Option(None, "--lead", help="Filter by lead: core or extension."),
) -> None:
    """List every stage with its ROI contract."""
    try:
        config = _load(config_dir, hardware=hardware, environment=environment, overrides=set_)
    except FrictionTuneError as exc:
        raise _fail(str(exc)) from exc
    application = Application.build(config)
    table = Table(title="pipeline stages")
    for column in ("stage", "tier", "lead", "roi", "compute", "inputs", "outputs"):
        table.add_column(column)
    for stage in application.stages.all():
        spec = stage.spec
        if lead and spec.lead.value != lead:
            continue
        table.add_row(
            spec.name,
            spec.tier.value,
            spec.lead.value,
            spec.roi,
            spec.cost.compute,
            spec.input_label(),
            ", ".join(spec.outputs) or "-",
        )
    console.print(table)
    console.print(
        f"[dim]Tier C stages are gated; run with --allow-gated. "
        f"{len(application.stages.core())} core / "
        f"{len(application.stages.all()) - len(application.stages.core())} extension.[/]"
    )


@app.command()
def validate(
    config_dir: Path = typer.Option(_DEFAULT_CONFIG_DIR, "--config-dir", help="Config directory."),
    hardware: str | None = typer.Option(None, "--hardware", help="Hardware preset name."),
    environment: str | None = typer.Option(None, "--environment", help="Environment overlay name."),
    set_: list[str] = typer.Option([], "--set", "-s", help="Override a config key: a.b=c."),
) -> None:
    """Load and validate the configuration, then print a summary."""
    try:
        config = _load(config_dir, hardware=hardware, environment=environment, overrides=set_)
        validate_stage_settings(config, default_registry())
    except FrictionTuneError as exc:
        raise _fail(str(exc)) from exc
    console.print(_summary(config))
    console.print("[bold green]configuration is valid[/]")


@app.command()
def run(
    config_dir: Path = typer.Option(_DEFAULT_CONFIG_DIR, "--config-dir", help="Config directory."),
    hardware: str | None = typer.Option(None, "--hardware", help="Hardware preset name."),
    environment: str | None = typer.Option(None, "--environment", help="Environment overlay name."),
    set_: list[str] = typer.Option([], "--set", "-s", help="Override a config key: a.b=c."),
    stage: list[str] = typer.Option([], "--stage", help="Run only these stages (repeatable)."),
    include_deps: bool = typer.Option(
        False, "--include-deps/--no-include-deps", help="Add upstream stages."
    ),
    allow_gated: bool = typer.Option(False, "--allow-gated", help="Permit Tier C stages to run."),
    force: bool = typer.Option(False, "--force", help="Ignore cached stage manifests."),
    pipeline: str = typer.Option(_DEFAULT_PIPELINE, "--pipeline", help="Pipeline name."),
    log_level: str = typer.Option("info", "--log-level", help="debug|info|warning|error|critical."),
    log_format: str = typer.Option("console", "--log-format", help="console|json."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the execution order and exit."),
) -> None:
    """Run the pipeline (or a subset of stages) end to end."""
    try:
        config = _load(config_dir, hardware=hardware, environment=environment, overrides=set_)
    except FrictionTuneError as exc:
        raise _fail(str(exc)) from exc

    level = _as_log_level(log_level)
    fmt = _as_log_format(log_format)
    graph = full_graph()

    selected = list(stage) or None
    try:
        application = Application.build(config, log_level=level, log_format=fmt)
    except FrictionTuneError as exc:
        raise _fail(str(exc)) from exc

    names = selected if selected is not None else [s.name for s in application.stages.core()]
    order = graph.closure(names) if include_deps else graph.order(names)
    if dry_run:
        console.print("execution order: " + " -> ".join(order))
        return

    async def _execute() -> RunManifest:
        context = application.run_context(pipeline)
        orchestrator = Orchestrator(
            run=context,
            registry=application.stages,
            graph=graph,
            force=force,
            allow_gated=allow_gated,
        )
        try:
            return await orchestrator.execute(selected, include_dependencies=include_deps)
        finally:
            await application.aclose()

    try:
        manifest = asyncio.run(_execute())
    except FrictionTuneError as exc:
        raise _fail(str(exc)) from exc

    table = Table(title=f"run: {pipeline}")
    for column in ("stage", "tier", "status", "time", "outputs"):
        table.add_column(column)
    for entry in manifest.stages:
        table.add_row(
            entry.stage,
            entry.spec.tier.value,
            "cached" if entry.cached else "ran",
            f"{entry.duration_s:.1f}s",
            ", ".join(entry.outputs),
        )
    console.print(table)
    console.print(f"[bold green]done[/] — wrote {_manifest_path(config, pipeline)}")


@app.command()
def inspect(
    config_dir: Path = typer.Option(_DEFAULT_CONFIG_DIR, "--config-dir", help="Config directory."),
    pipeline: str = typer.Option(_DEFAULT_PIPELINE, "--pipeline", help="Pipeline name."),
    stage: str | None = typer.Option(None, "--stage", help="Show artifacts of one stage only."),
) -> None:
    """List the stage runs and their artifacts for a pipeline."""
    try:
        config = _load(config_dir, hardware=None, environment=None, overrides=[])
    except FrictionTuneError as exc:
        raise _fail(str(exc)) from exc
    pipeline_dir = config.paths.artifacts_dir / pipeline
    if not pipeline_dir.is_dir():
        raise _fail(f"no artifacts under {pipeline_dir}")
    table = Table(title=f"artifacts: {pipeline}")
    for column in ("stage", "run_id", "artifact", "size"):
        table.add_column(column)
    for stage_dir in sorted(pipeline_dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        if stage and stage_dir.name != stage:
            continue
        for run_dir in sorted(stage_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            for item in sorted(run_dir.rglob("*")):
                if item.is_file() and item.name != "manifest.json":
                    table.add_row(
                        stage_dir.name,
                        run_dir.name[:12],
                        str(item.relative_to(run_dir)),
                        human_size(item.stat().st_size),
                    )
    console.print(table)


@app.command()
def report(
    config_dir: Path = typer.Option(_DEFAULT_CONFIG_DIR, "--config-dir", help="Config directory."),
    pipeline: str = typer.Option(_DEFAULT_PIPELINE, "--pipeline", help="Pipeline name."),
) -> None:
    """Render the markdown report for a completed run."""
    try:
        config = _load(config_dir, hardware=None, environment=None, overrides=[])
    except FrictionTuneError as exc:
        raise _fail(str(exc)) from exc
    path = _manifest_path(config, pipeline)
    if not path.is_file():
        raise _fail(f"no run manifest at {path}")
    manifest = read_run_manifest(path)
    console.print(render_run_report(manifest))


@app.command()
def dashboard(
    config_dir: Path = typer.Option(_DEFAULT_CONFIG_DIR, "--config-dir", help="Config directory."),
    pipeline: str = typer.Option(_DEFAULT_PIPELINE, "--pipeline", help="Pipeline name."),
) -> None:
    """Render the multi-signal replay dashboard from the last run."""
    try:
        config = _load(config_dir, hardware=None, environment=None, overrides=[])
    except FrictionTuneError as exc:
        raise _fail(str(exc)) from exc
    pipeline_dir = config.paths.artifacts_dir / pipeline
    candidate = _latest_dashboard(pipeline_dir)
    if candidate is None:
        raise _fail("no dashboard artifact found; run the evaluate_replay stage first")
    console.print(render_dashboard(ReplayDashboard.model_validate_json(candidate.read_text())))


def _latest_dashboard(pipeline_dir: Path) -> Path | None:
    """Return the dashboard of the most recent ``evaluate_replay`` run.

    The run manifest names the exact run the last execution used, so prefer it.
    Run ids are content hashes with no temporal order, so a lexical scan would
    otherwise surface an arbitrary (possibly stale) run; fall back to the newest
    run directory by mtime when no manifest is available.
    """
    manifest_path = pipeline_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = read_run_manifest(manifest_path)
        for entry in reversed(manifest.stages):
            if entry.stage != "evaluate_replay":
                continue
            named = pipeline_dir / "evaluate_replay" / entry.run_id
            if named.is_dir():
                found = next(iter(named.glob("dashboard*")), None)
                if found is not None:
                    return found
            break
    run_dirs = sorted(
        (path for path in pipeline_dir.glob("evaluate_replay/*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stage_dir in run_dirs:
        found = next(iter(stage_dir.glob("dashboard*")), None)
        if found is not None:
            return found
    return None


@app.command()
def eval_prompts(
    config_dir: Path = typer.Option(_DEFAULT_CONFIG_DIR, "--config-dir", help="Config directory."),
    hardware: str | None = typer.Option(None, "--hardware", help="Hardware preset name."),
    environment: str | None = typer.Option(None, "--environment", help="Environment overlay name."),
    set_: list[str] = typer.Option([], "--set", "-s", help="Override a config key: a.b=c."),
    fixture: Path | None = typer.Option(
        None, "--fixture", help="Golden JSONL path (defaults to the bundled set)."
    ),
) -> None:
    """Score the friction analyst on the golden episodes (calls an LLM endpoint)."""
    try:
        config = _load(config_dir, hardware=hardware, environment=environment, overrides=set_)
    except FrictionTuneError as exc:
        raise _fail(str(exc)) from exc

    episodes = load_golden(fixture) if fixture is not None else load_bundled_golden()
    if not episodes:
        raise _fail("the golden fixture is empty")
    try:
        application = Application.build(config)
    except FrictionTuneError as exc:
        raise _fail(str(exc)) from exc

    async def _score() -> GoldenReport:
        try:
            return await evaluate_golden(application.gateway, config, episodes)
        finally:
            await application.aclose()

    try:
        report = asyncio.run(_score())
    except FrictionTuneError as exc:
        raise _fail(str(exc)) from exc
    console.print(report.summary())


@app.command()
def tui() -> None:
    """Launch the interactive terminal UI (requires the ``tui`` extra)."""
    try:
        from frictiontune.tui.app import run_tui
    except ImportError as exc:  # pragma: no cover - only when extra is absent
        raise _fail(
            "the TUI requires optional dependencies: uv sync --package frictiontune --extra tui"
        ) from exc
    run_tui()


# ---------------------------------------------------------------------------
# Small presentation helpers
# ---------------------------------------------------------------------------


_LOG_LEVELS: dict[str, LogLevel] = {
    "debug": "debug",
    "info": "info",
    "warning": "warning",
    "error": "error",
    "critical": "critical",
}
_LOG_FORMATS: dict[str, LogFormat] = {"console": "console", "json": "json"}


def _as_log_level(value: str) -> LogLevel:
    level = _LOG_LEVELS.get(value)
    if level is None:
        raise _fail(f"unknown log level {value!r}; expected one of {sorted(_LOG_LEVELS)}")
    return level


def _as_log_format(value: str) -> LogFormat:
    log_format = _LOG_FORMATS.get(value)
    if log_format is None:
        raise _fail(f"unknown log format {value!r}; expected console or json")
    return log_format


if __name__ == "__main__":  # pragma: no cover - console-script fallback
    app()
