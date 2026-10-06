"""The Textual application: a read-only viewer over stages and artifacts.

Two tables, both derived from the same sources as the CLI:

* **Stages** — every registered stage with its ROI contract.
* **Artifacts** — the on-disk artifact tree of the last run per pipeline.

Launch with ``frictiontune tui`` (after ``uv sync --package frictiontune --extra tui``).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import DataTable, Footer, Header, Static, TabbedContent, TabPane

from frictiontune.application import Application
from frictiontune.config.loader import load_config
from frictiontune.core.errors import FrictionTuneError
from frictiontune.reporting.report import human_size

_DEFAULT_CONFIG_DIR = Path("configs")


class PipelineView(App[None]):
    """A read-only, refreshable view of the pipeline and its artifacts."""

    TITLE = "frictiontune"
    SUB_TITLE = "friction-aware agent fine-tuning"
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("q", "quit", "Quit"),
        ("r", "reload", "Reload"),
    ]
    CSS = """
    TabbedContent { height: 1fr; }
    DataTable { height: 1fr; }
    #status { height: 1; color: $text-muted; }
    """

    def __init__(self, config_dir: Path = _DEFAULT_CONFIG_DIR) -> None:
        """Bind the view to a config directory."""
        super().__init__()
        self._config_dir = config_dir
        self._status = "ready"

    def compose(self) -> ComposeResult:
        """Compose the header, the two tables, and the status line."""
        yield Header()
        with Horizontal():
            yield Static(id="status")
        with TabbedContent():
            with TabPane("Stages", id="tab-stages"):
                yield DataTable(id="stages")
            with TabPane("Artifacts", id="tab-artifacts"):
                yield DataTable(id="artifacts")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the tables once the widgets exist."""
        self.action_reload()

    def action_reload(self) -> None:
        """Rebuild both tables from the current config and artifact tree."""
        stages = self.query_one("#stages", DataTable)
        artifacts = self.query_one("#artifacts", DataTable)
        status = self.query_one("#status", Static)
        stages.clear(columns=True)
        artifacts.clear(columns=True)

        try:
            config = load_config(self._config_dir)
        except FrictionTuneError as exc:
            status.update(f"[red]{exc}[/]")
            return

        stages.add_columns("stage", "tier", "lead", "roi", "compute", "inputs", "outputs")
        try:
            application = Application.build(config)
        except FrictionTuneError as exc:
            status.update(f"[red]{exc}[/]")
            return
        for stage in application.stages.all():
            spec = stage.spec
            stages.add_row(
                spec.name,
                spec.tier.value,
                spec.lead.value,
                spec.roi,
                spec.cost.compute,
                ", ".join(spec.inputs) or "-",
                ", ".join(spec.outputs) or "-",
            )

        artifacts.add_columns("pipeline", "stage", "run", "artifact", "size")
        count = 0
        root = config.paths.artifacts_dir
        if root.is_dir():
            for pipeline_dir in sorted(root.iterdir()):
                if not pipeline_dir.is_dir():
                    continue
                for stage_dir in sorted(pipeline_dir.iterdir()):
                    if not stage_dir.is_dir():
                        continue
                    for run_dir in sorted(stage_dir.iterdir()):
                        if not run_dir.is_dir():
                            continue
                        for item in sorted(run_dir.rglob("*")):
                            if item.is_file() and item.name != "manifest.json":
                                artifacts.add_row(
                                    pipeline_dir.name,
                                    stage_dir.name,
                                    run_dir.name[:12],
                                    str(item.relative_to(run_dir)),
                                    human_size(item.stat().st_size),
                                )
                                count += 1
        status.update(f"{len(application.stages.all())} stages · {count} artifacts under {root}")


def run_tui(config_dir: Path = _DEFAULT_CONFIG_DIR) -> None:
    """Run the TUI against ``config_dir`` in a blocking loop."""
    PipelineView(config_dir).run()
