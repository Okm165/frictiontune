"""Terminal UI for frictiontune.

The TUI is a *viewer*: it never runs a stage. It reads the same stage registry
and the same artifact tree the CLI uses, so the two can never disagree. The
optional ``textual`` dependency is confined to :mod:`frictiontune.tui.app`.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["run_tui"]


def run_tui(config_dir: Path = Path("configs")) -> None:
    """Launch the TUI, importing the UI module lazily."""
    from frictiontune.tui.app import run_tui as _run

    _run(config_dir)
