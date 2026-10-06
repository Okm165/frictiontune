"""Reporting: markdown reports and the multi-signal replay dashboard."""

from frictiontune.reporting.dashboard import (
    PRIMARY_SIGNAL,
    ReplayDashboard,
    Signal,
    build_dashboard,
    make_signal,
    relative_drop,
)
from frictiontune.reporting.report import (
    render_dashboard,
    render_friction_report,
    render_run_report,
)

__all__ = [
    "PRIMARY_SIGNAL",
    "ReplayDashboard",
    "Signal",
    "build_dashboard",
    "make_signal",
    "relative_drop",
    "render_dashboard",
    "render_friction_report",
    "render_run_report",
]
