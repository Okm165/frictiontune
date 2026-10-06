"""Human-readable reports rendered from pipeline artifacts.

Reports are the explorable surface of the pipeline: every stage can render a
markdown summary a developer reads without opening the raw artifacts. Rendering
is pure and side-effect free; stages decide where to write the result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from frictiontune.domain.friction import FrictionProfile
    from frictiontune.pipeline.manifest import RunManifest
    from frictiontune.reporting.dashboard import ReplayDashboard


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    if not rows:
        return "_(none)_\n"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def _counter_table(title: str, counts: dict[str, int]) -> str:
    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return f"### {title}\n\n" + _table(["key", "count"], rows) + "\n"


def human_size(size: int) -> str:
    """Format a byte count in human-readable units."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{value:.1f}GB"


def render_run_report(manifest: RunManifest) -> str:
    """Render a run manifest as a markdown report."""
    lines: list[str] = [
        f"# Run report: {manifest.pipeline}",
        "",
        f"- project: `{manifest.project}`",
        f"- code revision: `{manifest.code_revision}`",
        f"- config fingerprint: `{manifest.config_fingerprint[:16]}`",
        f"- artifacts: `{manifest.artifacts_root}`",
        f"- started: {manifest.started_at}",
        f"- finished: {manifest.finished_at}",
        "",
        "## Stages",
        "",
    ]
    rows = [
        [
            entry.stage,
            entry.spec.tier.value,
            entry.spec.lead.value,
            "cached" if entry.cached else "ran",
            f"{entry.duration_s:.1f}s",
            ", ".join(entry.outputs),
        ]
        for entry in manifest.stages
    ]
    lines.append(_table(["stage", "tier", "lead", "status", "time", "outputs"], rows))
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    metric_rows = [
        [name, f"{value:.4f}"] for entry in manifest.stages for name, value in entry.metrics.items()
    ]
    lines.append(_table(["metric", "value"], metric_rows))
    return "\n".join(lines)


def render_friction_report(profile: FrictionProfile) -> str:
    """Render a friction profile as a markdown report."""
    lines: list[str] = [
        "# Friction profile",
        "",
        f"- episodes analysed: {profile.episodes_analysed}",
        f"- episodes with friction: {profile.episodes_with_friction} "
        f"({profile.coverage() * 100:.1f}%)",
        f"- total events: {profile.total_events}",
        f"- abstentions: {profile.abstentions}",
        "",
    ]
    lines.append(_counter_table("By form", profile.by_form))
    lines.append(_counter_table("By cause", profile.by_cause))
    lines.append(_counter_table("By leaf", profile.by_leaf))
    lines.append(_counter_table("By resolution", profile.by_resolution))
    return "\n".join(lines)


def render_dashboard(dashboard: ReplayDashboard) -> str:
    """Render the replay dashboard as a markdown report."""
    lines: list[str] = [
        "# Replay dashboard",
        "",
        f"**{dashboard.headline()}**",
        "",
        f"- episodes: {dashboard.episodes}",
        f"- seeds: {', '.join(str(seed) for seed in dashboard.seeds)}",
        f"- lucky-pass rate: {dashboard.lucky_pass_rate * 100:.1f}%",
        "",
        "## Signals",
        "",
    ]
    rows = [
        [
            signal.name,
            f"{signal.baseline:.3f}",
            f"{signal.tuned:.3f}",
            f"{signal.delta:+.3f}",
            f"[{signal.ci_low:+.3f}, {signal.ci_high:+.3f}]",
            "yes" if signal.improved else "no",
            signal.n,
        ]
        for signal in dashboard.signals
    ]
    lines.append(
        _table(
            ["signal", "baseline", "tuned", "delta", "95% CI", "improved", "n"],
            rows,
        )
    )
    if dashboard.notes:
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in dashboard.notes)
    return "\n".join(lines)
