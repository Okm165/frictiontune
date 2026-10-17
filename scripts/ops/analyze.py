#!/usr/bin/env python3
"""Anomaly and gap inference over the run registry.

Reads the per-stage registry produced by :mod:`log_registry` and turns it into
three artefacts an operator can act on:

* ``anomalies.json`` — typed findings (run, stage, class, severity, evidence),
* ``ANALYSIS.md``    — the human report,
* ``gaps.md``        — candidate engineering *gaps*, each backed by the
  anomalies that imply it and a suggested action.

Scope: this reasons about **operational telemetry** (retries, 5xx, stage
outcomes, metric sanity, duration outliers). It never makes a semantic judgment
about the pipeline's *data* — that stays with the pipeline's LLM agents, per the
project's first principle. It is a detector, not an oracle: every finding cites
the metric or log line it came from so a human (or an agent) can adjudicate.

Usage::

    python analyze.py --root journal/runs
    python analyze.py --watch --interval 30        # monitor sidecar
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Stage metrics that must be strictly positive on a healthy run, with the
# severity of a zero reading. Absent metrics are skipped (the stage may not have
# run); present-but-zero is what we flag.
_POSITIVE_METRICS: dict[str, dict[str, str]] = {
    "friction_detect": {"events": "S2", "episodes_with_friction": "S2"},
    "mine": {"candidates": "S1"},
    "judge": {"judged": "S1"},
    "build_sft": {"examples": "S1"},
    "curate": {"kept": "S1"},
    "decontaminate": {"train": "S1"},
    "compile_rules": {"rules_generated": "S3"},
    "train_sft": {"steps": "S1"},
}

# Fraction thresholds: (metric, comparison, threshold, severity, message).
_RATIO_CHECKS: dict[str, list[tuple[str, str, float, str, str]]] = {
    "friction_detect": [
        ("coverage", "<", 0.15, "S2", "detector coverage is low; most episodes abstain"),
    ],
    "decontaminate": [
        ("train", "<", 1.0, "S2", "no training examples survived decontamination"),
    ],
}

_SEVERITY_ORDER = {"S1": 0, "S2": 1, "S3": 2}


@dataclass
class Anomaly:
    """One operational finding, with the evidence that produced it."""

    severity: str
    pipeline: str
    stamp: str
    stage: str
    kind: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class Gap:
    """A candidate engineering gap implied by one or more anomalies."""

    id: str
    title: str
    severity: str
    action: str
    evidence: list[str] = field(default_factory=list)


def _finite(value: Any) -> bool:
    """True for a real, finite number."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_runs(root: Path) -> list[dict[str, Any]]:
    """Load every ``summary.json`` under ``root/<pipeline>/<stamp>``."""
    runs: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/*/summary.json")):
        try:
            runs.append(json.loads(summary_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return runs


def _stage_durations(runs: list[dict[str, Any]]) -> dict[str, float]:
    """Median duration per stage across all runs that completed it."""
    buckets: dict[str, list[float]] = {}
    for run in runs:
        for rec in run.get("stages", []):
            dur = rec.get("duration_s")
            if rec.get("status") == "done" and _finite(dur):
                buckets.setdefault(rec["stage"], []).append(float(dur))
    medians: dict[str, float] = {}
    for stage, values in buckets.items():
        values.sort()
        medians[stage] = values[len(values) // 2]
    return medians


def analyse(runs: list[dict[str, Any]]) -> list[Anomaly]:
    """Compute anomalies across all runs."""
    anomalies: list[Anomaly] = []
    medians = _stage_durations(runs)

    def add(run: dict[str, Any], stage: str, kind: str, sev: str, detail: str, **ev: Any) -> None:
        anomalies.append(
            Anomaly(
                severity=sev,
                pipeline=run.get("pipeline", ""),
                stamp=run.get("stamp", ""),
                stage=stage,
                kind=kind,
                detail=detail,
                evidence=ev,
            )
        )

    for run in runs:
        for rec in run.get("stages", []):
            stage = rec.get("stage", "?")
            if rec.get("status") != "done":
                add(
                    run,
                    stage,
                    "stage_failed",
                    "S1",
                    "stage did not reach done",
                    status=rec.get("status"),
                )

            if rec.get("retries"):
                add(
                    run,
                    stage,
                    "provider_retry",
                    "S3",
                    "transient provider retries",
                    retries=rec["retries"],
                )
            if rec.get("http_5xx"):
                add(
                    run,
                    stage,
                    "provider_5xx",
                    "S2",
                    "upstream 5xx during stage",
                    http_5xx=rec["http_5xx"],
                )
            if rec.get("errors"):
                add(run, stage, "stage_errors", "S2", "error-level log lines", errors=rec["errors"])

            metrics = rec.get("metrics") or {}
            for metric, sev in _POSITIVE_METRICS.get(stage, {}).items():
                if metric in metrics and _finite(metrics[metric]) and float(metrics[metric]) == 0.0:
                    add(run, stage, "zero_output", sev, f"{metric} is zero", metric=metric)

            if stage == "train_sft":
                loss = metrics.get("final_loss", metrics.get("train_loss"))
                if loss is not None and (not _finite(loss) or float(loss) <= 0):
                    add(
                        run,
                        stage,
                        "loss_unhealthy",
                        "S1",
                        "final loss is non-positive/non-finite",
                        loss=loss,
                    )

            if (
                stage == "build_sft"
                and metrics.get("examples")
                and metrics.get("truncated") == metrics.get("examples")
            ):
                add(
                    run,
                    stage,
                    "all_truncated",
                    "S3",
                    "every SFT example hit the sequence cap",
                    **{"examples": metrics["examples"], "truncated": metrics["truncated"]},
                )

            for metric, op, threshold, sev, msg in _RATIO_CHECKS.get(stage, []):
                val = metrics.get(metric)
                if _finite(val) and op == "<" and float(val) < threshold:
                    add(
                        run,
                        stage,
                        "metric_low",
                        sev,
                        msg,
                        metric=metric,
                        value=val,
                        threshold=threshold,
                    )

            dur = rec.get("duration_s")
            median = medians.get(stage)
            if _finite(dur) and median and float(dur) > 3.0 * median and median > 5.0:
                add(
                    run,
                    stage,
                    "duration_outlier",
                    "S3",
                    "stage ran >3x its median",
                    duration_s=dur,
                    median_s=round(median, 1),
                )

        # Attempt counters live on the run, not the stage.
        for code in run.get("exit_codes", []):
            if code != 0:
                add(run, "_run", "nonzero_exit", "S1", "attempt exited non-zero", rc=code)
    return anomalies


_GAP_RULES: list[tuple[str, str, str, str, str]] = [
    # (kind match, id, title, severity, action)
    (
        "stage_failed",
        "GAP-A1",
        "A stage aborted instead of degrading per item",
        "S1",
        "Make fan-out stages tolerate per-item failures (see common.gather_tolerant).",
    ),
    (
        "nonzero_exit",
        "GAP-A1",
        "A stage aborted instead of degrading per item",
        "S1",
        "Inspect the stage's tail log; harden the failing fan-out or contract.",
    ),
    (
        "provider_5xx",
        "GAP-A2",
        "The LLM endpoint sheds load under a 5xx storm",
        "S2",
        "Verify retry budget/backoff for this environment; consider a wider RetryPolicy.",
    ),
    (
        "provider_retry",
        "GAP-A2",
        "The LLM endpoint sheds load under a 5xx storm",
        "S3",
        "Monitor retry incidence; widen backoff if it grows.",
    ),
    (
        "zero_output",
        "GAP-A3",
        "A stage produced no output",
        "S1",
        "Check the upstream artifact and gate/threshold; a zero here starves dependents.",
    ),
    (
        "loss_unhealthy",
        "GAP-A4",
        "Training loss is not healthy",
        "S1",
        "Check learning rate, epochs and data volume; a non-finite loss needs a run abort.",
    ),
    (
        "metric_low",
        "GAP-A5",
        "A quality metric is below its floor",
        "S2",
        "Review the responsible prompt/taxonomy against the evidence episodes.",
    ),
    (
        "all_truncated",
        "GAP-A6",
        "Every SFT example hit the sequence cap",
        "S3",
        "Add bounded per-event context windowing so completions are not starved.",
    ),
    (
        "duration_outlier",
        "GAP-A7",
        "A stage ran far slower than its median",
        "S3",
        "Check for provider slowness or a cache miss that invalidated upstream stages.",
    ),
]


def find_gaps(anomalies: list[Anomaly]) -> list[Gap]:
    """Collapse anomalies into deduplicated candidate gaps."""
    gaps: dict[str, Gap] = {}
    by_kind: dict[str, list[Anomaly]] = {}
    for anomaly in anomalies:
        by_kind.setdefault(anomaly.kind, []).append(anomaly)
    for kind, gap_id, title, sev, action in _GAP_RULES:
        hits = by_kind.get(kind, [])
        if not hits:
            continue
        gap = gaps.setdefault(gap_id, Gap(id=gap_id, title=title, severity=sev, action=action))
        for hit in hits[:5]:
            gap.evidence.append(
                f"{hit.pipeline}/{hit.stamp} {hit.stage}: {hit.detail} ({hit.evidence})"
            )
    return sorted(gaps.values(), key=lambda g: _SEVERITY_ORDER.get(g.severity, 9))


def _render(runs: list[dict[str, Any]], anomalies: list[Anomaly], gaps: list[Gap]) -> str:
    """Render the markdown analysis report."""
    lines: list[str] = ["# Run analysis", ""]
    lines.append(f"- runs registered: **{len(runs)}**")
    counts = {sev: sum(1 for a in anomalies if a.severity == sev) for sev in ("S1", "S2", "S3")}
    lines.append(
        f"- anomalies: **{len(anomalies)}**  "
        f"(S1={counts['S1']}, S2={counts['S2']}, S3={counts['S3']})"
    )
    lines.append(f"- gap candidates: **{len(gaps)}**")
    lines.append("")

    lines += [
        "## Stage outcomes",
        "",
        "| pipeline | stamp | stage | status | duration | metrics |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        for rec in run.get("stages", []):
            dur = rec.get("duration_s")
            dur_s = f"{dur:.1f}s" if _finite(dur) else "?"
            metrics = ", ".join(f"{k}={v}" for k, v in (rec.get("metrics") or {}).items())
            lines.append(
                f"| {run.get('pipeline')} | {run.get('stamp')} | {rec.get('stage')} | "
                f"{rec.get('status')} | {dur_s} | {metrics} |"
            )
    lines.append("")

    lines += ["## Anomalies", ""]
    if not anomalies:
        lines.append("_none_")
    else:
        lines += ["| sev | stage | kind | detail | evidence |", "| --- | --- | --- | --- | --- |"]
        for a in sorted(anomalies, key=lambda x: _SEVERITY_ORDER.get(x.severity, 9)):
            lines.append(f"| {a.severity} | {a.stage} | {a.kind} | {a.detail} | `{a.evidence}` |")
    lines.append("")

    lines += ["## Candidate gaps", ""]
    if not gaps:
        lines.append("_none_")
    else:
        for gap in gaps:
            lines.append(f"### {gap.id} [{gap.severity}] {gap.title}")
            lines.append(f"- action: {gap.action}")
            for ev in gap.evidence:
                lines.append(f"- evidence: {ev}")
            lines.append("")
    return "\n".join(lines) + "\n"


def write_reports(
    root: Path, runs: list[dict[str, Any]], anomalies: list[Anomaly], gaps: list[Gap]
) -> None:
    """Write anomalies.json, ANALYSIS.md and gaps.md under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "anomalies.json").write_text(
        json.dumps([asdict(a) for a in anomalies], indent=2), encoding="utf-8"
    )
    (root / "ANALYSIS.md").write_text(_render(runs, anomalies, gaps), encoding="utf-8")
    gap_lines = ["# Candidate gaps", ""]
    for gap in gaps:
        gap_lines += [f"## {gap.id} [{gap.severity}] {gap.title}", f"{gap.action}", ""]
        gap_lines += [f"- {ev}" for ev in gap.evidence]
        gap_lines.append("")
    (root / "gaps.md").write_text("\n".join(gap_lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("journal/runs"))
    parser.add_argument("--watch", action="store_true", help="re-analyse every --interval seconds")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args(argv)

    while True:
        runs = load_runs(args.root)
        anomalies = analyse(runs)
        gaps = find_gaps(anomalies)
        write_reports(args.root, runs, anomalies, gaps)
        counts = {sev: sum(1 for a in anomalies if a.severity == sev) for sev in ("S1", "S2", "S3")}
        print(
            f"[analyze] runs={len(runs)} anomalies={len(anomalies)} {counts} gaps={len(gaps)}",
            flush=True,
        )
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
