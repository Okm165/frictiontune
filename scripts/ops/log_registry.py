#!/usr/bin/env python3
"""Register frictiontune runs' per-stage logs into a queryable registry.

The pipeline emits one combined stream per invocation (stdout + stderr). That is
fine to watch live and terrible to reason about afterwards: a multi-hour run is a
single blob in which a warning at hour two is separated from its stage by
thousands of lines. This tool does the plumbing that turns the blob back into
structure:

* split the stream into a segment per stage (``stages/<stage>.log``),
* parse each stage's start/done line into a typed record (run id, tier,
  duration, metrics, outputs),
* count operational signals that matter later (retries, HTTP 5xx, warnings,
  errors),
* write a machine-readable ``index.jsonl`` and a run ``summary.json``.

It makes no semantic judgment about the pipeline's data; it only reads the
operator log. ``analyze.py`` consumes the registry.

Usage::

    python log_registry.py register --pipeline v10 --stamp 2026-10-14T0536Z \\
        --log journal/partial/v10_run5.txt
    python log_registry.py backfill                      # every run*.txt on disk
    python log_registry.py show --pipeline v10
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# journal/runs/<pipeline>/<stamp>/{run.log,index.jsonl,summary.json,stages/*.log}
REGISTRY_ROOT = Path("journal/runs")
PARTIAL_ROOT = Path("journal/partial")

_LOG_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T[\d:.]+Z)\s+\[(?P<level>[a-z]+)\s*\]\s+(?P<msg>.*)$"
)
_KV_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>\S+)")
_METRICS_RE = re.compile(r"metrics=(?P<metrics>\{.*\})\s*(?:outputs=|$)")
_LINE_STASH = "journal/runs/.registry"


def _parse_ts(raw: str) -> datetime | None:
    """Parse a UTC ISO-8601 timestamp, tolerating a ``Z`` suffix."""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _literal(raw: str) -> Any:
    """Parse a Python literal (dict/list) or return the raw string."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


@dataclass
class StageRecord:
    """One stage execution parsed out of the combined run stream."""

    stage: str
    status: str = "unknown"  # started | done | failed
    run_id: str = ""
    tier: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    started: str = ""
    ended: str = ""
    duration_s: float | None = None
    retries: int = 0
    http_5xx: int = 0
    warnings: int = 0
    errors: int = 0

    def finish(self, ended: str) -> None:
        """Stamp the end time and compute the wall-clock duration."""
        self.ended = ended
        start = _parse_ts(self.started)
        end = _parse_ts(ended)
        if start is not None and end is not None:
            self.duration_s = round((end - start).total_seconds(), 3)


@dataclass
class RunSummary:
    """Aggregate view of one run invocation."""

    pipeline: str
    stamp: str
    source_log: str
    attempts: int = 0
    exit_codes: list[int] = field(default_factory=list)
    stages: list[StageRecord] = field(default_factory=list)

    @property
    def failed(self) -> list[str]:
        """Names of stages that did not reach ``done``."""
        return [s.stage for s in self.stages if s.status != "done"]

    def totals(self) -> dict[str, int]:
        """Operational counters summed across stages."""
        return {
            "stages_done": sum(1 for s in self.stages if s.status == "done"),
            "stages_failed": sum(1 for s in self.stages if s.status != "done"),
            "retries": sum(s.retries for s in self.stages),
            "http_5xx": sum(s.http_5xx for s in self.stages),
            "warnings": sum(s.warnings for s in self.stages),
            "errors": sum(s.errors for s in self.stages),
        }


def parse_run_log(text: str) -> RunSummary:
    """Parse one combined run stream into stage records.

    The parser is intentionally forgiving: lines that are not structlog records
    (supervisor markers, multi-line provider error bodies, podman noise) are
    attributed to the stage that is currently open and never raise.
    """
    summary = RunSummary(pipeline="", stamp="", source_log="")
    current: StageRecord | None = None
    for raw in text.splitlines():
        summary.attempts += int(raw.startswith("=== attempt ") and "start" in raw)
        m = re.match(r"^=== attempt \d+ exited rc=(-?\d+)", raw)
        if m:
            summary.exit_codes.append(int(m.group(1)))
        line = _LOG_RE.match(raw)
        if line is None:
            # Non-structlog line: fold it into the open stage's operational
            # counters, because provider error bodies arrive this way.
            if current is not None and ("Error code: 5" in raw or "'status': 5" in raw):
                current.http_5xx += 1
            continue
        level = line.group("level")
        msg = line.group("msg").strip()
        ts = line.group("ts")
        if current is not None and level == "warning":
            current.warnings += 1
            if msg.startswith("retrying call"):
                current.retries += 1
        if current is not None and level == "error":
            current.errors += 1
        if msg.startswith("stage start"):
            current = StageRecord(stage=_kv(msg, "stage"), status="started", started=ts)
        elif msg.startswith(("stage done", "stage failed")):
            if current is None:
                current = StageRecord(stage=_kv(msg, "stage"))
            current.status = "failed" if msg.startswith("stage failed") else "done"
            current.run_id = _kv(msg, "run_id") or current.run_id
            current.tier = _kv(msg, "tier") or current.tier
            current.inputs = _list_kv(msg, "inputs") or current.inputs
            current.outputs = _list_kv(msg, "outputs") or current.outputs
            current.metrics = _dict_kv(msg, "metrics") or current.metrics
            current.finish(ts)
            summary.stages.append(current)
            current = None
    return summary


def _kv(msg: str, key: str) -> str:
    """Return the scalar value of ``key`` in a structlog message."""
    m = re.search(rf"{re.escape(key)}=(\S+?)(?:\s|$)", msg)
    return m.group(1) if m else ""


def _list_kv(msg: str, key: str) -> list[str]:
    """Return a bracketed list value (e.g. ``inputs=['a', 'b']``)."""
    m = re.search(rf"{re.escape(key)}=(\[[^\]]*\])", msg)
    if not m:
        return []
    value = _literal(m.group(1))
    return [str(v) for v in value] if isinstance(value, list) else []


def _dict_kv(msg: str, key: str) -> dict[str, Any]:
    """Return a dict value (e.g. ``metrics={...}``) parsed leniently."""
    m = re.search(rf"{re.escape(key)}=(\{{.*?\}})(?:\s+\w+=|$)", msg)
    if not m:
        return {}
    value = _literal(m.group(1))
    return value if isinstance(value, dict) else {}


def register(pipeline: str, stamp: str, log_path: Path, root: Path = REGISTRY_ROOT) -> RunSummary:
    """Parse ``log_path`` and write the per-stage registry under ``root``."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    summary = parse_run_log(text)
    summary.pipeline = pipeline
    summary.stamp = stamp
    summary.source_log = str(log_path)

    out = root / pipeline / stamp
    stages_dir = out / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    # Split the raw stream into a file per stage for browsable evidence.
    _slice_stages(text, stages_dir)

    (out / "summary.json").write_text(
        json.dumps(asdict(summary) | {"totals": summary.totals()}, indent=2), encoding="utf-8"
    )
    with (out / "index.jsonl").open("w", encoding="utf-8") as fh:
        for rec in summary.stages:
            row = asdict(rec) | {"pipeline": pipeline, "stamp": stamp}
            fh.write(json.dumps(row) + "\n")
    _append_global_index(summary, root)
    return summary


def _slice_stages(text: str, stages_dir: Path) -> None:
    """Write one raw-log file per stage, plus a preamble for lead-in noise.

    A ``stage start`` line belongs to the stage it opens; a ``stage done`` line
    belongs to the stage it closes. Everything between two boundaries is
    attributed to the open stage, so provider error bodies land with their call.
    """
    current = "_preamble"
    buf: dict[str, list[str]] = {}
    for raw in text.splitlines():
        m = _LOG_RE.match(raw)
        msg = m.group("msg").strip() if m is not None else ""
        if msg.startswith("stage start"):
            current = _kv(msg, "stage")
        buf.setdefault(current, []).append(raw)
        if msg.startswith(("stage done", "stage failed")):
            current = "_between"
    for name, lines in buf.items():
        (stages_dir / f"{name}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_global_index(summary: RunSummary, root: Path) -> None:
    """Append a one-line pointer for this run to the registry-wide index."""
    root.mkdir(parents=True, exist_ok=True)
    row = {
        "pipeline": summary.pipeline,
        "stamp": summary.stamp,
        "source_log": summary.source_log,
        "attempts": summary.attempts,
        "exit_codes": summary.exit_codes,
        "totals": summary.totals(),
    }
    with (root / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def backfill(root: Path = REGISTRY_ROOT, partial: Path = PARTIAL_ROOT) -> list[RunSummary]:
    """Register every ``<pipeline>_run*.txt`` stream found under ``partial``."""
    seen: list[RunSummary] = []
    for log in sorted(partial.glob("*_run*.txt")):
        stem = log.stem  # e.g. v10_run5
        pipeline, _, run = stem.partition("_run")
        seen.append(register(pipeline or "v10", f"run{run}", log, root))
    return seen


def _show(pipeline: str, root: Path = REGISTRY_ROOT) -> int:
    """Print the stage table for every registered run of ``pipeline``."""
    for summary_path in sorted((root / pipeline).glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"== {pipeline} / {summary_path.parent.name} ==")
        for rec in summary["stages"]:
            dur = rec["duration_s"]
            dur_s = f"{dur:8.1f}s" if isinstance(dur, (int, float)) else "       ?"
            print(f"  {rec['stage']:16} {rec['status']:7} {dur_s}  {rec['metrics']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser("register", help="register one run stream")
    reg.add_argument("--pipeline", required=True)
    reg.add_argument("--stamp", required=True)
    reg.add_argument("--log", required=True, type=Path)

    sub.add_parser("backfill", help="register every run stream in journal/partial")

    show = sub.add_parser("show", help="print a pipeline's registered stages")
    show.add_argument("--pipeline", default="v10")

    args = parser.parse_args(argv)
    if args.cmd == "register":
        summary = register(args.pipeline, args.stamp, args.log)
        print(json.dumps(summary.totals()))
    elif args.cmd == "backfill":
        for summary in backfill():
            print(f"registered {summary.pipeline}/{summary.stamp}: {summary.totals()}")
    else:
        return _show(args.pipeline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
