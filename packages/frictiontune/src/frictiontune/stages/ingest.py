"""``ingest`` — read the source corpus into canonical, streamed messages.

Read-only and immutable: a live agent session is never disturbed. Messages are
written session-by-session into JSONL through a generator, and the compact
session index is written through a sink in the same pass, so the process holds
only one session's messages at a time rather than the whole corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from frictiontune.core.errors import StageError
from frictiontune.core.hashing import file_sha256
from frictiontune.corpus.opencode import CorpusStats, OpenCodeCorpus
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings

if TYPE_CHECKING:
    from collections.abc import Iterator

    from frictiontune.domain.trace import Message
    from frictiontune.pipeline.context import RunContext, StageContext


class IngestSettings(StageSettings):
    """Where to read the corpus from and how much of it to take."""

    db_path: Path | None = None
    limit_sessions: int | None = None
    min_messages: int = 1


def _resolve_db(settings: IngestSettings, ctx: StageContext) -> Path:
    db = settings.db_path or ctx.config.paths.corpus_db
    if db is None:
        raise StageError("ingest", "no corpus_db configured (paths.corpus_db or db_path)")
    if not db.is_file():
        raise StageError("ingest", f"corpus database not found: {db}")
    return db


class Ingest(Stage):
    """Stream the corpus database into message and session artifacts."""

    spec = StageSpec(
        name="ingest",
        tier=Tier.A,
        lead=Lead.CORE,
        roi="High",
        justification="Nothing downstream exists without canonical turns.",
        evidence=["opencode.db: 153 sessions, 16,238 messages"],
        cost=CostProfile(compute="cpu", api_usd="none", maintenance="low"),
        inputs={},
        outputs={"messages": "jsonl", "sessions": "jsonl", "ingest_report": "json"},
        kill_criterion="Superseded only if the corpus format changes entirely.",
    )
    settings_model = IngestSettings

    def external_input_hashes(self, run: RunContext) -> list[str]:
        """Hash the corpus file so a changed corpus yields a new run id."""
        raw = run.config.stages.get("ingest", {})
        override = raw.get("db_path")
        db = Path(str(override)) if override else run.config.paths.corpus_db
        if db is not None and db.is_file():
            return [file_sha256(db)]
        return []

    async def run(self, ctx: StageContext) -> StageResult:
        """Read the database and emit messages, sessions, and a parse report."""
        settings = ctx.settings(self.settings_model)
        db = _resolve_db(settings, ctx)
        corpus = OpenCodeCorpus(db)

        session_sink = ctx.jsonl_sink("sessions")
        totals = CorpusStats()

        def message_rows() -> Iterator[Message]:
            for index, (session, messages, stats) in enumerate(corpus.iter_sessions()):
                if settings.limit_sessions is not None and index >= settings.limit_sessions:
                    break
                if len(messages) < settings.min_messages:
                    totals.skipped += 1
                    continue
                session_sink.write(session)
                totals.sessions += 1
                totals.messages += stats.messages
                totals.skipped += stats.skipped
                totals.tool_calls += stats.tool_calls
                totals.tool_errors += stats.tool_errors
                yield from messages

        # emit_jsonl consumes the generator fully, writing sessions alongside.
        messages_ref = ctx.emit_jsonl("messages", message_rows())
        session_sink.close()

        report = {
            "database": str(db),
            "sessions": totals.sessions,
            "messages": totals.messages,
            "messages_written": messages_ref.rows,
            "skipped_sessions_or_rows": totals.skipped,
            "tool_calls": totals.tool_calls,
            "tool_errors": totals.tool_errors,
            "tool_error_rate": (
                totals.tool_errors / totals.tool_calls if totals.tool_calls else 0.0
            ),
            "limit_sessions": settings.limit_sessions,
        }
        ctx.emit_json("ingest_report", report)
        tool_error_pct = totals.tool_errors / totals.tool_calls * 100 if totals.tool_calls else 0.0
        ctx.emit_text(
            "report",
            "# Ingest\n\n"
            f"- database: `{db}`\n"
            f"- sessions: {totals.sessions}\n"
            f"- messages: {totals.messages}\n"
            f"- tool calls: {totals.tool_calls} "
            f"(errors {totals.tool_errors}, {tool_error_pct:.2f}%)\n",
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "sessions": float(totals.sessions),
                "messages": float(totals.messages),
                "tool_calls": float(totals.tool_calls),
                "tool_errors": float(totals.tool_errors),
            },
            notes=[f"skipped {totals.skipped} unparseable rows"] if totals.skipped else [],
        )
