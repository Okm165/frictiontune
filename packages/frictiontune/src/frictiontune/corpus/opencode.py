"""Read an OpenCode SQLite corpus into canonical domain objects.

This is pure plumbing: open the database read-only and immutable (so a live
session is never disturbed), parse the JSON message payloads, and yield typed
:class:`Session` and :class:`Message` objects. Every row is parsed defensively;
unparseable rows are skipped and counted rather than crashing the ingest.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from frictiontune.core.json import dumps, try_loads
from frictiontune.domain.trace import (
    CompactionPart,
    FileRefPart,
    IdlePart,
    Message,
    Part,
    ReasoningPart,
    Session,
    SystemPart,
    TextPart,
    ToolPart,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@dataclass(slots=True)
class CorpusStats:
    """Counters describing how clean the parse was."""

    sessions: int = 0
    messages: int = 0
    skipped: int = 0
    tool_calls: int = 0
    tool_errors: int = 0


def _epoch(value: Any) -> int:
    """Coerce a stored timestamp to an integer epoch, or ``0`` if it is not one.

    Real databases store timestamps as integers, floats, or occasionally strings;
    a non-numeric value must not crash the whole ingest, so it degrades to ``0``.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _model_id(value: Any) -> str | None:
    """Return a model/agent id from a stored value or its JSON string.

    Sessions and assistant turns both store their model as a JSON object
    (``{"id": ...}``) or a bare id; normalizing to the id here keeps
    ``Session.model`` and ``Message.model`` consistent (GAP-004).
    """
    if isinstance(value, str):
        decoded = try_loads(value)
        if isinstance(decoded, dict):
            identifier = decoded.get("id")
            return str(identifier) if identifier else None
        return value or None
    if isinstance(value, dict):
        identifier = value.get("id")
        return str(identifier) if identifier else None
    return None


def _switch_text(kind: str, data: dict[str, Any]) -> str:
    """Human-readable note for an ``agent-switched``/``model-switched`` event.

    These rows were previously dropped without a trace (GAP-004); a mid-session
    switch is real context, so it is rendered as a plain system note rather than
    discarded.
    """
    label = "model" if kind == "model-switched" else "agent"
    current = _model_id(data.get("model") or data.get("agent")) or "unknown"
    previous = _model_id(data.get("previous")) or "unknown"
    return f"[{label} switched: {previous} -> {current}]"


def _open(db_path: Path, params: str) -> sqlite3.Connection:
    """Open a read-only connection that forces SQLite to touch the file now."""
    connection = sqlite3.connect(f"file:{db_path}?{params}", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA query_only = 1")
    # sqlite3.connect is lazy; read the header so an unusable mode fails here,
    # where the caller can fall back, rather than on the first real query.
    connection.execute("PRAGMA schema_version").fetchone()
    return connection


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open the corpus read-only without disturbing a live writer.

    A database with a sidecar write-ahead log is being actively written and must
    be opened ``mode=ro`` so SQLite replays the WAL: ``immutable=1`` would
    silently ignore it and hide the newest rows. A database with no WAL is at
    rest, so it is opened ``immutable=1`` first — that touches no ``-shm``/``-wal``
    file next to the source (the point of GAP-011) — and only falls back to
    ``mode=ro`` if the immutable open is impossible. Both modes are attempted
    either way, so a read-only filesystem degrades rather than fails.
    """
    live = db_path.with_name(db_path.name + "-wal").exists()
    modes = ("mode=ro", "mode=ro&immutable=1") if live else ("mode=ro&immutable=1", "mode=ro")
    error: sqlite3.OperationalError | None = None
    for params in modes:
        try:
            return _open(db_path, params)
        except sqlite3.OperationalError as exc:
            error = exc
    raise sqlite3.OperationalError(f"could not open corpus {db_path}: {error}")


def _content_text(content: Any) -> str:
    """Render a tool result payload to text without discarding structured output.

    A plain string is used as-is; a content-part list is concatenated; a dict
    prefers its ``text`` field and otherwise serializes faithfully, so an object
    result (some tools return one) still reaches the trained transcript.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                chunks.append(text if isinstance(text, str) else dumps(item))
        return "\n".join(chunks)
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else dumps(content)
    if content is None:
        return ""
    return str(content)


def _parse_assistant_parts(parts: Any) -> tuple[list[Part], int, int]:
    parsed: list[Part] = []
    tool_calls = 0
    tool_errors = 0
    if not isinstance(parts, list):
        return parsed, tool_calls, tool_errors
    for part in parts:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "reasoning":
            parsed.append(ReasoningPart(text=str(part.get("text") or "")))
        elif kind == "text":
            parsed.append(TextPart(text=str(part.get("text") or "")))
        elif kind == "tool":
            raw_state = part.get("state")
            state: dict[str, Any] = raw_state if isinstance(raw_state, dict) else {}
            status = str(state.get("status") or part.get("status") or "")
            is_error = status in {"error", "failed"} or bool(state.get("error"))
            output = _content_text(state.get("content"))
            raw_input = state.get("input")
            if isinstance(raw_input, dict):
                arguments = raw_input
            elif isinstance(raw_input, str):
                decoded = try_loads(raw_input)
                arguments = decoded if isinstance(decoded, dict) else {}
            else:
                arguments = {}
            parsed.append(
                ToolPart(
                    name=str(part.get("name") or "unknown"),
                    call_id=str(part.get("callID") or part.get("id") or "") or None,
                    arguments=arguments,
                    status=status or None,
                    output=output or None,
                    is_error=is_error,
                )
            )
            tool_calls += 1
            tool_errors += int(is_error)
        elif kind == "step-start":
            continue
        elif kind == "compaction":
            parsed.append(CompactionPart(text=str(part.get("text") or "")))
    return parsed, tool_calls, tool_errors


def _parse_user_parts(data: dict[str, Any]) -> list[Part]:
    parts: list[Part] = []
    text = data.get("text")
    if isinstance(text, str) and text.strip():
        parts.append(TextPart(text=text))
    files = data.get("files")
    if isinstance(files, list):
        for item in files:
            if isinstance(item, dict):
                # Stored file attachments carry a name, a uri, and a mime type
                # rather than a path; prefer the most human-readable identifier so
                # an attachment never degrades to the literal "file".
                reference = item.get("name") or item.get("uri") or item.get("mime") or "file"
                parts.append(
                    FileRefPart(
                        path=str(reference),
                        description=str(item.get("description") or ""),
                    )
                )
    return parts


def _message_from_row(row: sqlite3.Row, index: int) -> tuple[Message, int, int] | None:
    raw = row["data"]
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    kind = str(row["type"])
    session_id = str(row["session_id"])
    time_created = _epoch(row["time_created"])

    if kind == "assistant":
        parts, tool_calls, tool_errors = _parse_assistant_parts(data.get("content"))
        model_id = _model_id(data.get("model"))
        message = Message(
            id=str(row["id"]),
            session_id=session_id,
            seq=index,
            role="assistant",
            time_created=time_created,
            parts=parts,
            agent=str(data.get("agent") or "") or None,
            model=str(model_id) if model_id else None,
        )
        return message, tool_calls, tool_errors

    if kind == "user":
        message = Message(
            id=str(row["id"]),
            session_id=session_id,
            seq=index,
            role="user",
            time_created=time_created,
            parts=_parse_user_parts(data),
        )
        return message, 0, 0

    if kind == "compaction":
        # The stored compaction message carries the generated `summary` (and the
        # retained `recent` tail), not a `text` field; reading `text` alone would
        # silently drop the session summary from the transcript.
        summary = data.get("summary") or data.get("text") or data.get("recent") or ""
        message = Message(
            id=str(row["id"]),
            session_id=session_id,
            seq=index,
            role="system",
            time_created=time_created,
            parts=[CompactionPart(text=str(summary))],
        )
        return message, 0, 0

    if kind in {"idle", "synthetic"}:
        message = Message(
            id=str(row["id"]),
            session_id=session_id,
            seq=index,
            role="system",
            time_created=time_created,
            parts=[IdlePart()],
        )
        return message, 0, 0

    if kind == "system":
        message = Message(
            id=str(row["id"]),
            session_id=session_id,
            seq=index,
            role="system",
            time_created=time_created,
            parts=[SystemPart(text=str(data.get("text") or ""))],
        )
        return message, 0, 0

    if kind in {"model-switched", "agent-switched"}:
        message = Message(
            id=str(row["id"]),
            session_id=session_id,
            seq=index,
            role="system",
            time_created=time_created,
            parts=[SystemPart(text=_switch_text(kind, data))],
        )
        return message, 0, 0

    return None


class OpenCodeCorpus:
    """Streaming reader over an OpenCode database."""

    def __init__(self, db_path: Path) -> None:
        """Bind the reader to a SQLite database path."""
        self._db_path = db_path

    def iter_sessions(self) -> Iterator[tuple[Session, list[Message], CorpusStats]]:
        """Yield each session with its ordered messages and parse stats."""
        with _connect(self._db_path) as connection:
            session_rows = connection.execute(
                "SELECT * FROM session_v2 ORDER BY time_created"
            ).fetchall()
            cursor = connection.cursor()
            for session_row in session_rows:
                columns = set(session_row.keys())
                session_id = str(session_row["id"])
                session = Session(
                    id=session_id,
                    project=str(session_row["project_id"]) if "project_id" in columns else None,
                    model=_model_id(session_row["model"])
                    if "model" in columns and session_row["model"]
                    else None,
                    agent=None,
                    title=str(session_row["title"])
                    if "title" in columns and session_row["title"]
                    else None,
                    time_created=_epoch(session_row["time_created"])
                    if "time_created" in columns
                    else 0,
                )
                message_rows = cursor.execute(
                    "SELECT * FROM session_message WHERE session_id = ? "
                    "ORDER BY time_created, seq, id",
                    (session_id,),
                ).fetchall()
                stats = CorpusStats(sessions=1)
                messages: list[Message] = []
                for index, row in enumerate(message_rows):
                    parsed = _message_from_row(row, index)
                    if parsed is None:
                        stats.skipped += 1
                        continue
                    message, tool_calls, tool_errors = parsed
                    messages.append(message)
                    stats.messages += 1
                    stats.tool_calls += tool_calls
                    stats.tool_errors += tool_errors
                yield session, messages, stats
