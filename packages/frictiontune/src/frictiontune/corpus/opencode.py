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


def _connect(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks)
    return ""


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
            arguments = state.get("input") if isinstance(state.get("input"), dict) else {}
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
                parts.append(
                    FileRefPart(
                        path=str(item.get("path") or item.get("name") or "file"),
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
    time_created = int(row["time_created"] or 0)

    if kind == "assistant":
        parts, tool_calls, tool_errors = _parse_assistant_parts(data.get("content"))
        model = data.get("model")
        model_id = model.get("id") if isinstance(model, dict) else None
        message = Message(
            id=str(row["id"]),
            session_id=session_id,
            seq=index,
            role="assistant",
            time_created=time_created,
            parts=parts,
            agent=str(data.get("agent") or "") or None,
            model=str(model_id) if model_id else None,
            raw=data,
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
            raw=data,
        )
        return message, 0, 0

    if kind == "compaction":
        message = Message(
            id=str(row["id"]),
            session_id=session_id,
            seq=index,
            role="system",
            time_created=time_created,
            parts=[CompactionPart(text=str(data.get("text") or ""))],
            raw=data,
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
            raw=data,
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
            raw=data,
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
                    model=str(session_row["model"])
                    if "model" in columns and session_row["model"]
                    else None,
                    agent=None,
                    title=str(session_row["title"])
                    if "title" in columns and session_row["title"]
                    else None,
                    time_created=int(session_row["time_created"] or 0)
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
