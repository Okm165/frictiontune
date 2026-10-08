"""Tests for defensive corpus parsing.

Real databases store heterogeneous payloads: tool results may be strings, content
part lists, or objects, and timestamps may be strings. Each must be parsed
faithfully or degraded, never crash the ingest.
"""

from __future__ import annotations

import json
import sqlite3

from frictiontune.corpus.opencode import (
    _content_text,
    _epoch,
    _message_from_row,
    _parse_assistant_parts,
    _parse_user_parts,
)
from frictiontune.domain.trace import CompactionPart, FileRefPart, TextPart, ToolPart


def test_epoch_accepts_numeric_and_numeric_string_forms() -> None:
    assert _epoch(1_700_000_000) == 1_700_000_000
    assert _epoch(1_700_000_000.9) == 1_700_000_000
    assert _epoch("1700000000") == 1_700_000_000


def test_epoch_degrades_non_timestamps_to_zero() -> None:
    assert _epoch(None) == 0
    assert _epoch("2024-01-01T00:00:00Z") == 0
    assert _epoch(True) == 0


def test_content_text_keeps_object_results_faithfully() -> None:
    assert _content_text('{"ok": true}') == '{"ok": true}'
    # A dict with a text field prefers that field over a JSON dump.
    assert _content_text({"text": "hello"}) == "hello"
    # A dict without one serializes rather than vanishing.
    assert _content_text({"output": "boom"}) == '{"output":"boom"}'


def test_content_text_concatenates_content_part_lists() -> None:
    assert _content_text(["a", {"text": "b"}]) == "a\nb"
    assert _content_text(None) == ""


def test_assistant_tool_part_reads_name_and_content_output() -> None:
    # The stored assistant content uses `name` and `state.content` (a list of
    # text/file parts), not `tool`/`state.output`.
    parts, calls, errors = _parse_assistant_parts(
        [
            {
                "type": "tool",
                "id": "prt_1",
                "name": "bash",
                "state": {
                    "status": "completed",
                    "input": {"command": "ls"},
                    "content": [{"type": "text", "text": "out"}],
                    "structured": {},
                },
            },
            {
                "type": "tool",
                "id": "prt_2",
                "name": "bash",
                "state": {
                    "status": "error",
                    "input": {"command": "false"},
                    "content": [{"type": "text", "text": "boom"}],
                    "structured": {},
                    "error": {"type": "unknown", "message": "boom"},
                },
            },
        ]
    )

    assert calls == 2
    assert errors == 1
    first, second = parts
    assert isinstance(first, ToolPart)
    assert first.name == "bash"
    assert first.output == "out"
    assert first.is_error is False
    assert isinstance(second, ToolPart)
    assert second.output == "boom"
    assert second.is_error is True


def test_user_file_attachments_keep_name_then_uri() -> None:
    parts = _parse_user_parts(
        {
            "text": "review",
            "files": [
                {
                    "uri": "file:///a.py",
                    "mime": "text/x-python",
                    "name": "a.py",
                    "description": "the file",
                },
                {"uri": "file:///b.png", "mime": "image/png"},
            ],
        }
    )

    assert parts[0] == TextPart(text="review")
    assert parts[1] == FileRefPart(path="a.py", description="the file")
    assert parts[2] == FileRefPart(path="file:///b.png", description="")


def _row(columns: dict[str, object]) -> sqlite3.Row:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE t (id TEXT, session_id TEXT, type TEXT, time_created INTEGER, data TEXT)"
    )
    connection.execute(
        "INSERT INTO t VALUES (:id, :session_id, :type, :time_created, :data)", columns
    )
    row = connection.execute("SELECT * FROM t").fetchone()
    assert isinstance(row, sqlite3.Row)
    return row


def test_compaction_message_keeps_the_generated_summary() -> None:
    row = _row(
        {
            "id": "msg_1",
            "session_id": "ses_1",
            "type": "compaction",
            "time_created": 5,
            "data": json.dumps(
                {
                    "type": "compaction",
                    "reason": "auto",
                    "summary": "we did X",
                    "recent": "tail",
                }
            ),
        }
    )

    parsed = _message_from_row(row, 0)

    assert parsed is not None
    message, tool_calls, tool_errors = parsed
    assert message.role == "system"
    assert message.parts == [CompactionPart(text="we did X")]
    assert (tool_calls, tool_errors) == (0, 0)
