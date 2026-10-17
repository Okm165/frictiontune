"""Tests for the canonical trace domain models."""

from __future__ import annotations

import pytest
from frictiontune.domain.friction import RecoverySpan
from frictiontune.domain.trace import Message
from pydantic import ValidationError


def _message(parts: list[object]) -> Message:
    return Message.model_validate(
        {
            "id": "m1",
            "session_id": "s1",
            "seq": 0,
            "role": "assistant",
            "time_created": 0,
            "parts": parts,
        }
    )


def test_content_text_concatenates_text_bearing_parts() -> None:
    message = _message(
        [
            {"kind": "reasoning", "text": "thinking"},
            {"kind": "text", "text": "answer"},
            {"kind": "tool", "name": "read", "output": "file body"},
        ]
    )

    text = message.content_text()

    assert text.startswith("thinking")
    assert "answer" in text
    assert "[tool:read] file body" in text


def test_content_text_includes_bounded_tool_arguments() -> None:
    message = _message(
        [
            {
                "kind": "tool",
                "name": "bash",
                "arguments": {"command": "pytest -q"},
                "output": "1 failed",
            }
        ]
    )

    assert message.content_text() == '[tool:bash {"command":"pytest -q"}] 1 failed'


def test_content_text_bounds_huge_tool_arguments() -> None:
    message = _message(
        [{"kind": "tool", "name": "write", "arguments": {"content": "x" * 5_000}, "output": "ok"}]
    )

    text = message.content_text()

    assert "chars]" in text  # the explicit truncation marker
    assert text.endswith("ok")  # output is never crowded out


def test_has_tool_error_detects_failed_tool() -> None:
    ok = _message([{"kind": "tool", "name": "read", "output": "fine"}])
    bad = _message([{"kind": "tool", "name": "bash", "output": "boom", "is_error": True}])

    assert not ok.has_tool_error()
    assert bad.has_tool_error()


def test_recovery_span_requires_both_indices_when_found() -> None:
    # A recovery with ``found`` true but no explicit end is the shape the
    # endpoint emitted, silently discarding every candidate; it must not parse.
    with pytest.raises(ValidationError):
        RecoverySpan(found=True, recovery_start=4, recovery_end=None, trigger="self")


def test_recovery_span_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError):
        RecoverySpan(found=True, recovery_start=5, recovery_end=3, trigger="self")
