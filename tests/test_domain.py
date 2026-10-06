"""Tests for the canonical trace domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from frictiontune.domain.trace import (
    Message,
    ReasoningPart,
    TextPart,
    ToolPart,
)


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


def test_has_tool_error_detects_failed_tool() -> None:
    ok = _message([{"kind": "tool", "name": "read", "output": "fine"}])
    bad = _message([{"kind": "tool", "name": "bash", "output": "boom", "is_error": True}])

    assert not ok.has_tool_error()
    assert bad.has_tool_error()


def test_discriminated_parts_parse_into_types() -> None:
    message = _message(
        [
            {"kind": "reasoning", "text": "r"},
            {"kind": "text", "text": "t"},
        ]
    )

    assert isinstance(message.parts[0], ReasoningPart)
    assert isinstance(message.parts[1], TextPart)


def test_unknown_part_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _message([{"kind": "nonsense", "text": "x"}])


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        _message([{"kind": "text", "text": "x", "surprise": 1}])


def test_tool_arguments_default_to_empty_mapping() -> None:
    part = ToolPart(name="read")
    assert part.arguments == {}
    assert part.is_error is False