"""Tests for recursive redaction plumbing in ``normalize``.

Secrets hide in tool-call ``arguments``, not only in top-level ``text``/``output``.
The scan must reach every string leaf while leaving structural keys (which the
message's validity and tool linkage depend on) untouched.
"""

from __future__ import annotations

from frictiontune.stages.normalize import _get_at, _iter_string_leaves, _set_at


def test_iter_string_leaves_reaches_tool_arguments_and_skips_structure() -> None:
    part = {
        "kind": "tool",
        "name": "read",
        "call_id": "c1",
        "status": "ok",
        "arguments": {"api_key": "sk-live-0123456789abcdef", "path": "a.py"},
    }

    values = {value for _path, value in _iter_string_leaves(part)}

    assert "sk-live-0123456789abcdef" in values
    assert "a.py" in values
    assert "read" not in values  # tool name is structural
    assert "c1" not in values  # call id links a call to its result
    assert "tool" not in values  # part discriminator
    assert "ok" not in values  # status


def test_get_and_set_at_navigate_a_nested_path() -> None:
    data = {"parts": [{"arguments": {"k": "v"}}]}

    _set_at(data, ("parts", 0, "arguments", "k"), "w")

    assert _get_at(data, ("parts", 0, "arguments", "k")) == "w"


def test_structural_keys_are_skipped_only_at_the_part_top_level() -> None:
    # `name` and `status` are structural at the part level, but inside a tool
    # call's arguments they are ordinary data and must still be scanned.
    part = {
        "kind": "tool",
        "name": "shell",
        "status": "ok",
        "arguments": {"name": "sk-live-0123456789abcdef", "status": "secret-value"},
    }

    values = {value for _path, value in _iter_string_leaves(part)}

    assert "shell" not in values
    assert "ok" not in values
    assert "sk-live-0123456789abcdef" in values
    assert "secret-value" in values
