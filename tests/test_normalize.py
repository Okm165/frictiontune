"""Tests for recursive redaction plumbing in ``normalize``.

Secrets hide in tool-call ``arguments``, not only in top-level ``text``/``output``.
The scan must reach every string leaf while leaving structural keys (which the
message's validity and tool linkage depend on) untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from frictiontune.domain.trace import Message, ReasoningPart, TextPart, ToolPart
from frictiontune.stages.normalize import (
    _get_at,
    _intent_context,
    _iter_string_leaves,
    _line_chunks,
    _number_lines,
    _schema_examples_text,
    _set_at,
    _shrink_example,
    elide_lines,
)

if TYPE_CHECKING:
    from frictiontune.core.json import JsonObject


def test_elide_lines_drops_lines_and_marks_gaps() -> None:
    text = "a\nb\nc\nd\ne"

    kept = elide_lines(text, [2, 3])

    assert kept == "a\n… 2 lines elided …\nd\ne"


def test_elide_lines_ignores_out_of_range_and_duplicate_numbers() -> None:
    # A hallucinated number can only fail to elide; it must never crash or corrupt.
    assert elide_lines("a\nb", [0, 1]) == "… 1 line elided …\nb"
    assert elide_lines("a\nb", [50, 60]) == "a\nb"
    assert elide_lines("a\nb", [1, 1]) == "… 1 line elided …\nb"
    assert elide_lines("a\nb", []) == "a\nb"


def test_elide_lines_preserves_a_trailing_newline_and_crlf() -> None:
    # Reconstruction is byte-exact: a no-op must not strip or rewrite terminators.
    assert elide_lines("a\nb\n", []) == "a\nb\n"
    assert elide_lines("a\r\nb", []) == "a\r\nb"
    assert elide_lines("a\nb\n", [1]) == "… 1 line elided …\nb\n"


def test_intent_context_uses_assistant_text_and_reasoning_not_tool_output() -> None:
    message = Message(
        id="m1",
        session_id="s1",
        role="assistant",
        seq=0,
        time_created=0,
        parts=[
            ReasoningPart(text="I will read the config"),
            TextPart(text="Reading it now"),
            ToolPart(kind="tool", name="read", call_id="c1", output="SECRET-BODY"),
        ],
    )

    context = _intent_context(message, 10_000)

    assert "read the config" in context
    assert "Reading it now" in context
    assert "SECRET-BODY" not in context


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


def test_line_chunks_keep_global_start_line_numbers() -> None:
    lines = [f"line {index}" for index in range(1, 8)]

    chunks = list(_line_chunks(lines, limit=20))

    # Each chunk starts at the global 1-based line number it follows from.
    assert [start for start, _chunk in chunks] == [1, 3, 5, 7]
    assert [chunk for _start, chunk in chunks] == [
        lines[0:2],
        lines[2:4],
        lines[4:6],
        lines[6:],
    ]


def test_line_chunks_emit_one_oversized_line_whole() -> None:
    # A single line longer than the limit is never split, so numbering stays
    # aligned: the model can point at a whole line and never at a half of one.
    huge = "x" * 500

    chunks = list(_line_chunks(["a", huge, "b"], limit=10))

    assert [(start, len(chunk)) for start, chunk in chunks] == [(1, 1), (2, 1), (3, 1)]
    assert chunks[1][1] == [huge]


def test_number_lines_offset_keeps_chunk_indices_absolute() -> None:
    # A later chunk is numbered from where the previous one stopped, so the
    # ranges it returns refer to the whole output, not the chunk.
    numbered = _number_lines("b\nc", start=2)

    assert numbered.splitlines()[0].strip() == "2 | b"
    assert numbered.splitlines()[1].strip() == "3 | c"


def test_shrink_example_truncates_long_values_and_caps_lists() -> None:
    # Schema inference reads shape, not content: a huge argument must not blow the
    # prompt past a server slot (the 400 that aborted the data phase).
    shrunk = _shrink_example({"content": "x" * 5000, "n": 3, "items": list(range(50))})

    assert isinstance(shrunk, dict)
    assert len(shrunk["content"]) <= 201  # 200 + the ellipsis
    assert shrunk["content"].endswith("…")
    assert shrunk["n"] == 3
    assert len(shrunk["items"]) == 10  # list length capped


def test_schema_examples_text_is_bounded() -> None:
    examples: list[JsonObject] = [{"path": "a", "content": "y" * 5000} for _ in range(12)]

    text = _schema_examples_text(examples, total=2_000)

    assert len(text) <= 2_500  # bounded well under any slot, even after the shrink
