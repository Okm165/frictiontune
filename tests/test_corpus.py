"""Tests for defensive corpus parsing.

Real databases store heterogeneous payloads: tool results may be strings, content
part lists, or objects, and timestamps may be strings. Each must be parsed
faithfully or degraded, never crash the ingest.
"""

from __future__ import annotations

from frictiontune.corpus.opencode import _content_text, _epoch


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
