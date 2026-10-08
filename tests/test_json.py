"""Tests for canonical JSON plumbing and JSON-object recovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from frictiontune.core.errors import ArtifactError
from frictiontune.core.json import (
    dumps,
    iter_jsonl,
    json_block_validity,
    json_blocks,
    loads,
    try_loads,
)
from frictiontune.llm.parsing import extract_json_object, iter_json_objects

if TYPE_CHECKING:
    from pathlib import Path


def test_dumps_is_canonical_and_sorted() -> None:
    assert dumps({"b": 1, "a": [2, 3]}) == '{"a":[2,3],"b":1}'


def test_dumps_round_trips_unicode() -> None:
    payload = {"text": "café ✓"}
    assert loads(dumps(payload)) == payload


def test_try_loads_returns_none_on_garbage() -> None:
    assert try_loads("{not json") is None
    assert try_loads("null") is None


def test_extract_json_object_plain() -> None:
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_object_from_prose() -> None:
    text = 'Sure! Here it is:\n```json\n{"leaf": "scope_creep", "n": 2}\n```\nDone.'
    assert extract_json_object(text) == {"leaf": "scope_creep", "n": 2}


def test_extract_json_object_ignores_leading_prose() -> None:
    text = 'My answer follows.\n{"leaf": "no_verification", "severity": 3}'
    assert extract_json_object(text) == {"leaf": "no_verification", "severity": 3}


def test_extract_json_object_skips_braces_in_leading_prose() -> None:
    text = 'The set {x} is not JSON. Result:\n{"leaf": "tool_misuse", "n": 1}'
    assert extract_json_object(text) == {"leaf": "tool_misuse", "n": 1}


def test_extract_json_object_ignores_trailing_prose_with_braces() -> None:
    text = '{"ok": true} Note the placeholder {x} in the template.'
    assert extract_json_object(text) == {"ok": True}


def test_extract_json_object_recovers_after_a_malformed_object() -> None:
    text = '{not valid json} {"leaf": "verbosity"}'
    assert extract_json_object(text) == {"leaf": "verbosity"}


def test_extract_json_object_returns_none_without_object() -> None:
    assert extract_json_object("no json here") is None
    assert extract_json_object("[1, 2, 3]") is None


def test_extract_json_object_keeps_fences_inside_string_values() -> None:
    text = '{"kept": "example:\\n```\\ncode\\n```", "rationale": "ok"}'
    assert extract_json_object(text) == {"kept": "example:\n```\ncode\n```", "rationale": "ok"}


def test_extract_json_object_keeps_braces_inside_string_values() -> None:
    text = '{"kept": "template {x} here", "n": 1}'
    assert extract_json_object(text) == {"kept": "template {x} here", "n": 1}


def test_extract_json_object_does_not_leak_an_inner_object_from_truncation() -> None:
    # The trailing brace is inside the still-open string, so a truncated reply
    # must not be mined for the inner {"inner": 1}.
    text = '{"outer": "text {"inner": 1}'
    assert extract_json_object(text) is None


def test_iter_json_objects_yields_every_top_level_object() -> None:
    text = 'example {"a": 1} then {"b": 2}'
    assert list(iter_json_objects(text)) == [{"a": 1}, {"b": 2}]


def test_extract_json_object_finds_a_nested_object_whole() -> None:
    text = '{"outer": {"inner": 1}, "ok": true}'
    assert extract_json_object(text) == {"outer": {"inner": 1}, "ok": True}


def test_json_block_validity_handles_nested_objects() -> None:
    assert json_block_validity('```json\n{"a": {"b": 1}}\n```') == 1.0


def test_json_block_validity_ignores_absent_json() -> None:
    # No JSON emitted means nothing is malformed.
    assert json_block_validity("plain prose answer") == 1.0


def test_json_block_validity_scores_parseable_blocks() -> None:
    assert json_block_validity('```json\n{"a": 1}\n```') == 1.0
    assert json_block_validity('```json\n{"a": }\n```') == 0.0


def test_json_block_validity_averages_over_blocks() -> None:
    text = '```json\n{"a": 1}\n```\n```json\n{oops}\n```'
    assert json_block_validity(text) == 0.5


def test_json_blocks_selects_only_bare_and_json_fences() -> None:
    text = (
        "```python\nprint(1)\n```\n"
        '```\n{"a": 1}\n```\n'
        '```json\n{"b": 2}\n```\n'
        "```bash\necho hi\n```"
    )
    assert [block.strip() for block in json_blocks(text)] == ['{"a": 1}', '{"b": 2}']


def test_json_block_validity_ignores_non_json_fences() -> None:
    # A python/bash fence is code, not JSON, so it is never "malformed JSON".
    assert json_block_validity("```python\nprint(1)\n```") == 1.0
    assert json_block_validity("```python\nprint(1)\n```\n```json\n{oops}\n```") == 0.0


def test_iter_jsonl_names_the_corrupt_line(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    target.write_text('{"a": 1}\n{oops}\n', encoding="utf-8")

    with pytest.raises(ArtifactError, match=r"rows\.jsonl:2"):
        list(iter_jsonl(target))


def test_iter_jsonl_can_skip_invalid_lines_for_external_corpora(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    target.write_text('{"a": 1}\n{oops}\n{"b": 2}\n', encoding="utf-8")

    assert list(iter_jsonl(target, skip_invalid=True)) == [{"a": 1}, {"b": 2}]
