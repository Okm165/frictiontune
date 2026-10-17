"""Tests for canonical JSON plumbing and JSON-object recovery."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from frictiontune.core.errors import ArtifactError
from frictiontune.core.json import (
    dumps,
    iter_jsonl,
    json_block_validity,
    json_blocks,
    try_loads,
)
from frictiontune.llm.parsing import extract_json_object, iter_json_objects

if TYPE_CHECKING:
    from pathlib import Path


def test_dumps_is_canonical_and_sorted() -> None:
    assert dumps({"b": 1, "a": [2, 3]}) == '{"a":[2,3],"b":1}'


def test_try_loads_returns_none_on_garbage() -> None:
    assert try_loads("{not json") is None
    assert try_loads("null") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        (
            'Sure! Here it is:\n```json\n{"leaf": "scope_creep", "n": 2}\n```\nDone.',
            {"leaf": "scope_creep", "n": 2},
        ),
        (
            'My answer follows.\n{"leaf": "no_verification", "severity": 3}',
            {"leaf": "no_verification", "severity": 3},
        ),
        # Leading/trailing prose may itself contain braces; braces in prose are
        # not structure, only a whole top-level object counts.
        (
            'The set {x} is not JSON. Result:\n{"leaf": "tool_misuse", "n": 1}',
            {"leaf": "tool_misuse", "n": 1},
        ),
        ('{"ok": true} Note the placeholder {x} in the template.', {"ok": True}),
        # A malformed object is skipped; the next whole object is recovered.
        ('{not valid json} {"leaf": "verbosity"}', {"leaf": "verbosity"}),
        # Braces and fences inside string values are content, not structure.
        (
            '{"kept": "example:\\n```\\ncode\\n```", "rationale": "ok"}',
            {"kept": "example:\n```\ncode\n```", "rationale": "ok"},
        ),
        ('{"kept": "template {x} here", "n": 1}', {"kept": "template {x} here", "n": 1}),
        ('{"outer": {"inner": 1}, "ok": true}', {"outer": {"inner": 1}, "ok": True}),
    ],
)
def test_extract_json_object_recovers_the_whole_object(text: str, expected: dict[str, Any]) -> None:
    assert extract_json_object(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        "[1, 2, 3]",  # a list is not the object we requested
        # The trailing brace is inside the still-open string, so a truncated
        # reply must not be mined for the inner {"inner": 1}.
        '{"outer": "text {"inner": 1}',
    ],
)
def test_extract_json_object_returns_none_without_a_whole_object(text: str) -> None:
    assert extract_json_object(text) is None


def test_iter_json_objects_yields_every_top_level_object() -> None:
    text = 'example {"a": 1} then {"b": 2}'
    assert list(iter_json_objects(text)) == [{"a": 1}, {"b": 2}]


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
