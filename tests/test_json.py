"""Tests for canonical JSON plumbing and JSON-object recovery."""

from __future__ import annotations

from frictiontune.core.json import dumps, json_block_validity, loads, try_loads
from frictiontune.llm.parsing import extract_json_object


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


def test_json_block_validity_ignores_absent_json() -> None:
    # No JSON emitted means nothing is malformed.
    assert json_block_validity("plain prose answer") == 1.0


def test_json_block_validity_scores_parseable_blocks() -> None:
    assert json_block_validity('```json\n{"a": 1}\n```') == 1.0
    assert json_block_validity('```json\n{"a": }\n```') == 0.0


def test_json_block_validity_averages_over_blocks() -> None:
    text = '```json\n{"a": 1}\n```\n```json\n{oops}\n```'
    assert json_block_validity(text) == 0.5
