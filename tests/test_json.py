"""Tests for canonical JSON plumbing and JSON-object recovery."""

from __future__ import annotations

from frictiontune.core.json import dumps, loads, try_loads
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


def test_extract_json_object_returns_none_without_object() -> None:
    assert extract_json_object("no json here") is None
    assert extract_json_object("[1, 2, 3]") is None