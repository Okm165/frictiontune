"""Tests for the LLM client's structured-output request shaping."""

from __future__ import annotations

from frictiontune.config.models import LlmRoleConfig
from frictiontune.llm.client import build_response_format

_SCHEMA: dict[str, object] = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def test_text_role_is_left_alone_without_force() -> None:
    config = LlmRoleConfig(model="m", response_format="text")

    assert build_response_format(config, schema=None, schema_name="x", force_json=False) is None


def test_force_json_upgrades_a_text_role() -> None:
    config = LlmRoleConfig(model="m", response_format="text")

    assert build_response_format(config, schema=None, schema_name="x", force_json=True) == {
        "type": "json_object"
    }


def test_json_object_role_requests_json() -> None:
    config = LlmRoleConfig(model="m", response_format="json_object")

    assert build_response_format(config, schema=None, schema_name="x", force_json=False) == {
        "type": "json_object"
    }


def test_json_schema_role_sends_the_schema() -> None:
    config = LlmRoleConfig(model="m", response_format="json_schema")

    built = build_response_format(config, schema=_SCHEMA, schema_name="Verdict", force_json=True)

    assert built == {
        "type": "json_schema",
        "json_schema": {"name": "Verdict", "schema": _SCHEMA, "strict": True},
    }


def test_json_schema_role_degrades_without_a_schema() -> None:
    config = LlmRoleConfig(model="m", response_format="json_schema")

    assert build_response_format(config, schema=None, schema_name="x", force_json=True) == {
        "type": "json_object"
    }
    # Even without `force_json`, a json_schema role still asks for JSON.
    assert build_response_format(config, schema=None, schema_name="x", force_json=False) == {
        "type": "json_object"
    }
