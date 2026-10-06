"""Tests for the prompt registry: parsing, rendering, and library integrity."""

from __future__ import annotations

import re

import pytest
from frictiontune.core.errors import PromptError
from frictiontune.llm.prompts.registry import (
    PromptRegistry,
    PromptTemplate,
    default_registry,
    render,
)

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

_MINIMAL_DOC = """---
name: demo
version: "1"
role: judge
schema: DemoResponse
description: A demo prompt.
---
:::system
You are a careful judge of {{ topic }}.
:::
:::user
Given {{ payload }}, answer.
:::
"""


def test_render_substitutes_and_rejects_missing() -> None:
    assert render("hi {{ name }}", {"name": "x"}) == "hi x"
    with pytest.raises(PromptError):
        render("hi {{ name }}", {})


def test_parse_minimal_document() -> None:
    template = PromptTemplate.parse(_MINIMAL_DOC)

    assert template.name == "demo"
    assert template.schema_name == "DemoResponse"
    system, user = template.render({"topic": "T", "payload": "P"})
    assert "careful judge of T" in system
    assert "Given P" in user


def test_parse_rejects_missing_sections() -> None:
    with pytest.raises(PromptError):
        PromptTemplate.parse("---\nname: x\nversion: '1'\n---\nno sections here\n")


def test_library_contains_the_expected_prompts() -> None:
    names = default_registry().names()

    assert len(names) >= 16
    assert "friction_analyst" in names
    assert "judge" in names


def test_every_library_prompt_parses_and_renders() -> None:
    registry: PromptRegistry = default_registry()

    for name in registry.names():
        template = registry.get(name)
        placeholders = set(_PLACEHOLDER.findall(template.system))
        placeholders |= set(_PLACEHOLDER.findall(template.user))
        values = {key: f"<{key}>" for key in placeholders}
        system, user = template.render(values)
        # No unresolved placeholder may survive a successful render.
        assert not _PLACEHOLDER.search(system)
        assert not _PLACEHOLDER.search(user)


def test_unknown_prompt_raises() -> None:
    with pytest.raises(PromptError):
        default_registry().get("definitely_not_a_prompt")
