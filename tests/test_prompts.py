"""Tests for the prompt registry: parsing, rendering, and library integrity."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

import frictiontune.domain as domain
import pytest
from frictiontune.config.models import LlmRoleConfig, RoleName
from frictiontune.core.errors import PromptError
from frictiontune.domain.responses import JudgeVerdict
from frictiontune.llm.client import LlmClient, LlmResponse
from frictiontune.llm.gateway import LlmGateway, schema_contract
from frictiontune.llm.prompts.registry import (
    PromptRegistry,
    PromptTemplate,
    default_registry,
    render,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from frictiontune.llm.messages import ChatMessage, ToolSpec

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


def test_every_prompt_declares_a_resolvable_schema() -> None:
    registry: PromptRegistry = default_registry()

    for name in registry.names():
        template = registry.get(name)
        assert template.schema_name, f"{name} declares no schema"
        assert hasattr(domain, template.schema_name), (
            f"{name} declares unknown schema {template.schema_name!r}"
        )


def test_prompt_metadata_is_consistent() -> None:
    registry: PromptRegistry = default_registry()

    for name in registry.names():
        template = registry.get(name)
        assert template.name == name, f"file {name}.md declares name {template.name!r}"
        assert template.role is not None, f"{name} declares no role"
        assert template.description.strip(), f"{name} has an empty description"
        assert template.version, f"{name} has no version"


def test_schema_contract_is_generated_from_the_model() -> None:
    contract = schema_contract(JudgeVerdict)

    assert "JSON Schema" in contract
    # Required and property names come from the pydantic model, not the prompt.
    assert "task_adherence" in contract
    assert "accept" in contract


class _StubClient:
    """A capture-only stand-in for :class:`LlmClient`."""

    def __init__(self) -> None:
        self.calls: list[Sequence[ChatMessage]] = []

    async def complete(
        self,
        role: RoleName,
        config: LlmRoleConfig,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
        force_json: bool = False,
        response_schema: dict[str, object] | None = None,
        schema_name: str = "response",
    ) -> LlmResponse:
        del role, config, tools, force_json, response_schema, schema_name
        self.calls.append(list(messages))
        verdict = (
            '{"task_adherence":5,"correctness":5,"friction_avoidance":5,'
            '"clarity":5,"evidence":5,"accept":true,"reason":"ok"}'
        )
        return LlmResponse(text=verdict, model="stub")


async def test_structured_call_injects_the_schema_contract() -> None:
    stub = _StubClient()
    gateway = LlmGateway(cast("LlmClient", stub), default_registry())

    verdict = await gateway.complete_structured(
        RoleName.JUDGE,
        LlmRoleConfig(model="stub"),
        "judge",
        {
            "candidate_id": "c1",
            "source": "unit",
            "track": "replay",
            "context": "do the thing",
            "completion": "the correct behaviour",
        },
        JudgeVerdict,
    )

    assert verdict.accept
    sent = stub.calls[0]
    assert [message.role for message in sent] == ["system", "system", "user"]
    assert "JSON Schema" in sent[1].content
    assert "task_adherence" in sent[1].content
