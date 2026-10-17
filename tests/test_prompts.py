"""Tests for the prompt registry: parsing, rendering, and library integrity."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

import frictiontune.domain as domain
import pytest
from frictiontune.config.models import LlmRoleConfig, RoleName
from frictiontune.core.errors import PromptError
from frictiontune.domain.friction import FrictionAnalysis
from frictiontune.domain.responses import InferredToolSchema, JudgeVerdict
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

_INCLUDE_DOC = """---
name: incl
version: "1"
role: judge
schema: DemoResponse
description: An include demo.
---
:::system
{{> greet }}
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


def test_includes_expand_recursively() -> None:
    template = PromptTemplate.parse(
        _INCLUDE_DOC, partials={"greet": "hello {{> who }}", "who": "world"}
    )

    assert template.system == "hello world"


def test_unknown_partial_is_an_error() -> None:
    with pytest.raises(PromptError):
        PromptTemplate.parse(_INCLUDE_DOC, partials={})


def test_cyclic_include_is_an_error() -> None:
    with pytest.raises(PromptError):
        PromptTemplate.parse(_INCLUDE_DOC, partials={"greet": "{{> greet }}"})


def test_library_prompts_have_no_unexpanded_includes() -> None:
    registry: PromptRegistry = default_registry()

    for name in registry.names():
        template = registry.get(name)
        assert "{{>" not in template.system, f"{name} left an include unexpanded"
        assert "{{>" not in template.user, f"{name} left an include unexpanded"


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


def test_schema_contract_preserves_declaration_order() -> None:
    contract = schema_contract(JudgeVerdict)

    # `reason` is declared before the scores and `accept`; the injected contract
    # must keep that order or reason-before-decision is lost.
    assert contract.index('"reason"') < contract.index('"accept"')


def test_schema_contract_inlines_nested_refs() -> None:
    contract = schema_contract(FrictionAnalysis)

    assert '"$ref"' not in contract
    assert '"$defs"' not in contract
    assert '"quote"' in contract  # EvidenceSpan.quote, inlined


def test_schema_contract_keeps_recursive_defs() -> None:
    contract = schema_contract(InferredToolSchema)

    # JsonValue is inherently recursive, so its definitions cannot be inlined.
    assert '"$defs"' in contract


def test_schema_contract_drops_title_noise() -> None:
    assert '"title"' not in schema_contract(JudgeVerdict)


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
    # One system message: the schema contract is folded into the prompt's system
    # text so endpoints that reject multiple system turns still work.
    assert [message.role for message in sent] == ["system", "user"]
    assert "JSON Schema" in sent[0].content
    assert "task_adherence" in sent[0].content


def test_prompt_fingerprint_is_stable_and_hex() -> None:
    registry = default_registry()

    first = registry.fingerprint()
    second = registry.fingerprint()

    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_prompt_fingerprint_scopes_to_the_named_prompts() -> None:
    class _DoctoredRegistry(PromptRegistry):
        """A registry whose ``judge`` prompt has been edited."""

        def get(self, name: str) -> PromptTemplate:
            template = super().get(name)
            if name == "judge":
                return template.model_copy(
                    update={"sections": {**template.sections, "system": "changed"}}
                )
            return template

    doctored = _DoctoredRegistry()
    clean = PromptRegistry()

    assert doctored.fingerprint(["judge"]) != clean.fingerprint(["judge"])
    # A stage that declares its prompts is keyed on those alone, so editing
    # `judge.md` must not re-key the stages that never call the judge.
    assert doctored.fingerprint(["dedup"]) == clean.fingerprint(["dedup"])
    # The whole-library default stays conservative.
    assert doctored.fingerprint() != clean.fingerprint()
    # Order is not part of the identity.
    assert clean.fingerprint(["judge", "dedup"]) == clean.fingerprint(["dedup", "judge"])


def test_prompt_fingerprint_carries_expanded_partials() -> None:
    registry = default_registry()
    template = registry.get("judge")
    body = template.system + template.user

    # Partials are expanded at parse time and the resolved text is what gets
    # hashed, so a change to a shared partial moves the fingerprint of exactly
    # the prompts that include it — no separate partial hash is needed.
    assert registry.partials()["untrusted-data"] in body
    assert "{{>" not in body
