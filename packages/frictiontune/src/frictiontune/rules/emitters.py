"""Emit a rule library into deployable runtime artifacts.

The same verified rules are packaged in two forms:

* a system-prompt block an agent can ingest directly, and
* a machine-readable policy a runtime checker or MCP server can execute.

Emitters are pure functions over a :class:`~frictiontune.domain.rules.RuleLibrary`;
adding a target means adding one emitter, never touching the compiler.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from frictiontune.core.json import dumps
from frictiontune.domain.rules import RuleLibrary, RuleStatus, RuntimeRule

EmitterTarget = Literal["system_prompt", "json_policy"]


class EmittedPolicy(BaseModel):
    """One emitted runtime artifact."""

    model_config = ConfigDict(extra="forbid")

    target: EmitterTarget
    filename: str
    content: str
    rule_ids: list[str] = Field(default_factory=list)
    rule_count: int = 0


@runtime_checkable
class RuleEmitter(Protocol):
    """Turns active rules into a deployable artifact."""

    target: EmitterTarget

    def emit(self, library: RuleLibrary) -> EmittedPolicy:
        """Emit ``library`` into this emitter's target format."""
        ...


def _active(library: RuleLibrary) -> list[RuntimeRule]:
    return [
        rule for rule in library.rules if rule.status in {RuleStatus.ACTIVE, RuleStatus.VERIFIED}
    ]


class SystemPromptEmitter:
    """Render active rules as a concise, injectable system-prompt block."""

    target: EmitterTarget = "system_prompt"

    def emit(self, library: RuleLibrary) -> EmittedPolicy:
        """Render the rules as markdown instructions."""
        rules = _active(library)
        lines: list[str] = [
            "# Runtime rules",
            "",
            "Apply a rule only when its applicability condition holds.",
            "",
        ]
        for rule in rules:
            lines.append(f"## {rule.id}")
            lines.append(f"- When: {rule.applicability}")
            lines.append(f"- Do: {rule.instruction}")
            if rule.verifier.failure_message:
                lines.append(f"- Violation: {rule.verifier.failure_message}")
            lines.append("")
        content = "\n".join(lines).rstrip() + "\n"
        return EmittedPolicy(
            target=self.target,
            filename="rules.system_prompt.md",
            content=content,
            rule_ids=[rule.id for rule in rules],
            rule_count=len(rules),
        )


class JsonPolicyEmitter:
    """Render active rules as a machine-readable policy document."""

    target: EmitterTarget = "json_policy"

    def emit(self, library: RuleLibrary) -> EmittedPolicy:
        """Render the rules as canonical JSON."""
        rules = _active(library)
        payload = {
            "version": "1",
            "generated_from": library.generated_from,
            "rules": [
                {
                    "id": rule.id,
                    "applicability": rule.applicability,
                    "instruction": rule.instruction,
                    "scope": rule.scope.model_dump(mode="json"),
                    "verifier": rule.verifier.model_dump(mode="json"),
                }
                for rule in rules
            ],
        }
        return EmittedPolicy(
            target=self.target,
            filename="rules.policy.json",
            content=dumps(payload),
            rule_ids=[rule.id for rule in rules],
            rule_count=len(rules),
        )


def default_emitters() -> list[RuleEmitter]:
    """Return every built-in emitter."""
    return [SystemPromptEmitter(), JsonPolicyEmitter()]


def emit_all(library: RuleLibrary) -> list[EmittedPolicy]:
    """Emit ``library`` with every default emitter."""
    return [emitter.emit(library) for emitter in default_emitters()]
