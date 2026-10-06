"""Runtime rule hooks.

Verification is an agentic act, not a regex. :func:`compile_hooks` turns active
rules into lightweight descriptors an agent runtime can ship, and
:class:`RuleRuntime` evaluates them against a live transcript by calling the
same ``rule_verifier`` agent used offline. This keeps the offline gate and the
online check on exactly one implementation of the rule judgement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from frictiontune.config.models import RoleName, RunConfig
from frictiontune.domain.responses import RuleVerdict
from frictiontune.domain.rules import RuleLibrary, RuleStatus, RuleVerifier

if TYPE_CHECKING:
    from collections.abc import Sequence

    from frictiontune.llm.gateway import LlmGateway

_RULE_VERIFIER_PROMPT = "rule_verifier"


class RuntimeHook(BaseModel):
    """A deployable descriptor of one active rule."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    applicability: str
    instruction: str
    verifier: RuleVerifier
    tools: list[str] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)


class HookResult(BaseModel):
    """The outcome of evaluating one hook on a transcript."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    verdict: RuleVerdict


def compile_hooks(library: RuleLibrary) -> list[RuntimeHook]:
    """Compile every active/verified rule into a runtime hook."""
    hooks: list[RuntimeHook] = []
    for rule in library.rules:
        if rule.status not in {RuleStatus.ACTIVE, RuleStatus.VERIFIED}:
            continue
        hooks.append(
            RuntimeHook(
                rule_id=rule.id,
                applicability=rule.applicability,
                instruction=rule.instruction,
                verifier=rule.verifier,
                tools=list(rule.scope.tools),
                paths=list(rule.scope.paths),
            )
        )
    return hooks


class RuleRuntime:
    """Evaluate hooks against a transcript using the shared verifier agent."""

    def __init__(self, gateway: LlmGateway, config: RunConfig) -> None:
        """Bind the runtime to a gateway and the active run configuration."""
        self._gateway = gateway
        self._config = config
        self._role = config.role(RoleName.RULE_VERIFIER)

    async def evaluate(
        self,
        hooks: Sequence[RuntimeHook],
        *,
        episode_id: str,
        transcript: str,
        friction_summary: str,
    ) -> list[HookResult]:
        """Evaluate each hook and return its verdict."""
        results: list[HookResult] = []
        for hook in hooks:
            verdict = await self._gateway.complete_structured(
                RoleName.RULE_VERIFIER,
                self._role,
                _RULE_VERIFIER_PROMPT,
                {
                    "applicability": hook.applicability,
                    "instruction": hook.instruction,
                    "verifier_spec": hook.verifier.spec,
                    "episode_id": episode_id,
                    "friction_summary": friction_summary,
                    "transcript": transcript,
                },
                RuleVerdict,
            )
            results.append(HookResult(rule_id=hook.rule_id, verdict=verdict))
        return results
