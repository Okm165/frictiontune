"""Runtime rules and their verifiers (the TRACE structure)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field

from frictiontune.domain.trace import DomainModel
from frictiontune.domain.types import enum_by_value


class VerifierTier(StrEnum):
    """How a rule is checked at runtime; all tiers are LLM/agent reasoning."""

    ARTIFACT_GROUNDED = "artifact_grounded"
    SEMANTIC = "semantic"
    INTENT = "intent"


class RuleLifecycle(StrEnum):
    """How a rule relates to existing rules on a re-run."""

    NEW = "new"
    UPDATE = "update"
    SUPERSEDE = "supersede"
    SPLIT = "split"
    NOOP = "noop"


class RuleStatus(StrEnum):
    """The stage of a rule in its lifecycle."""

    CANDIDATE = "candidate"
    VERIFIED = "verified"
    ACTIVE = "active"
    RETIRED = "retired"


class RuleScope(DomainModel):
    """Coarse cost hint for which rules are worth evaluating on a turn."""

    task_tags: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)


class RuleVerifier(DomainModel):
    """The runtime check for a rule."""

    # Strict validation keeps every other field exact; enums arrive by value.
    tier: Annotated[VerifierTier, enum_by_value(VerifierTier)]
    spec: str
    evidence: list[str] = Field(default_factory=list)
    failure_message: str = ""


class RuleDraft(DomainModel):
    """Agent response: one compiled rule before it is identified and verified."""

    applicability: str
    instruction: str
    scope: RuleScope = Field(default_factory=RuleScope)
    verifier: RuleVerifier
    rationale: str = ""


class RuntimeRule(RuleDraft):
    """One atomic, runtime-checkable rule."""

    id: str
    source_events: list[str] = Field(default_factory=list)
    lifecycle: RuleLifecycle = RuleLifecycle.NEW
    status: RuleStatus = RuleStatus.CANDIDATE
    verification: RuleVerification | None = None


class CompiledRules(DomainModel):
    """Agent response: the atomic rules compiled from observed friction."""

    rules: list[RuleDraft] = Field(default_factory=list)


class RuleVerification(DomainModel):
    """Aggregate evidence that a rule is safe and effective."""

    episodes_checked: int = 0
    fires: int = 0
    prevented: int = 0
    collateral: int = 0
    mean_confidence: float = 0.0

    def accepted(self, *, min_prevented: int, max_collateral: int = 0) -> bool:
        """Whether the rule passes the accept gate."""
        return self.prevented >= min_prevented and self.collateral <= max_collateral


class RuleLibrary(DomainModel):
    """The full set of rules and their status."""

    rules: list[RuntimeRule] = Field(default_factory=list)
    generated_from: str = ""


# Resolve the forward reference used above.
RuntimeRule.model_rebuild()
