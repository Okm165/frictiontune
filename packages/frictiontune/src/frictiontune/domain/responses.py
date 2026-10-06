"""Typed contracts for LLM/agent responses.

Every model here is a strict pydantic schema returned by exactly one prompt.
Keeping them together makes the agent I/O surface auditable in one place.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from frictiontune.core.json import JsonObject
from frictiontune.domain.profiles import BehaviorDimension
from frictiontune.domain.trace import DomainModel


class CompactedToolOutput(DomainModel):
    """normalize_compaction response."""

    kept: str
    preserved_failure: bool = False
    elided_lines: int = 0
    rationale: str = ""


class InferredToolSchema(DomainModel):
    """schema_inference response."""

    name: str
    parameters: JsonObject = Field(default_factory=dict)
    rationale: str = ""


class RedactionDecision(DomainModel):
    """One decision inside a :class:`RedactionReview`."""

    index: int = Field(ge=0)
    redact: bool
    reason: str = ""


class RedactionReview(DomainModel):
    """redaction_review response."""

    decisions: list[RedactionDecision] = Field(default_factory=list)


class JudgeVerdict(DomainModel):
    """judge response."""

    task_adherence: int = Field(ge=0, le=5)
    correctness: int = Field(ge=0, le=5)
    friction_avoidance: int = Field(ge=0, le=5)
    clarity: int = Field(ge=0, le=5)
    evidence: int = Field(ge=0, le=5)
    accept: bool
    reason: str = ""

    def total(self) -> int:
        """Sum of the rubric scores."""
        return (
            self.task_adherence
            + self.correctness
            + self.friction_avoidance
            + self.clarity
            + self.evidence
        )


class DedupVerdict(DomainModel):
    """dedup response."""

    duplicate: bool
    similarity: float = Field(ge=0.0, le=1.0)
    keep: Literal["a", "b", "either", "neither"] = "either"
    reason: str = ""


class RuleVerdict(DomainModel):
    """rule_verifier response."""

    fires: bool
    prevents: bool
    collateral: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_turns: list[int] = Field(default_factory=list)
    rationale: str = ""


class FrictionClusters(DomainModel):
    """friction_cluster response."""

    clusters: list[BehaviorDimension] = Field(default_factory=list)


class SyntheticExample(DomainModel):
    """One generated example inside a :class:`SyntheticBatch`."""

    context: str
    completion: str
    rationale: str = ""


class SyntheticBatch(DomainModel):
    """synthesize response."""

    examples: list[SyntheticExample] = Field(default_factory=list)


class SimulatedUserTurn(DomainModel):
    """user_simulator response."""

    message: str
    resolved: bool = False
    reason: str = ""


class ContaminationVerdict(DomainModel):
    """decontaminate response: whether an eval example leaks into train."""

    contaminated: bool
    tier: Literal["t3_semantic", "t4_task"] = "t3_semantic"
    reason: str = ""


class ReduceSummary(DomainModel):
    """reduce response: a faithful, decision-preserving summary of a span."""

    summary: str
    preserved_failures: int = 0
    dropped_turns: int = 0
    rationale: str = ""
