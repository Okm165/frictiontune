"""Friction events: the signal the whole pipeline is built around."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from frictiontune.domain.trace import DomainModel

Resolution = Literal["user-correction", "self-recovery", "none"]
RecoveryTrigger = Literal["self", "user-correction", "none"]


class EvidenceSpan(DomainModel):
    """A verbatim span that justifies a friction claim."""

    turn: int = Field(ge=0)
    quote: str

    @field_validator("quote")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence quote must not be empty")
        return value


class FrictionEventDraft(DomainModel):
    """Agent response: one friction occurrence before id assignment."""

    form: str
    cause: str
    cost: str
    resolution: Resolution
    leaf: str
    summary: str = ""
    evidence: list[EvidenceSpan] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class FrictionEvent(FrictionEventDraft):
    """A persisted, identified friction occurrence on the SWE-chat axes."""

    id: str
    episode_id: str
    ensemble_votes: int = 1


class FrictionAnalysis(DomainModel):
    """Agent response: the friction events found in one episode."""

    events: list[FrictionEventDraft] = Field(default_factory=list)
    abstain: bool = False
    notes: str = ""


class FrictionProfile(DomainModel):
    """Aggregate view of friction across episodes."""

    total_events: int = 0
    episodes_analysed: int = 0
    episodes_with_friction: int = 0
    by_form: dict[str, int] = Field(default_factory=dict)
    by_cause: dict[str, int] = Field(default_factory=dict)
    by_leaf: dict[str, int] = Field(default_factory=dict)
    by_resolution: dict[str, int] = Field(default_factory=dict)
    abstentions: int = 0

    def coverage(self) -> float:
        """Fraction of analysed episodes that contained at least one event."""
        if self.episodes_analysed == 0:
            return 0.0
        return self.episodes_with_friction / self.episodes_analysed


class RecoverySpan(DomainModel):
    """Agent response: the recovery span within a friction episode."""

    found: bool
    failure_turn: int | None = None
    recovery_start: int | None = None
    recovery_end: int | None = None
    trigger: RecoveryTrigger = "none"
    rationale: str = ""

    @field_validator("recovery_end")
    @classmethod
    def _end_after_start(cls, value: int | None, info: object) -> int | None:
        data = getattr(info, "data", {})
        start = data.get("recovery_start")
        if value is not None and start is not None and value < start:
            raise ValueError("recovery_end must be >= recovery_start")
        return value
