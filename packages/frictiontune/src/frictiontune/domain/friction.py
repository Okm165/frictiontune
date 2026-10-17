"""Friction events: the signal the whole pipeline is built around."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from frictiontune.domain.trace import DomainModel
from frictiontune.domain.types import NonNegativeInt, Probability

Resolution = Literal["user-correction", "self-recovery", "none"]
RecoveryTrigger = Literal["self", "user-correction", "none"]


class EvidenceSpan(DomainModel):
    """A verbatim span that justifies a friction claim."""

    turn: NonNegativeInt
    quote: str

    @field_validator("quote")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence quote must not be empty")
        return value


class FrictionEventDraft(DomainModel):
    """Agent response: one friction occurrence before axes and id are derived.

    The analyst chooses the ``leaf`` — the one semantic judgment — plus cost,
    resolution, evidence, and confidence. ``form`` and ``cause`` are *not* asked
    for: each leaf fixes them in the taxonomy, so the plumbing derives them from
    the leaf rather than trusting (and then rejecting) a model restatement of a
    declared mapping.
    """

    leaf: str
    cost: str
    resolution: Resolution
    summary: str = ""
    evidence: list[EvidenceSpan] = Field(min_length=1)
    confidence: Probability


class FrictionEvent(FrictionEventDraft):
    """A persisted, identified friction occurrence on the SWE-chat axes."""

    id: str
    episode_id: str
    ensemble_votes: int = 1
    form: str
    cause: str


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
    # Episodes the analyst could not be judged on at all — every ensemble pass
    # failed, so the vote could not be held. They are counted here rather than
    # folded into `episodes_analysed`, so coverage stays a statement about the
    # episodes that *were* analysed and a bad run cannot look like a clean one.
    unanalysed: int = 0

    def coverage(self) -> float:
        """Fraction of analysed episodes that contained at least one event."""
        if self.episodes_analysed == 0:
            return 0.0
        return self.episodes_with_friction / self.episodes_analysed


class RecoverySpan(DomainModel):
    """Agent response: the recovery span within a friction episode.

    ``recovery_start`` and ``recovery_end`` have no default so the generated
    JSON schema marks them *required*: the model must always emit both keys (a
    nullable value is allowed when ``found`` is false). Leaving them optional
    let the endpoint omit ``recovery_end`` on a single-turn recovery, which
    silently discarded every candidate downstream. The validator makes an
    omitted span a schema-repair error instead of a silent skip.
    """

    found: bool
    recovery_start: NonNegativeInt | None
    recovery_end: NonNegativeInt | None
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

    @model_validator(mode="after")
    def _span_present_when_found(self) -> RecoverySpan:
        """Require a concrete span whenever the agent claims a recovery."""
        if self.found and (self.recovery_start is None or self.recovery_end is None):
            raise ValueError("recovery_start and recovery_end are required when found is true")
        return self
