"""Behaviour profiles: attribution, clusters, and the Preference Profile."""

from __future__ import annotations

from pydantic import Field

from frictiontune.domain.trace import DomainModel


class Attribution(DomainModel):
    """Agent response: the earliest decisive step causing a friction event."""

    decisive_turn: int | None = Field(default=None, ge=0)
    cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class EventAttribution(DomainModel):
    """A persisted attribution bound to its friction event."""

    event_id: str
    attribution: Attribution


class BehaviorDimension(DomainModel):
    """A reusable behaviour theme derived from friction events."""

    name: str
    dimension: str
    target_behavior: str
    event_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class PreferenceProfile(DomainModel):
    """The positive behaviours the agent should adopt, with coverage."""

    dimensions: list[BehaviorDimension] = Field(default_factory=list)
    total_events: int = 0
    covered_events: int = 0

    def coverage(self) -> float:
        """Fraction of friction events assigned to a behaviour dimension."""
        if self.total_events == 0:
            return 0.0
        return self.covered_events / self.total_events
