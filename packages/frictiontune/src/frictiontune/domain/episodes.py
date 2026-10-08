"""Episodes and phases: the units of friction, mining, and evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from frictiontune.domain.trace import DomainModel, Message
from frictiontune.domain.types import NonNegativeInt

PhaseKind = Literal["plan", "execute", "verify", "recover"]


class PhaseDraft(DomainModel):
    """Agent response: one phase before the stage assigns it an id."""

    kind: PhaseKind
    start_turn: NonNegativeInt
    end_turn: NonNegativeInt
    goal: str = ""


class Phase(PhaseDraft):
    """A persisted, identified span of an episode (plan/execute/verify/recover)."""

    id: str
    episode_id: str = ""


class EpisodeFeatures(DomainModel):
    """Descriptive, non-judgmental features used for reporting and budgets."""

    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    reasoning_turns: int = 0
    text_turns: int = 0
    compactions: int = 0
    char_count: int = 0
    user_turns_in_session: int = 0


class Episode(DomainModel):
    """Everything between one user message and the next."""

    id: str
    session_id: str
    start_user_message_id: str
    start_seq: int
    end_seq: int
    time_created: int = 0
    user_text: str = ""
    messages: list[Message] = Field(default_factory=list)
    phases: list[Phase] = Field(default_factory=list)
    features: EpisodeFeatures = Field(default_factory=EpisodeFeatures)

    def assistant_turns(self) -> list[Message]:
        """Return the assistant messages in order."""
        return [message for message in self.messages if message.role == "assistant"]


class EpisodePhases(DomainModel):
    """Agent response: the phases dividing one episode."""

    phases: list[PhaseDraft] = Field(default_factory=list)


class ReducedSpan(DomainModel):
    """A compressed span of an episode produced by the ``reduce`` extension."""

    episode_id: str
    start_turn: int
    end_turn: int
    summary: str
    preserved_failures: int = 0
    dropped_turns: int = 0


class RouterDecision(DomainModel):
    """Agent response: cheap triage of an episode."""

    # `reason` first: the model explains before it routes.
    reason: str = ""
    decision: Literal["clearly_clean", "needs_analysis", "ambiguous"]
