"""Training examples and token statistics."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from frictiontune.core.json import JsonObject
from frictiontune.domain.trace import DomainModel, Message

Track = Literal["recovery", "success", "synthetic"]


class TrainingExample(DomainModel):
    """One model-ready example: masked context plus trained completion."""

    id: str
    source: str
    track: Track
    context: list[Message] = Field(default_factory=list)
    completion: list[Message] = Field(default_factory=list)
    tools: list[JsonObject] = Field(default_factory=list)
    meta: JsonObject = Field(default_factory=dict)
    judge_score: float | None = None
    kept: bool = True


class SFTRecord(DomainModel):
    """A tokenized, masked training record written to the SFT dataset."""

    id: str
    source: str
    track: Track
    input_ids: list[int] = Field(default_factory=list)
    labels: list[int] = Field(default_factory=list)
    meta: JsonObject = Field(default_factory=dict)


class PreferencePair(DomainModel):
    """One chosen/rejected training pair mined from a recovery episode.

    The rejected side is the assistant's *actual failing behaviour* — the turns
    the friction event's evidence points at, taken verbatim from the corpus —
    and the chosen side is the recovery the same session actually produced.
    Both hang off the same prompt: the pre-friction context. This is the
    negative half that positive-only SFT structurally cannot express (GAP-104),
    and it needs no rollouts: the pairs exist in artifacts the pipeline already
    produces.
    """

    id: str
    source: str
    prompt: list[Message]
    chosen: list[Message]
    rejected: list[Message]
    meta: JsonObject = Field(default_factory=dict)


class TokenStats(DomainModel):
    """Aggregate tokenization statistics for a built SFT set."""

    examples: int = 0
    total_tokens: int = 0
    masked_tokens: int = 0
    trained_tokens: int = 0
    truncated: int = 0
    max_tokens: int = 0
    min_tokens: int = 0
    mean_tokens: float = 0.0
    histogram: dict[str, int] = Field(default_factory=dict)

    def train_ratio(self) -> float:
        """Fraction of tokens that are actually trained on."""
        if self.total_tokens == 0:
            return 0.0
        return self.trained_tokens / self.total_tokens
