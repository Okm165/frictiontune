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


class DataSplits(DomainModel):
    """The train/eval partition produced by curation and decontamination."""

    train: list[TrainingExample] = Field(default_factory=list)
    eval: list[TrainingExample] = Field(default_factory=list)
    dropped_duplicates: int = 0
    dropped_contaminated: int = 0
    time_boundary: int | None = None
