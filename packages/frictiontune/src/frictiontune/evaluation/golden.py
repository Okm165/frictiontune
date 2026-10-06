"""Golden-episode evaluation for the friction analyst.

A small, hand-labelled set of episodes paired with the friction leaves a correct
analyst must report. Running it against a configured endpoint turns a prompt
change into a measurable precision/recall number instead of an opinion, so
prompt work can be justified rather than assumed.

The labels are synthetic and deliberately span the analyst's boundary cases
(friction versus healthy iteration). They are a calibration harness, not a
benchmark; grow the fixture as real failures are found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from frictiontune.config.models import RoleName
from frictiontune.core.errors import FrictionTuneError
from frictiontune.domain.friction import FrictionAnalysis
from frictiontune.stages.common import analyst_prompt_values

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from frictiontune.config.models import RunConfig
    from frictiontune.llm.gateway import LlmGateway

_PROMPT = "friction_analyst"
_FIXTURE_PACKAGE = "frictiontune.evaluation"
_FIXTURE_NAME = "fixtures/golden_episodes.jsonl"


class ExpectedEvent(BaseModel):
    """The friction a correct analyst must report for one episode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    leaf: str
    form: str
    cause: str


class GoldenEpisode(BaseModel):
    """One hand-labelled episode and its expected friction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    transcript: str
    expected: list[ExpectedEvent] = Field(default_factory=list)
    notes: str = ""


@dataclass(slots=True)
class EpisodeOutcome:
    """Per-episode prediction versus expectation, keyed by leaf."""

    episode_id: str
    predicted: dict[str, tuple[str, str]] = field(default_factory=dict)
    expected: dict[str, tuple[str, str]] = field(default_factory=dict)
    abstained: bool = False
    error: str | None = None


@dataclass(slots=True)
class GoldenReport:
    """Aggregate precision/recall of the analyst over the golden set."""

    outcomes: list[EpisodeOutcome]

    @property
    def scored(self) -> list[EpisodeOutcome]:
        """Outcomes that produced a prediction (no transport/validation error)."""
        return [outcome for outcome in self.outcomes if outcome.error is None]

    @property
    def true_positive(self) -> int:
        """Leaves predicted and expected."""
        return sum(len(set(o.predicted) & set(o.expected)) for o in self.scored)

    @property
    def false_positive(self) -> int:
        """Leaves predicted but not expected."""
        return sum(len(set(o.predicted) - set(o.expected)) for o in self.scored)

    @property
    def false_negative(self) -> int:
        """Leaves expected but not predicted."""
        return sum(len(set(o.expected) - set(o.predicted)) for o in self.scored)

    @property
    def axis_correct(self) -> int:
        """Matched leaves whose form and cause are also correct."""
        return sum(
            sum(
                1
                for leaf in set(o.predicted) & set(o.expected)
                if o.predicted[leaf] == o.expected[leaf]
            )
            for o in self.scored
        )

    @property
    def exact_matches(self) -> int:
        """Episodes whose predicted and expected leaf sets are equal."""
        return sum(1 for o in self.scored if set(o.predicted) == set(o.expected))

    @property
    def abstentions(self) -> int:
        """Episodes the analyst abstained on."""
        return sum(1 for o in self.scored if o.abstained)

    @property
    def errors(self) -> int:
        """Episodes that failed to produce a validated analysis."""
        return sum(1 for o in self.outcomes if o.error is not None)

    def precision(self) -> float:
        """True positives over all predicted positives."""
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    def recall(self) -> float:
        """True positives over all expected positives."""
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        precision, recall = self.precision(), self.recall()
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def summary(self) -> str:
        """A compact, human-readable report."""
        total = len(self.scored)
        lines = [
            f"friction analyst over {total} golden episode(s)",
            f"  precision {self.precision():.2f}  recall {self.recall():.2f}  f1 {self.f1():.2f}",
            f"  exact-episode match {self.exact_matches}/{total}",
            f"  axis (form/cause) accuracy {self.axis_correct}/{self.true_positive}",
            f"  abstentions {self.abstentions}  errors {self.errors}",
        ]
        misses = [
            o.episode_id for o in self.scored if set(o.predicted) != set(o.expected) or o.abstained
        ]
        if misses:
            lines.append(f"  misses: {', '.join(misses)}")
        for outcome in self.outcomes:
            if outcome.error is not None:
                lines.append(f"  ! {outcome.episode_id}: {outcome.error}")
        return "\n".join(lines)


def _parse_golden(text: str) -> list[GoldenEpisode]:
    """Parse JSONL golden text, skipping blank lines and ``#`` comments."""
    episodes: list[GoldenEpisode] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        episodes.append(GoldenEpisode.model_validate_json(stripped))
    return episodes


def load_golden(path: Path) -> list[GoldenEpisode]:
    """Load a golden fixture from a JSONL file."""
    return _parse_golden(path.read_text(encoding="utf-8"))


def load_bundled_golden() -> list[GoldenEpisode]:
    """Load the golden fixture shipped inside the package."""
    resource = files(_FIXTURE_PACKAGE).joinpath(_FIXTURE_NAME)
    return _parse_golden(resource.read_text(encoding="utf-8"))


async def evaluate_golden(
    gateway: LlmGateway,
    config: RunConfig,
    episodes: Sequence[GoldenEpisode],
    *,
    max_transcript_chars: int = 100_000,
) -> GoldenReport:
    """Run the friction analyst on every golden episode and score the result."""
    role = config.role(RoleName.FRICTION_ANALYST)
    outcomes: list[EpisodeOutcome] = []
    for episode in episodes:
        values = analyst_prompt_values(
            config,
            session_id="golden",
            episode_id=episode.id,
            turn_count=len(episode.transcript.splitlines()),
            tool_calls=0,
            tool_errors=0,
            compactions=0,
            transcript=episode.transcript,
            max_chars=max_transcript_chars,
        )
        try:
            analysis = await gateway.complete_structured(
                RoleName.FRICTION_ANALYST, role, _PROMPT, values, FrictionAnalysis
            )
        except FrictionTuneError as exc:
            outcomes.append(EpisodeOutcome(episode.id, error=str(exc)))
            continue
        predicted = {event.leaf: (event.form, event.cause) for event in analysis.events}
        expected = {event.leaf: (event.form, event.cause) for event in episode.expected}
        outcomes.append(
            EpisodeOutcome(
                episode.id, predicted=predicted, expected=expected, abstained=analysis.abstain
            )
        )
    return GoldenReport(outcomes)
