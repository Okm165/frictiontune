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
from frictiontune.domain.types import NonNegativeInt, Probability, Score


class CompactedToolOutput(DomainModel):
    """normalize_compaction response.

    The agent names the 1-based line numbers of a tool output that are safe to
    remove; the plumbing copies every other line verbatim and groups the numbers
    into ranges. Nothing is regenerated, so a failure line the agent left alone
    survives byte for byte, and the reply stays small however long the input is.

    A flat list of line numbers rather than ``{start_line, end_line}`` objects:
    on a local model the agent writes one entry per line regardless, so a range
    object per line cost ~10x the tokens for the same decision — the run that
    emitted 431 single-line ranges generated 18 KB, and a later one 24 K tokens,
    to say "remove lines 439-869". Naming lines keeps the reply as small as the
    decision; grouping lines into ranges is a set union, which is plumbing.

    There is deliberately no free-text ``rationale``: nothing read it, and on a
    local model it was the bulk of every reply — generation, not any decision,
    set the pace.
    """

    elide: list[NonNegativeInt] = Field(default_factory=list)
    preserved_failure: bool = False


class InferredToolSchema(DomainModel):
    """schema_inference response."""

    name: str
    parameters: JsonObject = Field(default_factory=dict)
    rationale: str = ""


class RedactionReview(DomainModel):
    """redaction_review response.

    The reviewer names only the candidates that **must** be redacted, by their
    ``index``; every candidate it does not name is kept as evidence. That is the
    complement of a per-candidate verdict list, and it is the whole point. On this
    corpus ~11% of flagged spans are secrets, so a reviewer made to annotate
    *every* candidate writes several times the text it needs — and a local model
    told "one decision per candidate, in order" does not stop at the batch: a
    live reply enumerated 552 verdicts for a 64-candidate batch and exhausted the
    output cap, which aborted the whole stage. A reply that names only the secrets
    cannot run away like that. The plumbing's fallback when a call fails stays
    conservative (redact every candidate in that batch), so an omitted index means
    "the reviewer judged it benign", never an accidental leak.
    """

    redact: list[NonNegativeInt] = Field(default_factory=list)


class JudgeVerdict(DomainModel):
    """judge response."""

    # `reason` is declared first so it is emitted before the decision fields:
    # writing the rationale first makes the score and accept decision deliberate
    # rather than post-hoc.
    reason: str = ""
    task_adherence: Score
    correctness: Score
    friction_avoidance: Score
    clarity: Score
    evidence: Score
    accept: bool

    def total(self) -> int:
        """Sum of the rubric scores."""
        return (
            self.task_adherence
            + self.correctness
            + self.friction_avoidance
            + self.clarity
            + self.evidence
        )


class ReplayVerdict(DomainModel):
    """judge response for one replay continuation.

    A *comparative* instrument, deliberately not :class:`JudgeVerdict`. The
    filter's rubric asks "is this good enough to train on?", so a capable model
    sits at its ceiling — the first end-to-end replay scored 5.00 on every axis
    for both the base model and the adapter, leaving `friction_drop` zero by
    construction. The axes here are exactly the ones the replay turns into
    signals, and nothing else: `task_adherence` and `evidence` were never read,
    and `accept` is a filter's output with no meaning in a comparison. Fields a
    stage converts back into nothing are ceremony, and ceremony is a place for
    the model to run away.
    """

    # `reason` first, for the same reason as the filter: a rationale written
    # before the scores makes the scores deliberate rather than post-hoc.
    reason: str = ""
    friction_avoidance: Score
    correctness: Score
    clarity: Score


class DedupVerdict(DomainModel):
    """dedup response."""

    reason: str = ""
    duplicate: bool
    similarity: Probability
    keep: Literal["a", "b", "either", "neither"] = "either"


class HarvestedSpan(DomainModel):
    """One focused run of assistant turns worth imitating.

    ``start`` and ``end`` are inclusive turn indices into the episode. The run is
    deliberately short — a behaviour, not a stretch of work — because the
    completion is what the model is trained to produce, and a twenty-message
    completion teaches "continue what you were doing" rather than the behaviour
    that made this stretch worth keeping.
    """

    start: NonNegativeInt
    end: NonNegativeInt
    behaviour: str


class HarvestedWork(DomainModel):
    """Agent response: the focused good work an episode contains.

    The complement of ``RecoverySpan``: instead of the repair after a failure,
    this is the assistant avoiding the failure in the first place. An empty list
    is a valid answer — most episodes contain none, and inventing one teaches
    noise.
    """

    spans: list[HarvestedSpan] = Field(default_factory=list)
    notes: str = ""


class RuleVerdict(DomainModel):
    """rule_verifier response."""

    # `rationale` is declared first so the model reasons before it judges.
    rationale: str = ""
    fires: bool
    prevents: bool
    collateral: bool
    confidence: Probability
    evidence_turns: list[NonNegativeInt] = Field(default_factory=list)


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

    # `reason` first: reason before committing to contamination or cleanliness.
    reason: str = ""
    contaminated: bool
    tier: Literal["t3_semantic", "t4_task"] = "t3_semantic"


class ReduceSummary(DomainModel):
    """reduce response: a faithful, decision-preserving summary of a span."""

    summary: str
    preserved_failures: NonNegativeInt = 0
    dropped_turns: NonNegativeInt = 0
    rationale: str = ""
