"""The multi-signal replay dashboard ("friction drop").

A single number is easy to game; a dashboard of independent signals is not. Each
:class:`Signal` compares the base model with the tuned model over the same
replayed episodes, with a seeded bootstrap interval so a small or noisy
difference is visible as such.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from frictiontune.core.stats import bootstrap_ci, mean

if TYPE_CHECKING:
    from collections.abc import Sequence

Direction = Literal["lower_better", "higher_better"]

PRIMARY_SIGNAL = "friction_avoidance"


class Signal(BaseModel):
    """One comparable metric across the baseline and the tuned model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    baseline: float
    tuned: float
    delta: float
    ci_low: float
    ci_high: float
    direction: Direction
    n: int
    improved: bool
    significant: bool = False


class ReplayDashboard(BaseModel):
    """The finish line: all replay signals plus the headline friction drop."""

    model_config = ConfigDict(extra="forbid")

    signals: list[Signal] = Field(default_factory=list)
    episodes: int = 0
    seeds: list[int] = Field(default_factory=list)
    friction_drop: float = 0.0
    friction_drop_ci_low: float = 0.0
    friction_drop_ci_high: float = 0.0
    # None when no lucky-pass measurement was produced (the core replay cannot
    # compute it; only the gated user-simulation extension can). It is omitted
    # rather than fabricated as 0.0, which would read as a real measurement.
    lucky_pass_rate: float | None = None
    notes: list[str] = Field(default_factory=list)

    def signal(self, name: str) -> Signal | None:
        """Return the signal called ``name`` if present."""
        for item in self.signals:
            if item.name == name:
                return item
        return None

    def headline(self) -> str:
        """A one-line human summary of the friction drop."""
        pct = self.friction_drop * 100.0
        return (
            f"friction drop {pct:.1f}% "
            f"[{self.friction_drop_ci_low * 100:.1f}, {self.friction_drop_ci_high * 100:.1f}] "
            f"over {self.episodes} episodes"
        )


def _improved(delta: float, direction: Direction) -> bool:
    return delta < 0 if direction == "lower_better" else delta > 0


def make_signal(
    name: str,
    baseline_values: Sequence[float],
    tuned_values: Sequence[float],
    *,
    direction: Direction,
    seed: int = 0,
) -> Signal:
    """Build a comparable signal from paired baseline/tuned observations."""
    if len(baseline_values) != len(tuned_values):
        raise ValueError(f"signal {name!r}: paired sequences differ in length")
    deltas = [t - b for b, t in zip(baseline_values, tuned_values, strict=True)]
    ci_low, ci_high = bootstrap_ci(deltas, seed=seed)
    delta = mean(deltas)
    # A movement is only significant when the bootstrap interval for the paired
    # deltas excludes zero; `improved` alone just reads the sign of the mean. A
    # single pair has a degenerate interval [d, d] and cannot support a claim of
    # significance, so at least two pairs are required.
    significant = len(deltas) >= 2 and (
        (ci_low > 0.0 and ci_high > 0.0) or (ci_low < 0.0 and ci_high < 0.0)
    )
    return Signal(
        name=name,
        baseline=mean(baseline_values),
        tuned=mean(tuned_values),
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        direction=direction,
        n=len(deltas),
        improved=_improved(delta, direction),
        significant=significant,
    )


def relative_drop(baseline: float, tuned: float) -> float:
    """Relative reduction from ``baseline`` to ``tuned`` (0.0 if undefined)."""
    if baseline == 0:
        return 0.0
    return (baseline - tuned) / baseline


def build_dashboard(
    signals: Sequence[Signal],
    *,
    episodes: int,
    seeds: Sequence[int],
    lucky_pass_rate: float | None = None,
    notes: Sequence[str] = (),
) -> ReplayDashboard:
    """Assemble a dashboard and derive the headline friction drop.

    The headline uses the primary signal's relative reduction and scales its
    bootstrap interval by the same ratio, which keeps the interval interpretable
    even though it is derived rather than bootstrapped directly.
    """
    primary = next((item for item in signals if item.name == PRIMARY_SIGNAL), None)
    friction_drop = 0.0
    ci_low = 0.0
    ci_high = 0.0
    if primary is not None and primary.baseline != 0:
        scale = 1.0 / primary.baseline
        if primary.direction == "lower_better":
            friction_drop = relative_drop(primary.baseline, primary.tuned)
            ci_low = -primary.ci_high * scale
            ci_high = -primary.ci_low * scale
        else:
            friction_drop = (primary.tuned - primary.baseline) / primary.baseline
            ci_low = primary.ci_low * scale
            ci_high = primary.ci_high * scale
    return ReplayDashboard(
        signals=list(signals),
        episodes=episodes,
        seeds=list(seeds),
        friction_drop=friction_drop,
        friction_drop_ci_low=ci_low,
        friction_drop_ci_high=ci_high,
        lucky_pass_rate=lucky_pass_rate,
        notes=list(notes),
    )
