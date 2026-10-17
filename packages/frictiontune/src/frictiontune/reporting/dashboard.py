"""The multi-signal replay dashboard ("friction drop").

A single number is easy to game; a dashboard of independent signals is not. Each
:class:`Signal` compares the base model with the tuned model over the same
replayed episodes, with a seeded bootstrap interval so a small or noisy
difference is visible as such.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from frictiontune.core.stats import bootstrap_ci, mean, min_attainable_p, sign_test_p

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
    # The exact sign-test p for the paired deltas, and the smallest p the
    # instrument could have produced with this many episodes. Together they
    # separate "the deltas align" from "this experiment could have detected
    # that": at n=4 the floor is 0.125, so `significant` is forced False even
    # when every delta agrees — the honest answer, not a pessimistic one.
    sign_test_p: float = 1.0
    min_attainable_p: float = 1.0
    powered: bool = False
    # The raw paired observations. n is small by construction; storing them
    # makes the judge's use of its scale visible (GAP-099: compression onto
    # {4,5} was invisible in means alone).
    base_scores: list[float] = Field(default_factory=list)
    tuned_scores: list[float] = Field(default_factory=list)
    # Episodes within one session are not independent (same task, tools and
    # context), so when cluster ids are supplied the primary power floor and
    # sign test are computed over cluster means, not raw episodes (GAP-097).
    clusters: int = 0
    cluster_sign_test_p: float | None = None
    cluster_min_attainable_p: float | None = None
    # The pre-registered minimal important difference (GAP-102): None when no
    # bar was declared; otherwise whether the measured delta reaches it. A
    # significant result below the bar is still "measured, below the bar".
    meets_mcid: bool | None = None


class ReplayDashboard(BaseModel):
    """The finish line: all replay signals plus the headline friction drop."""

    model_config = ConfigDict(extra="forbid")

    signals: list[Signal] = Field(default_factory=list)
    episodes: int = 0
    seeds: list[int] = Field(default_factory=list)
    friction_drop: float = 0.0
    friction_drop_ci_low: float = 0.0
    friction_drop_ci_high: float = 0.0
    # Instrument-level power floor over all replayed episodes: below six
    # episodes no two-sided 0.05 claim is attainable and every `significant`
    # flag is held False, whatever the deltas look like. When clusters are
    # supplied the floor is computed over clusters (GAP-097), which is lower
    # than the episode count on any clustered corpus.
    min_attainable_p: float = 1.0
    powered: bool = False
    clusters: int = 0
    # The declared minimal important difference, for the record; per-signal
    # verdicts live on the signals.
    min_important_delta: float | None = None
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
        text = (
            f"friction drop {pct:.1f}% "
            f"[{self.friction_drop_ci_low * 100:.1f}, {self.friction_drop_ci_high * 100:.1f}] "
            f"over {self.episodes} episodes"
        )
        if self.clusters and self.clusters < self.episodes:
            text += f" ({self.clusters} independent sessions)"
        if not self.powered:
            text += (
                f" — underpowered: n={self.episodes} cannot reach p<=0.05 "
                f"(floor {self.min_attainable_p:.3f}); no effect claim is possible"
            )
        primary = self.signal(PRIMARY_SIGNAL)
        if (
            self.min_important_delta is not None
            and primary is not None
            and primary.meets_mcid is False
        ):
            text += (
                f" — below the declared bar ({self.min_important_delta:g}); measured, not claimable"
            )
        return text


def _improved(delta: float, direction: Direction) -> bool:
    return delta < 0 if direction == "lower_better" else delta > 0


def make_signal(
    name: str,
    baseline_values: Sequence[float],
    tuned_values: Sequence[float],
    *,
    direction: Direction,
    seed: int = 0,
    cluster_ids: Sequence[str] | None = None,
    min_important_delta: float | None = None,
) -> Signal:
    """Build a comparable signal from paired baseline/tuned observations.

    ``cluster_ids`` labels the independent unit each observation belongs to
    (the session, for replayed episodes). When supplied, the *primary* power
    floor and sign test are computed over per-cluster mean deltas: episodes of
    one session share task, tools and context, so treating them as independent
    overstates the instrument's power (GAP-097).
    """
    if len(baseline_values) != len(tuned_values):
        raise ValueError(f"signal {name!r}: paired sequences differ in length")
    if cluster_ids is not None and len(cluster_ids) != len(baseline_values):
        raise ValueError(f"signal {name!r}: cluster ids differ in length from the pairs")
    deltas = [t - b for b, t in zip(baseline_values, tuned_values, strict=True)]
    ci_low, ci_high = bootstrap_ci(deltas, seed=seed)
    delta = mean(deltas)
    floor = min_attainable_p(len(deltas))
    powered = floor <= 0.05

    cluster_sign_p: float | None = None
    cluster_floor: float | None = None
    n_clusters = 0
    if cluster_ids is not None and len(set(cluster_ids)) < len(deltas):
        grouped: dict[str, list[float]] = {}
        for cluster, cluster_delta in zip(cluster_ids, deltas, strict=True):
            grouped.setdefault(cluster, []).append(cluster_delta)
        cluster_deltas = [mean(group) for group in grouped.values()]
        n_clusters = len(cluster_deltas)
        cluster_floor = min_attainable_p(n_clusters)
        cluster_sign_p = sign_test_p(cluster_deltas)
        # The clustered design is the honest one; power follows it when it is
        # the stricter of the two views.
        if cluster_floor > floor:
            powered = cluster_floor <= 0.05

    # Power first, significance second. The bootstrap interval is descriptive —
    # at n=4 it has only 35 distinct resamples and can exclude zero by luck — so
    # `significant` additionally requires the instrument to be *capable* of a
    # 0.05-level claim. Holding `significant` False below that floor is the
    # point: it stops a lucky interval from reading as a detection.
    significant = (
        powered
        and len(deltas) >= 2
        and ((ci_low > 0.0 and ci_high > 0.0) or (ci_low < 0.0 and ci_high < 0.0))
    )
    meets_mcid = (
        None
        if min_important_delta is None
        else abs(delta) >= min_important_delta and _improved(delta, direction)
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
        sign_test_p=sign_test_p(deltas),
        min_attainable_p=floor,
        powered=powered,
        base_scores=[float(value) for value in baseline_values],
        tuned_scores=[float(value) for value in tuned_values],
        clusters=n_clusters,
        cluster_sign_test_p=cluster_sign_p,
        cluster_min_attainable_p=cluster_floor,
        meets_mcid=meets_mcid,
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
    clusters: int = 0,
    min_important_delta: float | None = None,
) -> ReplayDashboard:
    """Assemble a dashboard and derive the headline friction drop.

    The headline uses the primary signal's relative reduction and scales its
    bootstrap interval by the same ratio, which keeps the interval interpretable
    even though it is derived rather than bootstrapped directly. ``clusters``
    is the number of independent units behind the observations; when it is
    lower than ``episodes`` the instrument-level power floor follows it
    (GAP-097).
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
    floor = min_attainable_p(clusters if clusters else episodes)
    return ReplayDashboard(
        signals=list(signals),
        episodes=episodes,
        seeds=list(seeds),
        friction_drop=friction_drop,
        friction_drop_ci_low=ci_low,
        friction_drop_ci_high=ci_high,
        min_attainable_p=floor,
        powered=floor <= 0.05,
        clusters=clusters,
        min_important_delta=min_important_delta,
        lucky_pass_rate=lucky_pass_rate,
        notes=list(notes),
    )
