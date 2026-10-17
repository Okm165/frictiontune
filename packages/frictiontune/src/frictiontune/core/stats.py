"""Deterministic descriptive statistics.

These are plumbing, not judgement: they summarise numbers the pipeline already
produced. A seeded bootstrap gives a confidence interval without any external
dependency, and the same seed always yields the same interval.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean (0.0 for an empty sequence)."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolated percentile (``quantile`` in ``[0, 1]``)."""
    if not values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def sign_test_p(values: Sequence[float]) -> float:
    """Exact two-sided sign-test p for paired deltas (zeros discarded).

    The assumption-free paired test: under a null of no effect each non-zero
    delta is equally likely positive or negative, so ``p = 2 · P(X >= max(k,
    m-k))`` for ``X ~ Binomial(m, 0.5)``. It is weaker than the bootstrap when
    magnitudes carry information, and the only paired test whose floor is
    computable in advance — which is what makes it the right check that an
    experiment has enough episodes to claim anything at all.
    """
    signs = [1 if value > 0 else -1 for value in values if value != 0]
    m = len(signs)
    if m == 0:
        return 1.0
    k = sum(1 for sign in signs if sign > 0)
    tail = max(k, m - k)
    upper = sum(math.comb(m, i) for i in range(tail, m + 1))
    return min(1.0, 2.0 * float(upper) / float(2**m))


def min_attainable_p(n: int) -> float:
    """Smallest two-sided sign-test p reachable with ``n`` paired episodes.

    All ``n`` deltas agreeing in sign is the most extreme outcome, worth
    ``2 · (1/2)^n``. This is the power floor of the instrument: at n=4 the floor
    is 0.125, so no 0.05-level claim is attainable *no matter how large the
    effect*; six episodes are the minimum for a two-sided 0.05 claim.
    """
    if n <= 0:
        return 1.0
    return min(1.0, 2.0 ** (1 - n))


def bootstrap_ci(
    values: Sequence[float],
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    if not values:
        return (0.0, 0.0)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng = random.Random(seed)
    size = len(values)
    means: list[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        means.append(sum(sample) / size)
    tail = (1.0 - confidence) / 2.0
    return (percentile(means, tail), percentile(means, 1.0 - tail))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two vectors.

    A dimension mismatch is a misconfiguration (two embedding models, or a wrong
    ``dimensions`` setting), not a similarity of zero: it is raised rather than
    silently returning 0.0, which would quietly blank out all retrieval. A
    degenerate zero vector still returns 0.0.
    """
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        raise ValueError(f"cosine: dimension mismatch {len(left)} != {len(right)}")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)
