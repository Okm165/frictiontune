"""Tests for the shared fan-out plumbing.

A stage that fans out independent per-item LLM calls must treat one item's
failure as that item's outcome, not the batch's. ``gather_tolerant`` is the one
place that guarantees it, so these lock its ordering and the failure labelling
that every tolerant stage reports.
"""

from __future__ import annotations

from frictiontune.core.errors import LlmError, ValidationError
from frictiontune.stages.common import failure_reason, gather_tolerant


async def test_gather_tolerant_returns_results_in_order() -> None:
    async def double(item: int) -> int:
        return item * 2

    assert await gather_tolerant(double, [1, 2, 3]) == [2, 4, 6]


async def test_gather_tolerant_isolates_a_failing_item() -> None:
    async def sometimes(item: int) -> int:
        if item == 2:
            raise LlmError("boom")
        return item

    results = await gather_tolerant(sometimes, [1, 2, 3])

    assert results[0] == 1
    assert isinstance(results[1], LlmError)
    assert results[2] == 3


async def test_gather_tolerant_handles_an_empty_batch() -> None:
    async def never(_item: int) -> int:
        raise AssertionError("must not be called")

    assert await gather_tolerant(never, []) == []


def test_failure_reason_labels_the_two_operator_relevant_modes() -> None:
    assert failure_reason(ValidationError("RecoverySpan", 4, "inverted span")) == "schema_rejected"
    assert failure_reason(LlmError("transport down")) == "llm_failed"


def test_failure_reason_surfaces_anything_else_by_type() -> None:
    assert failure_reason(KeyError("x")) == "error:KeyError"
