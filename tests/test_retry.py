"""Tests for the single shared retry primitive."""

from __future__ import annotations

import pytest

from frictiontune.core.retry import RetryPolicy, with_retry


async def test_with_retry_returns_after_transient_failures() -> None:
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("transient")
        return "ok"

    result = await with_retry(
        flaky,
        policy=RetryPolicy(max_attempts=4, initial_delay_s=0.0, jitter_s=0.0),
    )

    assert result == "ok"
    assert attempts == 3


async def test_with_retry_reraises_after_exhaustion() -> None:
    attempts = 0

    async def always_fails() -> str:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("nope")

    with pytest.raises(TimeoutError):
        await with_retry(
            always_fails,
            policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0, jitter_s=0.0),
        )

    assert attempts == 3


async def test_with_retry_does_not_retry_non_retryable() -> None:
    attempts = 0

    async def wrong_type() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("programmer error")

    with pytest.raises(ValueError):
        await with_retry(wrong_type, policy=RetryPolicy(max_attempts=5, initial_delay_s=0.0))

    assert attempts == 1