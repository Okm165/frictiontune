"""Tests for the single shared retry primitive."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import frictiontune
import pytest
from frictiontune.core.retry import RetryPolicy, _retry_after_seconds, with_retry
from tenacity import Future, RetryCallState, Retrying


class _FakeHttpError(Exception):
    """Mimic the OpenAI/httpx error shape: ``exc.response.headers`` and ``exc.body``."""

    def __init__(self, headers: dict[str, str], body: object = None) -> None:
        super().__init__("server error")
        self.response = type("_Response", (), {"headers": headers})()
        self.body = body


def _state(exc: BaseException) -> RetryCallState:
    """A retry call state carrying ``exc`` as its outcome, for wait-strategy tests."""
    state = RetryCallState(retry_object=Retrying(), fn=lambda: None, args=(), kwargs={})
    future = Future(attempt_number=1)
    future.set_exception(exc)
    state.outcome = future
    return state


def test_retry_after_seconds_reads_numeric_header() -> None:
    assert _retry_after_seconds(_FakeHttpError({"retry-after": "21"})) == 21.0


def test_retry_after_seconds_reads_the_error_body() -> None:
    # Cloudflare sends the delay in the JSON body, with no Retry-After header.
    exc = _FakeHttpError({}, body={"error_code": "origin_bad_gateway", "retry_after": 60})

    assert _retry_after_seconds(exc) == 60.0


def test_retry_after_seconds_prefers_the_header() -> None:
    exc = _FakeHttpError({"retry-after": "5"}, body={"retry_after": 60})

    assert _retry_after_seconds(exc) == 5.0


def test_retry_after_seconds_is_none_without_a_header() -> None:
    assert _retry_after_seconds(_FakeHttpError({})) is None
    assert _retry_after_seconds(ConnectionError("no response")) is None
    assert _retry_after_seconds(None) is None


def test_retry_after_seconds_parses_an_http_date() -> None:
    when = datetime.now(UTC) + timedelta(seconds=30)
    raw = format_datetime(when, usegmt=True)

    wait = _retry_after_seconds(_FakeHttpError({"retry-after": raw}))

    assert wait is not None
    assert 25.0 <= wait <= 31.0


def test_wait_honours_retry_after_over_backoff() -> None:
    # The backoff ceiling is 1s; the server says 30s. The server wins.
    policy = RetryPolicy(initial_delay_s=0.0, max_delay_s=1.0, jitter_s=0.0)

    assert policy.wait()(_state(_FakeHttpError({"retry-after": "30"}))) >= 30.0


def test_wait_falls_back_to_backoff_without_a_header() -> None:
    policy = RetryPolicy(initial_delay_s=0.5, max_delay_s=60.0, jitter_s=0.0)

    wait = policy.wait()(_state(ConnectionError("no header")))

    assert wait >= 0.5


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


def test_with_retry_serves_synchronous_callers() -> None:
    attempts = 0

    def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("transient")
        return "ok"

    result = with_retry(
        flaky,
        policy=RetryPolicy(max_attempts=4, initial_delay_s=0.0, jitter_s=0.0),
    )

    assert result == "ok"
    assert attempts == 3


def test_with_retry_sync_reraises_after_exhaustion() -> None:
    attempts = 0

    def always_fails() -> str:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("nope")

    with pytest.raises(TimeoutError):
        with_retry(always_fails, policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0))

    assert attempts == 3


def test_with_retry_accepts_an_exception_predicate() -> None:
    attempts = 0

    def predicate(exc: BaseException) -> bool:
        return isinstance(exc, ValueError)

    def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("transient")
        return "ok"

    result = with_retry(
        flaky,
        policy=RetryPolicy(max_attempts=4, initial_delay_s=0.0, jitter_s=0.0),
        retry_on=predicate,
    )

    assert result == "ok"
    assert attempts == 3


def test_with_retry_predicate_does_not_retry_unmatched() -> None:
    attempts = 0

    def predicate(exc: BaseException) -> bool:
        return isinstance(exc, ValueError)

    def fails() -> str:
        nonlocal attempts
        attempts += 1
        raise KeyError("permanent")

    with pytest.raises(KeyError):
        with_retry(
            fails,
            policy=RetryPolicy(max_attempts=4, initial_delay_s=0.0, jitter_s=0.0),
            retry_on=predicate,
        )

    assert attempts == 1


def test_tenacity_is_confined_to_the_retry_module() -> None:
    root = Path(frictiontune.__file__).parent

    offenders = [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*.py"))
        if path.name != "retry.py" and "tenacity" in path.read_text(encoding="utf-8")
    ]

    assert offenders == [], f"tenacity must only be imported by core/retry.py: {offenders}"
