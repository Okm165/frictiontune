"""Centralised retry policy.

Every network call (LLM, embeddings, downloads) goes through :func:`with_retry`.
There is exactly one retry implementation in the codebase; stages never roll
their own loop.

The primitive serves synchronous and asynchronous callers alike, so one
definition covers every call whatever its execution model. Pass an ``async def``
for a coroutine and a plain function otherwise; the matching tenacity loop is
used internally. (A bare ``lambda`` that merely *returns* an awaitable is treated
as synchronous — write an ``async def`` when the work is asynchronous.)
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any, TypeVar, overload

from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    # A retry selector: a tuple of exception types, or a predicate over the
    # exception for cases (HTTP status codes) a type tuple cannot express.
    RetryOn = tuple[type[BaseException], ...] | Callable[[BaseException], bool]

T = TypeVar("T")

# Transient OS-level transport failures. Provider SDKs raise their *own* error
# types, which are not OSError subclasses; those extend this set at the transport
# layer (see ``frictiontune.llm.transport.TRANSPORT_RETRYABLE``). Domain and
# validation errors are handled elsewhere.
RETRYABLE: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def _retry_after_seconds(exc: BaseException | None) -> float | None:
    """Return the server's requested wait, from a ``Retry-After`` header.

    A 429 is not just "try again soon": the server states how long to wait, and
    a fixed exponential backoff that ignores it retries too early and exhausts
    its attempts inside one rate-limit window.

    Two shapes are honoured, both duck-typed so this stays dependency-free in
    ``core``: a ``Retry-After`` header (seconds or an HTTP date) on
    ``exc.response.headers``, and a ``retry_after`` field in the error body on
    ``exc.body`` — some gateways (Cloudflare, seen at the Bitdeer endpoint) put
    the delay in the JSON body only.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        delay = _parse_retry_after(headers.get("retry-after"))
        if delay is not None:
            return delay
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return _parse_retry_after(body.get("retry_after"))
    return None


def _parse_retry_after(raw: object) -> float | None:
    """Parse a ``Retry-After`` value: seconds, an HTTP date, or ``None``."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    if isinstance(raw, (int, float)):
        return max(0.0, float(raw))
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


class RetryPolicy(BaseModel):
    """Declarative retry configuration shared by all callers."""

    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=5, ge=1)
    initial_delay_s: float = Field(default=0.5, ge=0.0)
    max_delay_s: float = Field(default=60.0, ge=0.0)
    multiplier: float = Field(default=2.0, ge=1.0)
    jitter_s: float = Field(default=0.25, ge=0.0)

    def wait(self) -> Callable[[RetryCallState], float]:
        """Build the wait strategy: exponential backoff, floored by ``Retry-After``.

        The server's requested delay always wins when it is longer, because the
        backoff exists to survive transient failures, not to overrule the party
        that knows when it is ready to serve again.
        """
        exponential = wait_exponential_jitter(
            initial=self.initial_delay_s,
            max=self.max_delay_s,
            exp_base=self.multiplier,
            jitter=self.jitter_s,
        )

        def strategy(retry_state: RetryCallState) -> float:
            server_wait = _retry_after_seconds(
                retry_state.outcome.exception() if retry_state.outcome else None
            )
            backoff = exponential(retry_state)
            return max(backoff, server_wait) if server_wait is not None else backoff

        return strategy

    @classmethod
    def immediate(cls, attempts: int) -> RetryPolicy:
        """A policy that retries at once, for cheap in-process work.

        Schema repair re-asks a model that has just failed validation; a backoff
        would only add latency, so every retry is immediate.
        """
        return cls(
            max_attempts=attempts,
            initial_delay_s=0.0,
            max_delay_s=0.0,
            jitter_s=0.0,
        )


def _log_retry(*, logger: object | None, retry_state: RetryCallState) -> None:
    if logger is None:
        return
    log = getattr(logger, "warning", None)
    if callable(log):
        log(
            "retrying call",
            attempt=retry_state.attempt_number,
            exception=str(retry_state.outcome.exception()) if retry_state.outcome else None,
        )


def _attempt_kwargs(
    active: RetryPolicy,
    retry_on: RetryOn,
    logger: object | None,
) -> dict[str, Any]:
    """The tenacity configuration shared by the sync and async paths."""
    return {
        "stop": stop_after_attempt(active.max_attempts),
        "wait": active.wait(),
        "retry": retry_if_exception(retry_on)
        if callable(retry_on)
        else retry_if_exception_type(retry_on),
        "reraise": True,
        "before_sleep": lambda state: _log_retry(logger=logger, retry_state=state),
    }


async def _retry_async(
    operation: Callable[[], Awaitable[T]],
    active: RetryPolicy,
    retry_on: RetryOn,
    logger: object | None,
) -> T:
    async for attempt in AsyncRetrying(**_attempt_kwargs(active, retry_on, logger)):
        with attempt:
            return await operation()
    raise AssertionError("unreachable: tenacity always exits via return or raise")


def _retry_sync(
    operation: Callable[[], T],
    active: RetryPolicy,
    retry_on: RetryOn,
    logger: object | None,
) -> T:
    for attempt in Retrying(**_attempt_kwargs(active, retry_on, logger)):
        with attempt:
            return operation()
    raise AssertionError("unreachable: tenacity always exits via return or raise")


@overload
async def with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = ...,
    retry_on: RetryOn = ...,
    logger: object | None = ...,
) -> T: ...


@overload
def with_retry(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy | None = ...,
    retry_on: RetryOn = ...,
    logger: object | None = ...,
) -> T: ...


def with_retry(
    operation: Callable[[], Any],
    *,
    policy: RetryPolicy | None = None,
    retry_on: RetryOn = RETRYABLE,
    logger: object | None = None,
) -> Any:
    """Run ``operation`` with the shared retry policy.

    ``operation`` is a zero-argument factory so each attempt starts fresh
    (important for HTTP clients and generators). It may be synchronous or a
    coroutine function; the matching execution model is used, so the one retry
    definition serves every caller. ``retry_on`` is either a tuple of exception
    types or a predicate over the exception, for status-code-aware retries.
    """
    active = policy or RetryPolicy()
    if inspect.iscoroutinefunction(operation):
        return _retry_async(operation, active, retry_on, logger)
    return _retry_sync(operation, active, retry_on, logger)
