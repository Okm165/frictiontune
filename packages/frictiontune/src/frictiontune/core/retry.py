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
from typing import TYPE_CHECKING, Any, TypeVar, overload

from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

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


class RetryPolicy(BaseModel):
    """Declarative retry configuration shared by all callers."""

    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=4, ge=1)
    initial_delay_s: float = Field(default=0.5, ge=0.0)
    max_delay_s: float = Field(default=30.0, ge=0.0)
    multiplier: float = Field(default=2.0, ge=1.0)
    jitter_s: float = Field(default=0.25, ge=0.0)

    def wait(self) -> wait_exponential_jitter:
        """Build the tenacity wait strategy for this policy."""
        return wait_exponential_jitter(
            initial=self.initial_delay_s,
            max=self.max_delay_s,
            exp_base=self.multiplier,
            jitter=self.jitter_s,
        )

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
    retry_on: tuple[type[BaseException], ...],
    logger: object | None,
) -> dict[str, Any]:
    """The tenacity configuration shared by the sync and async paths."""
    return {
        "stop": stop_after_attempt(active.max_attempts),
        "wait": active.wait(),
        "retry": retry_if_exception_type(retry_on),
        "reraise": True,
        "before_sleep": lambda state: _log_retry(logger=logger, retry_state=state),
    }


async def _retry_async(
    operation: Callable[[], Awaitable[T]],
    active: RetryPolicy,
    retry_on: tuple[type[BaseException], ...],
    logger: object | None,
) -> T:
    async for attempt in AsyncRetrying(**_attempt_kwargs(active, retry_on, logger)):
        with attempt:
            return await operation()
    raise AssertionError("unreachable: tenacity always exits via return or raise")


def _retry_sync(
    operation: Callable[[], T],
    active: RetryPolicy,
    retry_on: tuple[type[BaseException], ...],
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
    retry_on: tuple[type[BaseException], ...] = ...,
    logger: object | None = ...,
) -> T: ...


@overload
def with_retry(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy | None = ...,
    retry_on: tuple[type[BaseException], ...] = ...,
    logger: object | None = ...,
) -> T: ...


def with_retry(
    operation: Callable[[], Any],
    *,
    policy: RetryPolicy | None = None,
    retry_on: tuple[type[BaseException], ...] = RETRYABLE,
    logger: object | None = None,
) -> Any:
    """Run ``operation`` with the shared retry policy.

    ``operation`` is a zero-argument factory so each attempt starts fresh
    (important for HTTP clients and generators). It may be synchronous or a
    coroutine function; the matching execution model is used, so the one retry
    definition serves every caller.
    """
    active = policy or RetryPolicy()
    if inspect.iscoroutinefunction(operation):
        return _retry_async(operation, active, retry_on, logger)
    return _retry_sync(operation, active, retry_on, logger)
