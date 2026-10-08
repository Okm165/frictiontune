"""The provider transport-error taxonomy, shared by every HTTP/LLM call.

:data:`frictiontune.core.retry.RETRYABLE` covers OS-level failures, but the two
HTTP stacks used here do not raise those:

* the OpenAI SDK has its own hierarchy — ``APIConnectionError`` (which
  ``APITimeoutError`` extends), ``RateLimitError``, and ``InternalServerError``
  are not ``OSError`` subclasses;
* raw ``httpx`` callers raise ``httpx.TransportError`` for connect, read, write,
  timeout, and protocol failures.

Without naming them, a dropped connection or a 429 would never be retried. This
is the single definition of a retryable transport failure, passed to the one
retry primitive by every HTTP caller in the workspace: the chat client, the
embedding backend, the agent rollout, base-weight downloads, and the GRPO judge.
Raw ``httpx`` callers use :func:`is_transport_retryable`, which also retries a
429/5xx surfaced as ``HTTPStatusError`` by ``raise_for_status``.
"""

from __future__ import annotations

from httpx import HTTPStatusError, TransportError
from openai import APIConnectionError, InternalServerError, RateLimitError

from frictiontune.core.retry import RETRYABLE

TRANSPORT_RETRYABLE: tuple[type[BaseException], ...] = (
    *RETRYABLE,
    TransportError,  # raw httpx: connect/read/write/timeout/protocol failures
    APIConnectionError,  # OpenAI SDK: connection drops and timeouts
    RateLimitError,  # OpenAI SDK: HTTP 429
    InternalServerError,  # OpenAI SDK: HTTP 5xx
)


def is_transport_retryable(exc: BaseException) -> bool:
    """Whether ``exc`` is worth retrying.

    Covers the :data:`TRANSPORT_RETRYABLE` types plus a raw ``httpx`` response
    whose ``raise_for_status`` produced an :class:`~httpx.HTTPStatusError`:
    a 429 or any 5xx is transient, while other 4xx are permanent and are not
    retried. Raw httpx callers pass this predicate to :func:`with_retry` instead
    of the type tuple.
    """
    if isinstance(exc, TRANSPORT_RETRYABLE):
        return True
    if isinstance(exc, HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False
