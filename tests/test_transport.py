"""Tests for the transport-error taxonomy.

The retryable set must cover provider SDK errors and raw httpx transport
failures, and the predicate must additionally retry a 429/5xx that
``raise_for_status`` surfaces as ``HTTPStatusError`` while leaving other 4xx
permanent.
"""

from __future__ import annotations

import httpx
from frictiontune.llm.transport import is_transport_retryable


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://example.test")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_retryable_for_transport_and_connection_failures() -> None:
    assert is_transport_retryable(httpx.ConnectError("dropped"))
    assert is_transport_retryable(ConnectionError("dropped"))
    assert is_transport_retryable(TimeoutError("slow"))


def test_retryable_for_rate_limit_and_server_errors() -> None:
    assert is_transport_retryable(_status_error(429))
    assert is_transport_retryable(_status_error(500))
    assert is_transport_retryable(_status_error(503))


def test_not_retryable_for_permanent_client_errors() -> None:
    assert not is_transport_retryable(_status_error(400))
    assert not is_transport_retryable(_status_error(401))
    assert not is_transport_retryable(_status_error(404))
