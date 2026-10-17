"""The single OpenAI-compatible LLM client.

All model access in frictiontune goes through :class:`LlmClient`. It owns:

* one ``AsyncOpenAI`` per ``(base_url, api_key)`` endpoint, and round-robin
  across several when a role names replicas,
* transport retries via the shared :func:`~frictiontune.core.retry.with_retry`,
* content-hash caching so identical requests never pay twice,
* per-role concurrency limiting.

It performs **no** prompt construction and **no** schema repair; those live in
:mod:`frictiontune.llm.prompts` and :mod:`frictiontune.llm.gateway`.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from frictiontune.core.errors import LlmError
from frictiontune.core.hashing import content_hash
from frictiontune.core.logging import get_logger
from frictiontune.core.retry import with_retry
from frictiontune.llm.transport import is_transport_retryable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from frictiontune.config.models import LlmRoleConfig, RoleName
    from frictiontune.core.cache import ContentCache
    from frictiontune.llm.messages import ChatMessage, ToolSpec

_log = get_logger(__name__)


class TokenUsage(BaseModel):
    """Token accounting returned by the provider."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LlmResponse(BaseModel):
    """A normalised chat-completion response."""

    model_config = ConfigDict(extra="forbid")

    text: str
    model: str
    finish_reason: str | None = None
    # The provider's private chain-of-thought, when it returns one. Kept rather
    # than discarded so the reasoning stream is auditable (and usable by a
    # reasoning-training policy); structured calls ignore it and read ``text``.
    reasoning: str = ""
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cached: bool = False


def _api_key(config: LlmRoleConfig) -> str:
    if config.api_key_env:
        return os.environ.get(config.api_key_env, "EMPTY")
    return "EMPTY"


def base_urls(config: LlmRoleConfig) -> tuple[str, ...]:
    """Split ``base_url`` into one or more endpoints.

    A single URL behaves exactly as before. A comma-separated list names
    *identical* replicas — the shape a data-parallel local server takes when the
    model fits one card and the box has several: one process per card, each on
    its own port. The client spreads calls across them round-robin, so no proxy
    sits between the pipeline and the models. The endpoints must serve the same
    model and sampling knobs; the cache key covers the whole list, so a change
    to the replica set invalidates it the same way a change to one URL would.
    """
    urls = tuple(part.strip().rstrip("/") for part in (config.base_url or "").split(","))
    return tuple(url for url in urls if url) or ("",)


def _message_text(message: Any) -> str:
    """Extract text from a completion message, tolerating content-part lists.

    OpenAI-compatible servers return ``content`` as a string, but some return a
    list of content parts. Concatenating the text parts keeps a real reply from
    being rejected as a non-string, and anything unrecognised degrades to ``""``
    so the gateway's repair loop can re-ask rather than crashing on a type error.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "\n".join(chunks)
    return ""


def _reasoning_text(message: Any) -> str:
    """Extract a provider's ``reasoning_content`` stream, or ``""``.

    Reasoning-capable providers return chain-of-thought alongside ``content``
    under a vendor key (``reasoning_content`` for vLLM/SGLang and the zai/GLM
    family). It is captured for auditing but never treated as the answer.
    """
    reasoning = getattr(message, "reasoning_content", None)
    return reasoning if isinstance(reasoning, str) else ""


def build_response_format(
    config: LlmRoleConfig,
    *,
    schema: dict[str, Any] | None,
    schema_name: str,
    force_json: bool,
) -> dict[str, Any] | None:
    """Return the provider ``response_format`` for one call.

    ``json_schema`` asks a capable endpoint for grammar-constrained decoding;
    when no schema is available it degrades to ``json_object``. ``force_json``
    guarantees a JSON reply even for a role configured for plain text.
    """
    if config.response_format == "json_schema":
        if schema is None:
            return {"type": "json_object"}
        # ``strict: true`` requires every property to be required and no numeric
        # bounds or unions, which pydantic's default-mode schema does not
        # satisfy; providers then answer 400. Omit it and let the endpoint apply
        # its own schema handling.
        return {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema},
        }
    if force_json or config.response_format == "json_object":
        return {"type": "json_object"}
    return None


class LlmClient:
    """Async, cached, concurrency-limited OpenAI-compatible client."""

    def __init__(self, cache: ContentCache) -> None:
        """Store the content cache used to deduplicate identical calls."""
        self._cache = cache
        self._sdk: dict[tuple[str, str, float], AsyncOpenAI] = {}
        self._semaphores: dict[RoleName, asyncio.Semaphore] = {}
        # Round-robin cursor per role, advanced when a role names several
        # replicas. asyncio runs on one thread, so the read-and-increment needs
        # no lock; it happens before the first ``await``.
        self._round_robin: dict[RoleName, int] = {}

    def _endpoint(self, role: RoleName, config: LlmRoleConfig) -> AsyncOpenAI:
        # A role may name several identical replicas (comma-separated). Pick the
        # next one round-robin so concurrent calls fan out across the cards; with
        # one URL this is a no-op. The SDK cache key covers the concrete endpoint
        # along with every other transport-affecting field — roles commonly share
        # a base_url but differ in timeout, and keying on (base_url, api_key)
        # alone would let the first role's timeout silently govern the rest.
        urls = base_urls(config)
        index = self._round_robin.get(role, 0)
        base_url = urls[index % len(urls)]
        self._round_robin[role] = index + 1
        key = (base_url, _api_key(config), config.timeout_s)
        client = self._sdk.get(key)
        if client is None:
            client = AsyncOpenAI(
                base_url=base_url or None,
                api_key=_api_key(config),
                timeout=config.timeout_s,
                max_retries=0,  # we own retries centrally
            )
            self._sdk[key] = client
        return client

    def _semaphore(self, role: RoleName, config: LlmRoleConfig) -> asyncio.Semaphore:
        semaphore = self._semaphores.get(role)
        if semaphore is None:
            semaphore = asyncio.Semaphore(config.concurrency)
            self._semaphores[role] = semaphore
        return semaphore

    def _cache_key(
        self,
        role: RoleName,
        config: LlmRoleConfig,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None,
        response_schema: dict[str, Any] | None,
        *,
        force_json: bool,
        schema_name: str,
        seed: int | None,
    ) -> str:
        return content_hash(
            {
                "role": role.value,
                "provider": config.provider,
                "base_url": config.base_url,
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "response_format": config.response_format,
                "extra_body": config.extra_body,
                "force_json": force_json,
                "schema_name": schema_name,
                "response_schema": response_schema,
                "seed": seed,
                "messages": [message.to_openai() for message in messages],
                "tools": [tool.to_openai() for tool in tools] if tools else None,
            }
        )

    async def complete(
        self,
        role: RoleName,
        config: LlmRoleConfig,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
        force_json: bool = False,
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        seed: int | None = None,
    ) -> LlmResponse:
        """Run one chat completion with caching and centralized retries.

        ``seed`` is forwarded to the provider when set, so a stochastic
        generation is reproducible (``None`` leaves sampling unseeded). It is
        part of the cache key: the same prompt at a different seed is a
        different request.
        """
        cache_key = self._cache_key(
            role,
            config,
            messages,
            tools,
            response_schema,
            force_json=force_json,
            schema_name=schema_name,
            seed=seed,
        )
        if config.cache:
            cached = self._cache.get(f"llm/{role.value}", cache_key)
            if isinstance(cached, dict):
                _log.debug("llm cache hit", role=role.value)
                return LlmResponse.model_validate({**cached, "cached": True})

        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [message.to_openai() for message in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if tools:
            payload["tools"] = [tool.to_openai() for tool in tools]
        if seed is not None:
            payload["seed"] = seed
        response_format = build_response_format(
            config, schema=response_schema, schema_name=schema_name, force_json=force_json
        )
        if response_format is not None:
            payload["response_format"] = response_format

        async def _call() -> Any:
            return await self._endpoint(role, config).chat.completions.create(
                **payload, extra_body=config.extra_body or None
            )

        semaphore = self._semaphore(role, config)
        try:
            async with semaphore:
                completion = await with_retry(
                    _call,
                    policy=config.retry,
                    retry_on=is_transport_retryable,
                    logger=_log,
                )
        except OSError as exc:
            raise LlmError(f"{role.value}: transport failure: {exc}") from exc
        except Exception as exc:  # provider SDK errors
            raise LlmError(f"{role.value}: completion failed: {exc}") from exc

        choices = getattr(completion, "choices", None)
        if not choices:
            raise LlmError(f"{role.value}: provider returned no choices")
        choice = choices[0]
        message = getattr(choice, "message", None)
        usage = getattr(completion, "usage", None)
        response = LlmResponse(
            text=_message_text(message),
            model=getattr(completion, "model", None) or config.model,
            finish_reason=getattr(choice, "finish_reason", None),
            reasoning=_reasoning_text(message),
            usage=TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            ),
        )
        if config.cache:
            self._cache.put(f"llm/{role.value}", cache_key, response.model_dump(mode="json"))
        return response

    async def aclose(self) -> None:
        """Close every cached SDK client."""
        for client in self._sdk.values():
            await client.close()
        self._sdk.clear()
