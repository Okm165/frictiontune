"""OpenAI-compatible embedding backend with content-hash caching."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from frictiontune.core.errors import LlmError
from frictiontune.core.hashing import content_hash
from frictiontune.core.retry import RetryPolicy, with_retry
from frictiontune.llm.transport import TRANSPORT_RETRYABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from frictiontune.config.models import EmbeddingConfig
    from frictiontune.core.cache import ContentCache


class OpenAiEmbedder:
    """Embedder backed by any OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(self, config: EmbeddingConfig, cache: ContentCache) -> None:
        """Store the embedding configuration and shared cache."""
        self._config = config
        self._cache = cache
        api_key = os.environ.get(config.api_key_env, "EMPTY") if config.api_key_env else "EMPTY"
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=api_key,
            timeout=config.timeout_s,
            max_retries=0,
        )

    def _cache_key(self, texts: Sequence[str]) -> str:
        return content_hash({"model": self._config.model, "texts": list(texts)})

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` in order, caching the whole batch by content hash."""
        if not texts:
            return []
        key = self._cache_key(texts)
        if self._config.cache:
            cached = self._cache.get("embedding", key)
            if isinstance(cached, list):
                return [[float(x) for x in row] for row in cached]

        async def _call() -> object:
            if self._config.dimensions is not None:
                return await self._client.embeddings.create(
                    model=self._config.model,
                    input=list(texts),
                    dimensions=self._config.dimensions,
                )
            return await self._client.embeddings.create(
                model=self._config.model,
                input=list(texts),
            )

        try:
            response = await with_retry(_call, policy=RetryPolicy(), retry_on=TRANSPORT_RETRYABLE)
        except OSError as exc:
            raise LlmError(f"embedding transport failure: {exc}") from exc

        vectors = [list(item.embedding) for item in response.data]  # type: ignore[attr-defined]
        if self._config.cache:
            self._cache.put("embedding", key, vectors)
        return vectors

    async def aclose(self) -> None:
        """Close the underlying client."""
        await self._client.close()
