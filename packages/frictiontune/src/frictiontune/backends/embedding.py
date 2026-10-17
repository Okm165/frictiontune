"""OpenAI-compatible embedding backend with content-hash caching."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from frictiontune.core.errors import LlmError
from frictiontune.core.hashing import content_hash
from frictiontune.core.retry import RetryPolicy, with_retry
from frictiontune.llm.transport import is_transport_retryable

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
        return content_hash(
            {
                "model": self._config.model,
                "base_url": self._config.base_url,
                "dimensions": self._config.dimensions,
                "texts": list(texts),
            }
        )

    def _fit(self, text: str) -> str:
        """Bound one input to the embedder's context, keeping both ends.

        A too-long text is a hard provider error, so the head and tail are kept
        (with an elision marker between): for the pipeline's texts the head is
        the task/context and the tail is the completion, and summaries and
        dedup signals live at both ends. This is input plumbing, not a judgment.
        The marker is charged against the budget so the result never exceeds it.
        """
        limit = self._config.max_input_chars
        if len(text) <= limit:
            return text
        marker = "\n...\n"
        budget = limit - len(marker)
        if budget < 2:
            return text[:limit]
        head = budget // 2
        return f"{text[:head]}{marker}{text[-(budget - head) :]}"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` in order, caching the whole batch by content hash."""
        if not texts:
            return []
        prepared = [self._fit(text) for text in texts]
        key = self._cache_key(prepared)
        if self._config.cache:
            cached = self._cache.get("embedding", key)
            if isinstance(cached, list):
                return [[float(x) for x in row] for row in cached]

        async def _call() -> Any:
            if self._config.dimensions is not None:
                return await self._client.embeddings.create(
                    model=self._config.model,
                    input=prepared,
                    dimensions=self._config.dimensions,
                )
            return await self._client.embeddings.create(
                model=self._config.model,
                input=prepared,
            )

        try:
            response = await with_retry(
                _call, policy=RetryPolicy(), retry_on=is_transport_retryable
            )
        except OSError as exc:
            raise LlmError(f"embedding transport failure: {exc}") from exc
        except Exception as exc:  # provider SDK errors
            raise LlmError(f"embedding request failed: {exc}") from exc

        # The API may return embeddings in a different order than the inputs;
        # ``index`` is the authoritative position. Sorting by it (stable when
        # absent) keeps every vector aligned with the text that produced it.
        rows = sorted(response.data, key=lambda item: getattr(item, "index", 0))
        vectors = [list(item.embedding) for item in rows]
        if len(vectors) != len(texts):
            raise LlmError(
                f"embedding endpoint returned {len(vectors)} vectors for {len(texts)} texts"
            )
        if self._config.cache:
            self._cache.put("embedding", key, vectors)
        return vectors

    async def aclose(self) -> None:
        """Close the underlying client."""
        await self._client.close()
