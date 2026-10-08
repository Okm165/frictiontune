"""Tests for the OpenAI-compatible embedding backend's cache identity.

The cache key must distinguish requests that would produce different vectors:
same text against a different endpoint or embedding dimension is not the same
result, so reusing a cached vector would silently corrupt retrieval.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from frictiontune.backends.embedding import OpenAiEmbedder
from frictiontune.config.models import EmbeddingConfig
from frictiontune.core.cache import ContentCache
from frictiontune.core.errors import LlmError

if TYPE_CHECKING:
    from pathlib import Path


def _embedder(tmp_path: Path, rows: list[SimpleNamespace]) -> OpenAiEmbedder:
    """An embedder whose endpoint returns exactly ``rows``."""

    class _Embeddings:
        async def create(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(data=rows)

    class _Client:
        embeddings = _Embeddings()

    cache = ContentCache(tmp_path)
    embedder = OpenAiEmbedder(
        EmbeddingConfig(model="m", base_url="http://a/v1", cache=False), cache
    )
    embedder._client = cast("Any", _Client())
    return embedder


def test_embedding_cache_key_distinguishes_endpoint_and_dimensions(tmp_path: Path) -> None:
    cache = ContentCache(tmp_path)
    a = OpenAiEmbedder(EmbeddingConfig(model="m", base_url="http://a/v1"), cache)
    b = OpenAiEmbedder(EmbeddingConfig(model="m", base_url="http://b/v1"), cache)
    wider = OpenAiEmbedder(
        EmbeddingConfig(model="m", base_url="http://a/v1", dimensions=256), cache
    )

    assert a._cache_key(["x"]) == a._cache_key(["x"])
    assert a._cache_key(["x"]) != b._cache_key(["x"])
    assert a._cache_key(["x"]) != wider._cache_key(["x"])
    assert a._cache_key(["x"]) != a._cache_key(["y"])


def test_embed_reorders_vectors_by_response_index(tmp_path: Path) -> None:
    import asyncio

    rows = [
        SimpleNamespace(index=1, embedding=[2.0, 2.0]),
        SimpleNamespace(index=0, embedding=[1.0, 1.0]),
    ]
    embedder = _embedder(tmp_path, rows)

    vectors = asyncio.run(embedder.embed(["a", "b"]))

    assert vectors == [[1.0, 1.0], [2.0, 2.0]]


def test_embed_rejects_short_response(tmp_path: Path) -> None:
    import asyncio

    embedder = _embedder(tmp_path, [SimpleNamespace(index=0, embedding=[1.0])])

    with pytest.raises(LlmError, match="returned 1 vectors for 2 texts"):
        asyncio.run(embedder.embed(["a", "b"]))
