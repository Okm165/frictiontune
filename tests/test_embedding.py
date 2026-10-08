"""Tests for the OpenAI-compatible embedding backend's cache identity.

The cache key must distinguish requests that would produce different vectors:
same text against a different endpoint or embedding dimension is not the same
result, so reusing a cached vector would silently corrupt retrieval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from frictiontune.backends.embedding import OpenAiEmbedder
from frictiontune.config.models import EmbeddingConfig
from frictiontune.core.cache import ContentCache

if TYPE_CHECKING:
    from pathlib import Path


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
