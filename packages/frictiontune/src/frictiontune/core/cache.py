"""Disk-backed content cache.

Cache keys are :func:`~frictiontune.core.hashing.content_hash` values, so the
same logical request always maps to the same file. The cache is intentionally
dumb: it stores bytes under ``<namespace>/<aa>/<hash>.json``. Cache misses and
hits are logged by callers, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from frictiontune.core.hashing import content_hash
from frictiontune.core.json import dumps, try_loads

if TYPE_CHECKING:
    from pathlib import Path


class ContentCache:
    """A namespaced, file-per-key JSON cache."""

    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        """Create a content cache rooted at ``root``."""
        self._root = root
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Whether reads and writes are active."""
        return self._enabled

    def key(self, payload: Any) -> str:
        """Return the canonical cache key for ``payload``."""
        return content_hash(payload)

    def _path(self, namespace: str, key: str) -> Path:
        return self._root / namespace / key[:2] / f"{key}.json"

    def get(self, namespace: str, key: str) -> Any | None:
        """Return the cached value for ``key`` or ``None`` on a miss."""
        if not self._enabled:
            return None
        path = self._path(namespace, key)
        if not path.is_file():
            return None
        return try_loads(path.read_text(encoding="utf-8"))

    def put(self, namespace: str, key: str, value: Any) -> None:
        """Store ``value`` under ``key``."""
        if not self._enabled:
            return
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(value), encoding="utf-8")
