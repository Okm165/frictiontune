"""Deterministic hashing used for run identity, caching, and dedup plumbing.

Two ideas matter:

* :func:`content_hash` — a stable hash of any JSON-able value, used for cache
  keys and the exact-duplicate dedup pass.
* :func:`compute_run_id` — a hash of *(config, code revision, input hashes)* so
  a stage is idempotent: same inputs and same code can never produce a new run.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from frictiontune.core.json import dumps

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

_HASH_PREFIX_LEN = 12


def content_hash(value: Any) -> str:
    """Return the SHA-256 hex digest of the canonical JSON form of ``value``."""
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    """Return the SHA-256 hex digest of UTF-8 ``text`` (exact byte identity)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Hash a file in a streaming fashion so large artifacts never enter RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def combine_hashes(hashes: Iterable[str]) -> str:
    """Order-sensitive combination of child hashes into one digest."""
    digest = hashlib.sha256()
    for item in hashes:
        digest.update(item.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def compute_run_id(
    *,
    stage: str,
    config: Mapping[str, Any],
    code_revision: str,
    input_hashes: Iterable[str],
) -> str:
    """Deterministic run identifier for a stage invocation."""
    return content_hash(
        {
            "stage": stage,
            "config": config,
            "code_revision": code_revision,
            "inputs": list(input_hashes),
        }
    )


def short(hash_value: str, length: int = _HASH_PREFIX_LEN) -> str:
    """Abbreviate a hash for display in logs, run directories, and reports."""
    return hash_value[:length]
