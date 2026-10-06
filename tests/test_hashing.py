"""Tests for deterministic hashing and run identity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from frictiontune.core.hashing import (
    combine_hashes,
    compute_run_id,
    content_hash,
    file_sha256,
    short,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_content_hash_is_stable_and_order_insensitive_to_keys() -> None:
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


def test_content_hash_changes_with_value() -> None:
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_combine_hashes_is_order_sensitive() -> None:
    assert combine_hashes(["x", "y"]) != combine_hashes(["y", "x"])


def test_run_id_is_deterministic_and_input_sensitive() -> None:
    first = compute_run_id(
        stage="judge",
        config={"threshold": 20},
        code_revision="abc",
        input_hashes=["h1", "h2"],
    )
    same = compute_run_id(
        stage="judge",
        config={"threshold": 20},
        code_revision="abc",
        input_hashes=["h1", "h2"],
    )
    changed = compute_run_id(
        stage="judge",
        config={"threshold": 21},
        code_revision="abc",
        input_hashes=["h1", "h2"],
    )
    assert first == same
    assert first != changed


def test_file_sha256_matches_streamed_read(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    target.write_bytes(b"hello world")
    assert file_sha256(target) == file_sha256(target, chunk_size=2)


def test_short_truncates() -> None:
    assert short("0123456789abcdef") == "0123456789ab"
    assert len(short("abc", length=2)) == 2
