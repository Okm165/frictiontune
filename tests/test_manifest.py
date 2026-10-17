"""Tests for the run manifest's code revision fingerprint."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from frictiontune.pipeline import manifest as manifest_module
from frictiontune.version import __version__

if TYPE_CHECKING:
    import pytest


def test_clean_tree_revision_is_the_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_git(*args: str) -> str | None:
        return "abc123\n" if args[0] == "rev-parse" else ""

    monkeypatch.setattr(manifest_module, "_git", fake_git)

    assert manifest_module.code_revision() == "abc123"


def test_dirty_tree_revision_appends_a_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_git(*args: str) -> str | None:
        if args[0] == "rev-parse":
            return "abc123\n"
        if args[0] == "status":
            return " M packages/frictiontune/src/frictiontune/domain/trace.py\n"
        if args[0] == "diff":
            return "diff --git a/x b/x\n-old\n+new\n"
        return ""

    monkeypatch.setattr(manifest_module, "_git", fake_git)

    revision = manifest_module.code_revision()

    assert revision.startswith("abc123+dirty.")
    assert revision != "abc123"


def test_editing_an_already_dirty_file_changes_the_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The dirty *path list* is identical across both calls; only the file's
    # content differs. The revision must still change, or a fixed stage would
    # silently reuse the artifact from the previous code.
    diffs = iter(["diff --git a/x b/x\n-a\n", "diff --git a/x b/x\n-b\n"])

    def fake_git(*args: str) -> str | None:
        if args[0] == "rev-parse":
            return "abc123\n"
        if args[0] == "status":
            return " M x.py\n"
        if args[0] == "diff":
            return next(diffs)
        return ""

    monkeypatch.setattr(manifest_module, "_git", fake_git)

    assert manifest_module.code_revision() != manifest_module.code_revision()


def test_a_new_untracked_source_file_changes_the_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_git(*args: str) -> str | None:
        if args[0] == "rev-parse":
            return "abc123\n"
        if args[0] == "status":
            return "?? packages/frictiontune/new.py\n"
        if args[0] == "diff":
            return ""
        if args[0] == "ls-files":
            return "packages/frictiontune/new.py\n"
        return ""

    monkeypatch.setattr(manifest_module, "_git", fake_git)
    monkeypatch.setattr(pathlib.Path, "read_bytes", lambda self: b"print(1)\n")

    assert manifest_module.code_revision().startswith("abc123+dirty.")


def test_missing_git_falls_back_to_the_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manifest_module, "_git", lambda *args: None)

    assert manifest_module.code_revision() == f"frictiontune/{__version__}"
