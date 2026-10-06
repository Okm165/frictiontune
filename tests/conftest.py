"""Shared pytest fixtures for the frictiontune test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def config_dir() -> Path:
    """The real, repository config tree (kept in sync by the docs)."""
    return REPO_ROOT / "configs"