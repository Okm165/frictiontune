"""Tests for the corpus reader registry — the chat-log-source seam.

Supporting a new chat-log source is one reader class (satisfying the
`CorpusReader` protocol) and one entry in `READERS`; the pipeline itself never
learns a source's name.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from frictiontune.corpus import READERS, CorpusReader, OpenCodeCorpus, get_reader


def test_opencode_is_registered_and_satisfies_the_protocol() -> None:
    assert "opencode" in READERS
    reader = get_reader("opencode")
    assert reader is OpenCodeCorpus
    # The structural contract: bind to a path, walk sessions.
    assert callable(reader.iter_sessions)


def test_unknown_format_names_the_available_set() -> None:
    with pytest.raises(KeyError, match="available: opencode"):
        get_reader("claude-logs")


def test_readers_bind_a_single_source_path() -> None:
    # The contract ingest dispatches on: one constructor argument.
    reader = get_reader("opencode")(Path("/tmp/does-not-matter.db"))
    assert isinstance(reader, CorpusReader)
