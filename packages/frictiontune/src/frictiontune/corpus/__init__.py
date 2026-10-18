"""Corpus readers: turn external sources into canonical trace objects.

A reader is anything that can walk a chat-log source and yield canonical
:class:`~frictiontune.domain.trace.Session` / :class:`~frictiontune.domain.trace.Message`
objects; :class:`CorpusReader` is the (structural) contract, and readers
register in :data:`READERS` under the name ``ingest``'s ``corpus_format``
setting selects. Supporting a new source — Claude logs, ChatGPT exports, an
internal tool's database — means writing one class with
:meth:`CorpusReader.iter_sessions`, adding it to :data:`READERS`, and setting
one config value; ``ingest`` streams one session's messages at a time, so
readers never need to hold the corpus in memory.

Redaction and compaction happen *downstream* (``normalize``), so a reader's only
quality obligations are faithful roles, timestamps that sort chronologically,
and honest tool-call structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from frictiontune.corpus.opencode import CorpusStats, OpenCodeCorpus

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from frictiontune.domain.trace import Message, Session


@runtime_checkable
class CorpusReader(Protocol):
    """The structural contract for a chat-log source.

    Satisfying this protocol is all ``ingest`` needs: the constructor binds the
    reader to its source, ``iter_sessions`` walks it. No registration
    machinery beyond :data:`READERS` — a source is a class and a name.
    """

    def __init__(self, db_path: Path) -> None:
        """Bind the reader to its source path."""
        ...

    def iter_sessions(self) -> Iterator[tuple[Session, list[Message], CorpusStats]]:
        """Yield ``(session, messages, stats)`` in chronological order."""
        ...


# The reader registry: ``ingest``'s ``corpus_format`` setting selects here.
# Adding a source is one class and one entry — no pipeline changes.
READERS: dict[str, type[CorpusReader]] = {
    "opencode": OpenCodeCorpus,
}


def get_reader(name: str) -> type[CorpusReader]:
    """Return the reader class registered under ``name``.

    Unknown names fail with the available set, so a typo in the config names
    its options instead of surfacing as a missing-class error mid-run.
    """
    reader = READERS.get(name)
    if reader is None:
        available = ", ".join(sorted(READERS)) or "(none registered)"
        raise KeyError(f"unknown corpus format {name!r}; available: {available}")
    return reader


__all__ = ["READERS", "CorpusReader", "CorpusStats", "OpenCodeCorpus", "get_reader"]
