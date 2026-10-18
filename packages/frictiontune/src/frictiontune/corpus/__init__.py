"""Corpus readers: turn external sources into canonical trace objects.

A reader is anything that can walk a chat-log source and yield canonical
:class:`~frictiontune.domain.trace.Session` / :class:`~frictiontune.domain.trace.Message`
objects; :class:`CorpusReader` is the (structural) contract and
:class:`OpenCodeCorpus` is the first implementation. Supporting a new source —
Claude logs, ChatGPT exports, an internal tool's database — means writing one
class with :meth:`CorpusReader.iter_sessions` and importing it here; ``ingest``
streams one session's messages at a time, so readers never need to hold the
corpus in memory.

Redaction and compaction happen *downstream* (``normalize``), so a reader's only
quality obligations are faithful roles, timestamps that sort chronologically,
and honest tool-call structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from frictiontune.corpus.opencode import CorpusStats, OpenCodeCorpus

if TYPE_CHECKING:
    from collections.abc import Iterator

    from frictiontune.domain.trace import Message, Session


class CorpusReader(Protocol):
    """The structural contract for a chat-log source.

    Satisfying this protocol is all ``ingest`` needs; no registration step
    exists on purpose — import your reader and pass it (or extend ``ingest``'s
    thin dispatch) rather than configuring strings the pipeline must interpret.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Accept source-specific connection arguments."""

    def iter_sessions(self) -> Iterator[tuple[Session, list[Message], CorpusStats]]:
        """Yield ``(session, messages, stats)`` in chronological order."""
        ...


__all__ = ["CorpusReader", "CorpusStats", "OpenCodeCorpus"]
