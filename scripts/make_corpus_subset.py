"""Build a small, reproducible corpus subset from an OpenCode SQLite database.

The full corpus is large and expensive to analyse, so the first pipeline runs use
a handful of *substantive* sessions. Selecting them by a deterministic score
(message count, tool calls, tool errors) keeps "start small" honest: the subset
is content, not a random slice, and the same flags always pick the same sessions.

This is plumbing only. It copies the ``session_v2`` and ``session_message``
tables verbatim (schema and rows), so the pipeline's read-only ingest path sees a
faithful database rather than a lossy projection.

Usage::

    uv run python scripts/make_corpus_subset.py \\
        --source .context/opencode.db --dest .context/small.db \\
        --session-id ses_abc --session-id ses_def
    uv run python scripts/make_corpus_subset.py \\
        --source .context/opencode.db --dest .context/top10.db --top 10

The ``--top`` ranking favours the largest sessions, which cost hours of LLM
analysis each. To pick a *budgeted* set of friction-rich sessions instead, bound
the size and require evidence of friction before taking the top N::

    uv run python scripts/make_corpus_subset.py \\
        --source .context/opencode.db --dest .context/medium.db \\
        --max-messages 400 --min-tool-errors 4 --top 25
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

_COPY_TABLES = ("session_v2", "session_message")


@dataclass(frozen=True, slots=True)
class SessionScore:
    """A session's size and friction signals, used only for ranking."""

    session_id: str
    title: str
    messages: int
    user_turns: int
    tool_calls: int
    tool_errors: int

    @property
    def score(self) -> float:
        """Rank by friction-bearing volume, with errors weighted heavily.

        A session with many tool errors exercises the friction analyser; user
        turns exercise episode cutting. The weights are a selection heuristic
        for *this* tool only — never a pipeline judgment.
        """
        return self.messages + 20.0 * self.tool_errors + 5.0 * self.user_turns


def _open_ro(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _tool_stats(connection: sqlite3.Connection, session_id: str) -> tuple[int, int]:
    """Return ``(tool_calls, tool_errors)`` for one session's assistant rows."""
    calls = errors = 0
    for row in connection.execute(
        "SELECT data FROM session_message WHERE session_id = ? AND type = 'assistant'",
        (session_id,),
    ):
        try:
            data = json.loads(row["data"])
        except (json.JSONDecodeError, TypeError):
            continue
        for part in data.get("content") or []:
            if not isinstance(part, dict) or part.get("type") != "tool":
                continue
            calls += 1
            state = part.get("state") or {}
            if state.get("status") in {"error", "failed"} or state.get("error"):
                errors += 1
    return calls, errors


def rank_sessions(connection: sqlite3.Connection) -> list[SessionScore]:
    """Rank every session by friction-bearing volume, descending."""
    scores: list[SessionScore] = []
    rows = connection.execute(
        """
        SELECT s.id, s.title,
               COUNT(m.id) AS messages,
               SUM(CASE WHEN m.type = 'user' THEN 1 ELSE 0 END) AS user_turns
        FROM session_v2 s
        LEFT JOIN session_message m ON m.session_id = s.id
        GROUP BY s.id
        """
    )
    for row in rows:
        session_id = str(row["id"])
        calls, errors = _tool_stats(connection, session_id)
        scores.append(
            SessionScore(
                session_id=session_id,
                title=str(row["title"] or ""),
                messages=int(row["messages"] or 0),
                user_turns=int(row["user_turns"] or 0),
                tool_calls=calls,
                tool_errors=errors,
            )
        )
    return sorted(scores, key=lambda score: score.score, reverse=True)


def _table_sql(connection: sqlite3.Connection, table: str) -> str:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    if row is None or not row["sql"]:
        raise SystemExit(f"source database has no table {table!r}")
    return str(row["sql"])


def build_subset(source: Path, dest: Path, session_ids: list[str]) -> int:
    """Copy the named sessions (and their messages) into a new database."""
    if not source.is_file():
        raise SystemExit(f"source database not found: {source}")
    if dest.exists():
        dest.unlink()
    source_connection = _open_ro(source)
    try:
        available = {
            str(row["id"]) for row in source_connection.execute("SELECT id FROM session_v2")
        }
        missing = [session_id for session_id in session_ids if session_id not in available]
        if missing:
            raise SystemExit(f"unknown session ids: {', '.join(missing)}")

        destination = sqlite3.connect(dest)
        try:
            for table in _COPY_TABLES:
                destination.execute(_table_sql(source_connection, table))
            placeholders = ",".join("?" for _ in session_ids)
            for table in _COPY_TABLES:
                column = "id" if table == "session_v2" else "session_id"
                rows = source_connection.execute(
                    f"SELECT * FROM {table} WHERE {column} IN ({placeholders})", session_ids
                ).fetchall()
                if not rows:
                    continue
                keys = rows[0].keys()
                insert = (
                    f"INSERT INTO {table} ({', '.join(keys)}) "
                    f"VALUES ({', '.join('?' for _ in keys)})"
                )
                destination.executemany(insert, [tuple(row) for row in rows])
            destination.commit()
        finally:
            destination.close()
    finally:
        source_connection.close()
    return len(session_ids)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Source OpenCode DB.")
    parser.add_argument("--dest", type=Path, required=True, help="Destination DB.")
    parser.add_argument(
        "--session-id", action="append", default=[], help="Session to include (repeatable)."
    )
    parser.add_argument(
        "--top", type=int, default=0, help="Instead of ids, take the top N by score."
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help="Exclude sessions longer than this (0 = no bound); keeps analysis cost bounded.",
    )
    parser.add_argument(
        "--min-tool-errors",
        type=int,
        default=0,
        help="Require at least this many tool errors (0 = no requirement); favours friction.",
    )
    parser.add_argument("--list", action="store_true", help="Print the ranking and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    connection = _open_ro(args.source)
    try:
        ranking = rank_sessions(connection)
    finally:
        connection.close()

    if args.max_messages:
        ranking = [score for score in ranking if score.messages <= args.max_messages]
    if args.min_tool_errors:
        ranking = [score for score in ranking if score.tool_errors >= args.min_tool_errors]

    if args.list:
        print(f"{'score':>10} {'msgs':>6} {'user':>5} {'tools':>6} {'terr':>5}  title / id")
        for score in ranking:
            print(
                f"{score.score:>10.0f} {score.messages:>6} {score.user_turns:>5} "
                f"{score.tool_calls:>6} {score.tool_errors:>5}  "
                f"{score.title[:40]!r} {score.session_id}"
            )
        return 0

    session_ids = list(args.session_id)
    if not session_ids and args.top:
        session_ids = [score.session_id for score in ranking[: args.top]]
    if not session_ids:
        raise SystemExit("provide --session-id (repeatable), --top N, or --list")

    count = build_subset(args.source, args.dest, session_ids)
    print(f"wrote {count} session(s) to {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
