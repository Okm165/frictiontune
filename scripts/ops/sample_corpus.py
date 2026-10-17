#!/usr/bin/env python3
"""Sample the newest sessions of a corpus DB into a smaller one.

The pipeline's cost is linear in the corpus: `normalize` compacts every large
tool output and `friction_detect` analyses every episode, so a corpus 11x the
size of the one you developed against is roughly 11x the local inference. That
makes "run it on everything" a multi-day job and a poor way to find out what a
larger corpus does to the dataset.

This writes a *contiguous newest slice* of an OpenCode corpus — `session_v2` and
`session_message` rows, same schema — so the pipeline can be run at a chosen
scale. Contiguous and newest on purpose: `decontaminate` splits train from
held-out by a time quantile, so a slice that ends "now" keeps the same
past-trains / future-evaluates shape as the full corpus.

The slice is sized by **message characters**, not by session count, because
characters are what the LLM stages pay for — sessions differ by three orders of
magnitude, and 40 sessions can be less work than 25.

Usage:
    uv run scripts/ops/sample_corpus.py --source .context/opencode.db \
        --out .context/scale.db --max-chars 200000000
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_TABLES = ("session_v2", "session_message")


def _schema(connection: sqlite3.Connection, table: str) -> str:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if row is None or not row[0]:
        raise SystemExit(f"table {table!r} not found in the source database")
    return str(row[0])


def _columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')]


def main(argv: list[str]) -> int:
    """Write a newest-first slice of ``source`` into ``out``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--max-chars",
        type=int,
        required=True,
        help="Stop after this many message characters (the pipeline's cost driver).",
    )
    args = parser.parse_args(argv)

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; remove it or pick another path")

    source = sqlite3.connect(f"file:{args.source}?mode=ro", uri=True)
    out = sqlite3.connect(args.out)
    try:
        for table in _TABLES:
            out.execute(_schema(source, table))

        sessions = list(source.execute("SELECT * FROM session_v2 ORDER BY time_created DESC"))
        session_columns = _columns(source, "session_v2")
        message_columns = _columns(source, "session_message")
        message_placeholders = ", ".join("?" for _ in message_columns)
        session_placeholders = ", ".join("?" for _ in session_columns)

        taken = 0
        chars = 0
        messages = 0
        for session in sessions:
            rows = list(
                source.execute("SELECT * FROM session_message WHERE session_id = ?", (session[0],))
            )
            size = sum(len(str(row[-1] or "")) for row in rows)
            if chars and chars + size > args.max_chars:
                break
            out.execute(f"INSERT INTO session_v2 VALUES ({session_placeholders})", tuple(session))
            for row in rows:
                out.execute(
                    f"INSERT INTO session_message VALUES ({message_placeholders})",
                    tuple(row),
                )
            taken += 1
            chars += size
            messages += len(rows)
        out.commit()
    finally:
        source.close()
        out.close()

    print(
        f"wrote {args.out}: {taken} sessions, {messages:,} messages, "
        f"{chars:,} characters (source had {len(sessions)} sessions)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
