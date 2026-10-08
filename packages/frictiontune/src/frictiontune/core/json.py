"""Canonical JSON handling and JSONL streaming.

This is *plumbing*: stable serialization, tolerant parsing, and line-oriented
streaming. No semantic judgment lives here.
"""

from __future__ import annotations

import json
import re
from typing import IO, TYPE_CHECKING, Any, TypeAlias

from pydantic import BaseModel
from typing_extensions import TypeAliasType

from frictiontune.core.errors import ArtifactError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

JsonScalar: TypeAlias = bool | int | float | str | None

# A fenced ```json (...)``` block: the explicit "this is the payload" signal a
# coding agent emits for a tool call. The content is matched up to the closing
# fence (not to the first brace), so a nested object is captured whole. Used only
# to *measure* structural validity (a reward/format term), never to author or
# repair content.
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# A recursive JSON alias. It must be a lazy ``TypeAliasType`` (not an implicit
# ``list["JsonValue"]`` alias): pydantic resolves field annotations eagerly, and
# an implicit recursive alias makes it recurse until the stack overflows. The
# backport keeps this working on Python 3.11 without PEP 695 syntax.
JsonValue = TypeAliasType("JsonValue", "JsonScalar | list[JsonValue] | dict[str, JsonValue]")
JsonObject: TypeAlias = dict[str, JsonValue]


def dumps(value: Any, *, sort_keys: bool = True) -> str:
    """Serialize ``value`` to compact, deterministic JSON.

    Pydantic models are serialized in their JSON mode so that the canonical form
    (used for hashing) never differs between runs.
    """
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        sort_keys=sort_keys,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def loads(text: str) -> JsonValue:
    """Parse JSON, raising the standard :class:`json.JSONDecodeError` on failure."""
    return json.loads(text)  # type: ignore[no-any-return]


def try_loads(text: str) -> JsonValue | None:
    """Parse JSON, returning ``None`` instead of raising."""
    try:
        return json.loads(text)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, TypeError):
        return None


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    """Stream ``rows`` to a JSONL file, returning the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(dumps(row))
            handle.write("\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterator[JsonValue]:
    """Yield parsed JSON objects from a JSONL file, skipping blank lines.

    A malformed line names its file and line number rather than surfacing a bare
    :class:`json.JSONDecodeError`, so an interrupted write is diagnosable.
    """
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"{path}:{number}: malformed JSON line: {exc.msg}") from exc


def append_jsonl(handle: IO[str], row: Any) -> None:
    """Append one record to an already-open JSONL handle."""
    handle.write(dumps(row))
    handle.write("\n")
    handle.flush()


def json_block_validity(text: str) -> float:
    """Fraction of fenced JSON blocks in ``text`` that parse.

    A structural measurement, not a judgment: it reports whether the JSON the
    model chose to emit is well-formed and returns ``1.0`` when it emitted no
    JSON at all (there is nothing to be malformed). Shared by the replay
    ``format_valid`` signal and the GRPO ``tool`` reward so the two cannot drift.
    """
    blocks = _JSON_BLOCK.findall(text)
    if not blocks:
        return 1.0
    valid = 0
    for block in blocks:
        try:
            json.loads(block)
        except json.JSONDecodeError:
            continue
        valid += 1
    return valid / len(blocks)
