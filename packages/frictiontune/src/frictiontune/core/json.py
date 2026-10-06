"""Canonical JSON handling and JSONL streaming.

This is *plumbing*: stable serialization, tolerant parsing, and line-oriented
streaming. No semantic judgment lives here.
"""

from __future__ import annotations

import json
from typing import IO, TYPE_CHECKING, Any, TypeAlias

from pydantic import BaseModel
from typing_extensions import TypeAliasType

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

JsonScalar: TypeAlias = bool | int | float | str | None

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
    return json.dumps(value, sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=False)


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
    """Yield parsed JSON objects from a JSONL file, skipping blank lines."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield loads(line)


def append_jsonl(handle: IO[str], row: Any) -> None:
    """Append one record to an already-open JSONL handle."""
    handle.write(dumps(row))
    handle.write("\n")
    handle.flush()
