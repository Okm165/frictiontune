"""Artifact store: the single place that touches run outputs.

Layout::

    artifacts/<pipeline>/<run_id>/
        manifest.json
        report.md
        data/<name>.jsonl

Every write returns a :class:`DatasetRef` carrying the content hash so downstream
stages can prove which bytes they consumed. Data artifacts are line-oriented, so
a stage can stream them record by record and never holds its output in memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from frictiontune.core.errors import ArtifactError
from frictiontune.core.hashing import content_hash, file_sha256
from frictiontune.core.json import append_jsonl, dumps, iter_jsonl, try_loads, write_jsonl

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path


class DatasetRef(BaseModel):
    """A pointer to a written artifact plus its integrity hash and kind."""

    name: str
    path: str
    sha256: str
    rows: int
    kind: str = "text"


def _as_dict(record: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, BaseModel):
        return record.model_dump(mode="json")
    return record


class JsonlSink:
    """A line-oriented sink for heterogeneous nested records.

    JSONL tolerates records with differing shapes, which nested conversation
    traces have. Stages that emit traces stream through this sink.
    """

    def __init__(self, target: Path) -> None:
        """Open an append handle for the JSONL file at ``target``."""
        self._target = target
        target.parent.mkdir(parents=True, exist_ok=True)
        self._handle = target.open("w", encoding="utf-8")
        self._rows = 0
        self._closed = False

    def write(self, record: BaseModel | dict[str, Any]) -> None:
        """Append one record as a JSON line."""
        if self._closed:
            raise ArtifactError("cannot write to a closed JsonlSink")
        append_jsonl(self._handle, _as_dict(record))
        self._rows += 1

    def close(self) -> DatasetRef:
        """Finalise the file and return its integrity reference."""
        if self._closed:
            raise ArtifactError("JsonlSink closed twice")
        self._handle.close()
        self._closed = True
        return DatasetRef(
            name=self._target.stem,
            path=str(self._target),
            sha256=file_sha256(self._target),
            rows=self._rows,
            kind="jsonl",
        )


class ArtifactStore:
    """Read and write everything a stage produces, under one run directory."""

    def __init__(self, root: Path, pipeline: str, run_id: str) -> None:
        """Bind a store to one stage run under ``root/pipeline``."""
        self._root = root
        self._pipeline = pipeline
        self._run_id = run_id

    @property
    def directory(self) -> Path:
        """The run directory for this (pipeline, run_id)."""
        return self._root / self._pipeline / self._run_id

    @property
    def run_id(self) -> str:
        """The run identifier."""
        return self._run_id

    def ensure(self) -> None:
        """Create the run and data directories if absent."""
        (self.directory / "data").mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        """Resolve an artifact name to a path under the run directory."""
        return self.directory / name

    @staticmethod
    def _with_suffix(name: str, suffix: str) -> str:
        """Append ``suffix`` to a logical name unless it already carries it."""
        return name if name.endswith(suffix) else name + suffix

    def data_path(self, name: str, suffix: str = ".jsonl") -> Path:
        """Resolve a data artifact name (extension added if missing)."""
        return self.directory / "data" / self._with_suffix(name, suffix)

    def exists(self, name: str) -> bool:
        """Whether an artifact with ``name`` has been written."""
        return self.path(name).exists()

    # -- structured text ---------------------------------------------------

    def write_json(self, name: str, value: Any) -> DatasetRef:
        """Write a JSON artifact and return its integrity reference."""
        target = self.path(self._with_suffix(name, ".json"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(dumps(value), encoding="utf-8")
        return DatasetRef(
            name=target.stem,
            path=str(target),
            sha256=file_sha256(target),
            rows=1,
            kind="json",
        )

    def read_json(self, name: str) -> Any:
        """Read a JSON artifact, raising :class:`ArtifactError` if absent."""
        target = self.path(self._with_suffix(name, ".json"))
        if not target.is_file():
            raise ArtifactError(f"missing artifact: {name}")
        return try_loads(target.read_text(encoding="utf-8"))

    def write_text(self, name: str, text: str) -> DatasetRef:
        """Write a plain-text artifact (e.g. a markdown report)."""
        target = self.path(self._with_suffix(name, ".md"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return DatasetRef(
            name=target.stem,
            path=str(target),
            sha256=file_sha256(target),
            rows=1,
            kind="text",
        )

    # -- jsonl -------------------------------------------------------------

    def write_jsonl(self, name: str, rows: Iterable[BaseModel | dict[str, Any]]) -> DatasetRef:
        """Stream records to a JSONL data artifact."""
        target = self.data_path(name, ".jsonl")
        count = write_jsonl(target, (_as_dict(row) for row in rows))
        return DatasetRef(
            name=target.stem,
            path=str(target),
            sha256=file_sha256(target),
            rows=count,
            kind="jsonl",
        )

    def iter_jsonl(self, name: str) -> Iterator[Any]:
        """Iterate a JSONL data artifact."""
        target = self.data_path(name, ".jsonl")
        if not target.is_file():
            raise ArtifactError(f"missing artifact: {name}")
        yield from iter_jsonl(target)

    def jsonl_sink(self, name: str) -> JsonlSink:
        """Return a streaming, append-style JSONL sink for a data artifact."""
        return JsonlSink(self.data_path(name, ".jsonl"))

    def digest_of(self, name: str) -> str:
        """Hash an already-written artifact by logical name.

        The suffix a logical name maps to depends on how it was written
        (``.json``/``.md`` at the run root; ``.jsonl`` under ``data/``), so every
        known location is probed before failing.
        """
        candidates = (
            self.path(name),
            self.path(self._with_suffix(name, ".json")),
            self.path(self._with_suffix(name, ".md")),
            self.data_path(name, ".jsonl"),
        )
        for target in candidates:
            if target.is_file():
                return file_sha256(target)
        raise ArtifactError(f"cannot hash missing artifact: {name}")

    @staticmethod
    def config_fingerprint(config: Any) -> str:
        """Hash a configuration object for the manifest."""
        return content_hash(config)
