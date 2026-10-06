"""Artifact store: the single place that touches run outputs.

Layout::

    artifacts/<pipeline>/<run_id>/
        manifest.json
        report.md
        data/<name>.parquet
        data/<name>.jsonl
        data/<name>.json

Every write returns a :class:`DatasetRef` carrying the content hash so downstream
stages can prove which bytes they consumed. Parquet is written in streaming row
groups, so a stage never holds its full output in memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from frictiontune.core.errors import ArtifactError
from frictiontune.core.hashing import content_hash, file_sha256
from frictiontune.core.json import append_jsonl, dumps, iter_jsonl, try_loads, write_jsonl

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

ModelT = TypeVar("ModelT", bound=BaseModel)

_DEFAULT_BATCH = 2_000


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


class ParquetSink:
    """An append-style Parquet writer for streaming, async producers.

    Records are buffered into row groups and flushed as the schema fills, so a
    stage can interleave ``await`` work with writing without holding its whole
    output in memory. :meth:`close` finalises the file and returns its reference.
    """

    def __init__(self, target: Path, *, batch_size: int = _DEFAULT_BATCH) -> None:
        """Prepare a sink that will write to ``target`` in batches."""
        self._target = target
        self._batch_size = batch_size
        self._writer: pq.ParquetWriter | None = None
        self._batch: list[dict[str, Any]] = []
        self._rows = 0
        self._closed = False
        target.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: BaseModel | dict[str, Any]) -> None:
        """Buffer one record, flushing a row group when the batch is full."""
        if self._closed:
            raise ArtifactError("cannot write to a closed ParquetSink")
        self._batch.append(_as_dict(record))
        if len(self._batch) >= self._batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._batch:
            return
        table = pa.Table.from_pylist(self._batch)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._target, table.schema)
        self._writer.write_table(table)
        self._rows += len(self._batch)
        self._batch = []

    def close(self) -> DatasetRef:
        """Finalise the file and return its integrity reference."""
        if self._closed:
            raise ArtifactError("ParquetSink closed twice")
        try:
            self._flush()
            if self._writer is None:
                empty = pa.Table.from_pylist([])
                self._writer = pq.ParquetWriter(self._target, empty.schema)
        except pa.ArrowInvalid as exc:
            raise ArtifactError(f"parquet schema mismatch: {exc}") from exc
        finally:
            if self._writer is not None:
                self._writer.close()
        self._closed = True
        return DatasetRef(
            name=self._target.stem,
            path=str(self._target),
            sha256=file_sha256(self._target),
            rows=self._rows,
            kind="parquet",
        )


class JsonlSink:
    """A line-oriented sink for heterogeneous nested records.

    JSONL tolerates records with differing shapes, which nested conversation
    traces have; Parquet does not. Stages that emit traces therefore stream
    through this sink while flat tables use Parquet.
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

    def data_path(self, name: str, suffix: str = ".parquet") -> Path:
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

    # -- parquet -----------------------------------------------------------

    def parquet_sink(self, name: str, *, batch_size: int = _DEFAULT_BATCH) -> ParquetSink:
        """Return a streaming, append-style Parquet sink for a data artifact."""
        return ParquetSink(self.data_path(name, ".parquet"), batch_size=batch_size)

    def write_parquet(
        self,
        name: str,
        records: Iterable[BaseModel | dict[str, Any]],
        *,
        batch_size: int = _DEFAULT_BATCH,
    ) -> DatasetRef:
        """Stream records to a Parquet data artifact in row groups."""
        target = self.data_path(name, ".parquet")
        target.parent.mkdir(parents=True, exist_ok=True)

        writer: pq.ParquetWriter | None = None
        rows = 0
        batch: list[dict[str, Any]] = []
        try:
            for record in records:
                batch.append(_as_dict(record))
                if len(batch) >= batch_size:
                    table = pa.Table.from_pylist(batch)
                    if writer is None:
                        writer = pq.ParquetWriter(target, table.schema)
                    writer.write_table(table)
                    rows += len(batch)
                    batch = []
            if batch or writer is None:
                table = pa.Table.from_pylist(batch)
                if writer is None:
                    writer = pq.ParquetWriter(target, table.schema)
                writer.write_table(table)
                rows += len(batch)
        except pa.ArrowInvalid as exc:  # inconsistent record shapes
            raise ArtifactError(f"parquet schema mismatch while writing {name}: {exc}") from exc
        finally:
            if writer is not None:
                writer.close()

        return DatasetRef(
            name=target.stem,
            path=str(target),
            sha256=file_sha256(target),
            rows=rows,
            kind="parquet",
        )

    def read_frame(self, name: str) -> pl.DataFrame:
        """Read a Parquet data artifact as a Polars frame (lazy by caller's choice)."""
        target = self.data_path(name, ".parquet")
        if not target.is_file():
            raise ArtifactError(f"missing artifact: {name}")
        return pl.read_parquet(target)

    def scan_frame(self, name: str) -> pl.LazyFrame:
        """Lazily scan a Parquet data artifact (streaming-friendly)."""
        target = self.data_path(name, ".parquet")
        if not target.is_file():
            raise ArtifactError(f"missing artifact: {name}")
        return pl.scan_parquet(target)

    def iter_records(self, name: str, model: type[ModelT]) -> Iterator[ModelT]:
        """Stream a Parquet artifact into validated pydantic records.

        Uses Polars' row iteration so memory stays bounded for large artifacts.
        """
        frame = self.read_frame(name)
        for row in frame.iter_rows(named=True):
            yield model.model_validate(row)

    def list_data(self) -> Sequence[Path]:
        """List every file under the run's ``data/`` directory."""
        data_dir = self.directory / "data"
        if not data_dir.is_dir():
            return ()
        return tuple(sorted(p for p in data_dir.iterdir() if p.is_file()))

    def digest_of(self, name: str) -> str:
        """Hash an already-written artifact by logical name.

        The suffix a logical name maps to depends on how it was written
        (``.json``/``.md`` at the run root; ``.jsonl``/``.parquet`` under
        ``data/``), so every known location is probed before failing.
        """
        candidates = (
            self.path(name),
            self.path(self._with_suffix(name, ".json")),
            self.path(self._with_suffix(name, ".md")),
            self.data_path(name, ".jsonl"),
            self.data_path(name, ".parquet"),
        )
        for target in candidates:
            if target.is_file():
                return file_sha256(target)
        raise ArtifactError(f"cannot hash missing artifact: {name}")

    @staticmethod
    def config_fingerprint(config: Any) -> str:
        """Hash a configuration object for the manifest."""
        return content_hash(config)
