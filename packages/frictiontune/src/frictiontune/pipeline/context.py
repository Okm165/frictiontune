"""Run context, artifact catalog, and per-stage context.

The :class:`ArtifactCatalog` is the connective tissue of the pipeline: each stage
publishes its outputs into the catalog, and downstream stages read their inputs
from it by name. Because every entry carries a content hash, a stage's run id is
a function of the exact bytes it consumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import polars as pl
import pyarrow.parquet as pq
from pydantic import BaseModel

from frictiontune.core.artifacts import ArtifactStore, DatasetRef, JsonlSink, ParquetSink
from frictiontune.core.errors import ValidationError as FtValidationError
from frictiontune.core.json import iter_jsonl, try_loads

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import structlog

    from frictiontune.backends.registry import BackendRegistry
    from frictiontune.config.models import RunConfig
    from frictiontune.core.cache import ContentCache
    from frictiontune.llm.gateway import LlmGateway

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(slots=True, frozen=True)
class CatalogEntry:
    """One published artifact."""

    ref: DatasetRef
    path: Path
    kind: str  # parquet | jsonl | json | text


class ArtifactCatalog:
    """Name → artifact directory, so stages can find their inputs."""

    def __init__(self) -> None:
        """Create an empty catalog."""
        self._entries: dict[str, CatalogEntry] = {}

    def publish(self, ref: DatasetRef, *, kind: str) -> None:
        """Record an artifact under its name."""
        self._entries[ref.name] = CatalogEntry(ref=ref, path=Path(ref.path), kind=kind)

    def has(self, name: str) -> bool:
        """Whether an artifact with ``name`` is published."""
        return name in self._entries

    def get(self, name: str) -> CatalogEntry:
        """Return the entry for ``name`` or raise a clear error."""
        try:
            return self._entries[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._entries)) or "(none)"
            raise FtValidationError(
                "ArtifactCatalog", 1, f"missing input {name!r}; available: {available}"
            ) from exc

    def names(self) -> list[str]:
        """List every published artifact name."""
        return sorted(self._entries)

    def hashes(self, names: Sequence[str]) -> list[str]:
        """Return the content hashes for ``names`` (used for the run id)."""
        return [self.get(name).ref.sha256 for name in names]

    # -- typed readers -----------------------------------------------------

    def read_frame(self, name: str) -> pl.DataFrame:
        """Read an artifact as a Polars frame (Parquet or NDJSON)."""
        entry = self.get(name)
        if entry.kind == "jsonl":
            return pl.read_ndjson(entry.path)
        return pl.read_parquet(entry.path)

    def stream(
        self, name: str, model: type[ModelT], *, batch_size: int = 2_000
    ) -> Iterator[ModelT]:
        """Stream an artifact into validated models, dispatching on its kind."""
        entry = self.get(name)
        if entry.kind == "jsonl":
            for row in iter_jsonl(entry.path):
                yield model.model_validate(row)
            return
        parquet = pq.ParquetFile(entry.path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield model.model_validate(row)

    def iter_records(self, name: str, model: type[ModelT]) -> Iterator[ModelT]:
        """Stream a Parquet artifact into validated models."""
        yield from self.stream(name, model)

    def iter_batches(
        self, name: str, model: type[ModelT], *, batch_size: int = 2_000
    ) -> Iterator[ModelT]:
        """Alias of :meth:`stream` kept for readability at call sites."""
        yield from self.stream(name, model, batch_size=batch_size)

    def read_json(self, name: str) -> object:
        """Read a JSON artifact."""
        entry = self.get(name)
        return try_loads(entry.path.read_text(encoding="utf-8"))

    def iter_jsonl(self, name: str) -> Iterator[object]:
        """Iterate a JSONL artifact."""
        yield from iter_jsonl(self.get(name).path)

    def read_text(self, name: str) -> str:
        """Read a plain-text artifact."""
        return self.get(name).path.read_text(encoding="utf-8")


@dataclass(slots=True)
class RunContext:
    """Everything a stage needs that is shared across the whole run."""

    config: RunConfig
    pipeline: str
    cache: ContentCache
    gateway: LlmGateway
    backends: BackendRegistry
    logger: structlog.stdlib.BoundLogger
    artifacts_root: Path
    catalog: ArtifactCatalog = field(default_factory=ArtifactCatalog)
    metrics: dict[str, float] = field(default_factory=dict)

    def store(self, stage: str, run_id: str) -> ArtifactStore:
        """Build the artifact store for one stage invocation."""
        store = ArtifactStore(self.artifacts_root, stage, run_id)
        store.ensure()
        return store


@dataclass(slots=True)
class PublishedSink:
    """A write-through sink that publishes its reference when closed."""

    sink: ParquetSink | JsonlSink
    publish: Callable[[DatasetRef], DatasetRef]

    def write(self, record: object) -> None:
        """Buffer or append one record."""
        self.sink.write(record)  # type: ignore[arg-type]

    def close(self) -> DatasetRef:
        """Finalise the file and publish its reference."""
        return self.publish(self.sink.close())


@dataclass(slots=True)
class StageContext:
    """The per-stage execution context handed to :meth:`Stage.run`."""

    run: RunContext
    stage: str
    run_id: str
    store: ArtifactStore
    outputs: dict[str, DatasetRef] = field(default_factory=dict)

    @property
    def config(self) -> RunConfig:
        """The run configuration."""
        return self.run.config

    @property
    def catalog(self) -> ArtifactCatalog:
        """The shared artifact catalog."""
        return self.run.catalog

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """A logger bound to this stage and run."""
        return self.run.logger.bind(stage=self.stage, run_id=self.run_id)

    def settings(self, model: type[ModelT]) -> ModelT:
        """Parse this stage's configuration block into a typed settings model."""
        raw = self.config.stages.get(self.stage, {})
        try:
            return model.model_validate(raw)
        except Exception as exc:
            raise FtValidationError(model.__name__, 1, str(exc)) from exc

    # -- emission (write + publish + record) -------------------------------

    def _publish(self, ref: DatasetRef, kind: str) -> DatasetRef:
        self.outputs[ref.name] = ref
        self.catalog.publish(ref, kind=ref.kind or kind)
        return ref

    def emit_parquet(self, name: str, records: object) -> DatasetRef:
        """Write, publish, and record a Parquet output."""
        ref = self.store.write_parquet(name, records)  # type: ignore[arg-type]
        return self._publish(ref, "parquet")

    def parquet_sink(self, name: str, *, batch_size: int = 2_000) -> PublishedSink:
        """Return a streaming Parquet sink that publishes on close."""
        return PublishedSink(
            self.store.parquet_sink(name, batch_size=batch_size),
            lambda ref: self._publish(ref, "parquet"),
        )

    def jsonl_sink(self, name: str) -> PublishedSink:
        """Return a streaming JSONL sink that publishes on close."""
        return PublishedSink(
            self.store.jsonl_sink(name),
            lambda ref: self._publish(ref, "jsonl"),
        )

    def emit_jsonl(self, name: str, records: object) -> DatasetRef:
        """Write, publish, and record a JSONL output."""
        ref = self.store.write_jsonl(name, records)  # type: ignore[arg-type]
        return self._publish(ref, "jsonl")

    def emit_json(self, name: str, value: object) -> DatasetRef:
        """Write, publish, and record a JSON output."""
        ref = self.store.write_json(name, value)
        return self._publish(ref, "json")

    def emit_text(self, name: str, text: str) -> DatasetRef:
        """Write, publish, and record a text output."""
        ref = self.store.write_text(name, text)
        return self._publish(ref, "text")
