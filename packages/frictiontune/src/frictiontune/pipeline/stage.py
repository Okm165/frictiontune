"""Stage base class and result contract.

A :class:`Stage` is an atomic, independently runnable transform with a declared
ROI contract (:class:`~frictiontune.pipeline.spec.StageSpec`) and a typed
settings model. Stages receive a :class:`~frictiontune.pipeline.context.StageContext`
and return a :class:`StageResult`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from frictiontune.core.artifacts import DatasetRef

if TYPE_CHECKING:
    from frictiontune.pipeline.context import RunContext, StageContext
    from frictiontune.pipeline.spec import StageSpec


class StageSettings(BaseModel):
    """Base class for a stage's typed knobs.

    Settings are strict: an unknown key in a stage config block is an error, so
    configuration typos never pass silently.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class StageResult(BaseModel):
    """The self-describing outcome of running a stage."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    run_id: str
    outputs: dict[str, DatasetRef] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def from_context(
        cls,
        ctx: StageContext,
        *,
        metrics: dict[str, float] | None = None,
        notes: list[str] | None = None,
    ) -> StageResult:
        """Assemble a result from everything the stage emitted."""
        return cls(
            stage=ctx.stage,
            run_id=ctx.run_id,
            outputs=dict(ctx.outputs),
            metrics=metrics or {},
            notes=notes or [],
        )


class Stage(ABC):
    """One atomic pipeline step."""

    spec: ClassVar[StageSpec]
    settings_model: ClassVar[type[StageSettings]] = StageSettings

    @property
    def name(self) -> str:
        """The stage name."""
        return self.spec.name

    def input_names(self) -> tuple[str, ...]:
        """Names of the artifacts this stage consumes."""
        return tuple(self.spec.inputs)

    def external_input_hashes(self, run: RunContext) -> list[str]:
        """Hashes of non-catalog inputs (e.g. the corpus DB), for the run id."""
        return []

    @abstractmethod
    async def run(self, ctx: StageContext) -> StageResult:
        """Execute the stage and return its result."""
        raise NotImplementedError
