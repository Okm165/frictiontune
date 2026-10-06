"""Built-in stages and their registration.

Core stages form the default ROI-justified pipeline. Extension stages are
registered too, but the orchestrator gates them: a Tier C stage only runs when
``allow_gated`` is set. Registration is split so callers can build a core-only
registry if they wish.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from frictiontune.stages._extensions import register_extension_stages
from frictiontune.stages.build_sft import BuildSft
from frictiontune.stages.compile_rules import CompileRules
from frictiontune.stages.curate import Curate
from frictiontune.stages.decontaminate import Decontaminate
from frictiontune.stages.evaluate_replay import EvaluateReplay
from frictiontune.stages.export import Export
from frictiontune.stages.fetch_base import FetchBase
from frictiontune.stages.friction_detect import FrictionDetect
from frictiontune.stages.ingest import Ingest
from frictiontune.stages.judge import Judge
from frictiontune.stages.mine import Mine
from frictiontune.stages.normalize import Normalize
from frictiontune.stages.segment import Segment
from frictiontune.stages.train_sft import TrainSft
from frictiontune.stages.verify_rules import VerifyRules

if TYPE_CHECKING:
    from frictiontune.pipeline.registry import StageRegistry

# The core pipeline, in rough dependency order. The orchestrator re-derives the
# true order from the DAG, so this order is documentation, not control flow.
CORE_STAGES: tuple[type, ...] = (
    FetchBase,
    Ingest,
    Normalize,
    Segment,
    FrictionDetect,
    Mine,
    CompileRules,
    VerifyRules,
    Judge,
    BuildSft,
    Curate,
    Decontaminate,
    TrainSft,
    EvaluateReplay,
    Export,
)


def register_core_stages(registry: StageRegistry) -> None:
    """Register the core stages on ``registry``.

    Extension stages are registered as well so a gated run can reach them; the
    orchestrator enforces the gate, not the registry.
    """
    for stage_cls in CORE_STAGES:
        registry.register(stage_cls)
    register_extension_stages(registry)


__all__ = ["CORE_STAGES", "register_core_stages"]
