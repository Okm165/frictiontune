"""Plugin entry point: wire the GPU-side backends into a run.

Loaded through the ``frictiontune.plugins`` entry-point group. Importing this
module stays light: the concrete backends below import torch/transformers only
inside the calls that need them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from frictiontune.core.logging import get_logger

from frictiontune_train.peft_export import LocalServingBackend
from frictiontune_train.rollout import HttpRolloutBackend
from frictiontune_train.training import LocalTrainingBackend

if TYPE_CHECKING:
    from frictiontune.backends.registry import BackendRegistry
    from frictiontune.pipeline.registry import StageRegistry

_log = get_logger("frictiontune_train.plugin")


def register(registry: StageRegistry, backends: BackendRegistry) -> None:
    """Register training, serving, and rollout backends.

    The caller (the application) owns the embedder; this plugin never overrides
    it. Stages are unchanged: the plugin only supplies implementations for the
    protocols the core declares.
    """
    del registry  # the core already registers every stage
    backends.set_training(LocalTrainingBackend())
    backends.set_serving(LocalServingBackend())
    backends.set_rollout(HttpRolloutBackend())
    _log.debug("frictiontune-train backends registered")
