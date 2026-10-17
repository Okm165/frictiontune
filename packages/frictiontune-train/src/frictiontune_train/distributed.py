"""The one definition of training topology: rank, world size, DDP knobs.

A training job runs either as a single process on one card or as a torchrun
data-parallel group across the cards the hardware preset names. Everything that
has to know *which* of those two worlds it is in asks this module, so the answer
cannot drift between the launcher, the loader, and the trainer config:

* :func:`device_map` — single process places the model ``auto`` on its one
  visible card; each DDP rank pins the whole model to *its* card (``auto`` would
  shard it, which breaks TRL's fused loss).
* :func:`config_kwargs` — the trainer arguments DDP needs and single-process
  training must not set.
* :func:`is_main_process` — only rank 0 writes artifacts, so three processes do
  not race on one result file or adapter directory.

This is deterministic plumbing: it reads environment variables torchrun sets and
makes no semantic decision.
"""

from __future__ import annotations

import os
from typing import Any


def world_size() -> int:
    """Number of training processes; ``1`` outside a distributed launch."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    """Global rank of this process; ``0`` outside a distributed launch."""
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    """Index of this process's card within its node."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    """Whether this process is one of several training in parallel."""
    return world_size() > 1


def is_main_process() -> bool:
    """Whether this process owns the artifacts (result file, adapter)."""
    return rank() == 0


def device_map() -> Any:
    """Where to place the base model for this process."""
    # ``auto`` is right for the single-process case: the one visible card. Under
    # DDP every rank sees every card, so pin the whole model to this rank's card.
    return {"": local_rank()} if is_distributed() else "auto"


def config_kwargs() -> dict[str, Any]:
    """Trainer arguments that DDP requires and single-process training must not set."""
    if not is_distributed():
        return {}
    # DDP must know whether every parameter receives a gradient, or it waits for a
    # gradient that never arrives. `find_unused_parameters=True` makes it check,
    # but the check retains the autograd graph across the reduction, costing both
    # time and — at an 8k window on a 24 GB card — the headroom that decides
    # whether the step fits. For LoRA the adapter is attached to every target
    # module and used on every step, so DDP itself reports no unused parameter.
    # The check defaults off; a run that adds a conditional branch, or a MoE
    # adapter that skips experts, turns it back on with
    # FRICTIONTUNE_DDP_FIND_UNUSED=1.
    find_unused = os.environ.get("FRICTIONTUNE_DDP_FIND_UNUSED", "0") != "0"
    return {"ddp_find_unused_parameters": find_unused}
