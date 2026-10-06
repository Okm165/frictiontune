"""The :class:`TrainingBackend` implementation.

The heavy work is synchronous (it drives torch training loops); this adapter
runs it in a worker thread so the pipeline's asyncio event loop is never blocked.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from frictiontune_train.grpo import run_grpo
from frictiontune_train.sft import run_sft

if TYPE_CHECKING:
    from frictiontune.backends.models import GrpoRequest, TrainingRequest, TrainingResult


class LocalTrainingBackend:
    """Runs SFT and GRPO jobs on the local machine's accelerators."""

    async def train(self, request: TrainingRequest) -> TrainingResult:
        """Run a supervised QLoRA job in a worker thread."""
        return await asyncio.to_thread(run_sft, request)

    async def train_grpo(self, request: GrpoRequest) -> TrainingResult:
        """Run a GRPO job in a worker thread."""
        return await asyncio.to_thread(run_grpo, request)
