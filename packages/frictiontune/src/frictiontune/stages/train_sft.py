"""``train_sft`` — run QLoRA supervised fine-tuning on the curated dataset.

The stage itself is thin: it validates the base probe, builds a typed
:class:`TrainingRequest`, and delegates to the registered training backend. That
keeps the heavy ML stack out of the data pipeline and behind one seam. The MoE
trap is handled upstream (``target_parameters`` is empty by default).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from frictiontune.backends.models import TrainingRequest, TrainingResult
from frictiontune.core.errors import StageError
from frictiontune.domain.base import BaseProbe
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings

if TYPE_CHECKING:
    from frictiontune.config.models import HardwareConfig
    from frictiontune.pipeline.context import StageContext


class TrainSftSettings(StageSettings):
    """Hyperparameters for the SFT job (defaults are a POC starting point)."""

    output_name: str = "adapter"
    learning_rate: float = 2e-4
    epochs: float = 3.0
    # The training budget is measured in *optimizer steps*, and on a small
    # curated set "3 epochs" is a handful of them: 23 records / (3 ranks x 4
    # accumulation) is ~2 steps per epoch, so the job would take six updates and
    # stop before the held-out loss has found its minimum. Measured on the real
    # corpus: the minimum fell at epoch 8 (eval loss 1.927) while 3 epochs
    # reached only 2.109. `min_steps` therefore raises the epoch count until the
    # minimum is *reachable*; it is a ceiling, not a target, because the trainer
    # stops early once the held-out loss stops improving and ships the best
    # checkpoint rather than the last. `max_epochs` bounds the ceiling.
    min_steps: int = Field(default=60, ge=0)
    max_epochs: float = Field(default=40.0, ge=1.0)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=list)
    resume_from: str | None = None
    max_steps: int = -1
    save_steps: int = 100


def _epochs_for(n_records: int, hardware: HardwareConfig, settings: TrainSftSettings) -> float:
    """Raise ``epochs`` until the job takes at least ``min_steps`` optimizer steps.

    ``devices`` may be ``auto``; the trainer resolves that itself, so an
    unresolved topology is treated as one rank — which under-estimates the batch
    and therefore over-estimates the epochs needed, never the other way round.
    """
    if settings.min_steps <= 0 or n_records <= 0:
        return settings.epochs
    devices = hardware.devices if isinstance(hardware.devices, int) else 1
    effective_batch = max(1, devices * hardware.per_device_batch * hardware.grad_accum)
    steps_per_epoch = max(1, -(-n_records // effective_batch))
    needed = settings.min_steps / steps_per_epoch
    return min(settings.max_epochs, max(settings.epochs, needed))


class TrainSft(Stage):
    """Fine-tune an adapter on the curated SFT set."""

    spec = StageSpec(
        name="train_sft",
        tier=Tier.A,
        lead=Lead.CORE,
        roi="High",
        justification="The core deliverable: a model that needs less steering.",
        evidence=["QLoRA; target_parameters=[] for MoE; loss on completion only"],
        cost=CostProfile(compute="gpu", api_usd="none", human="none", maintenance="high"),
        inputs={"train": "jsonl", "eval": "jsonl", "base_probe": "json"},
        outputs={"training_result": "json", "adapter": "json"},
        kill_criterion=(
            "Drop the full run if the friction drop is within noise (see evaluate_replay)."
        ),
    )
    settings_model = TrainSftSettings

    async def run(self, ctx: StageContext) -> StageResult:
        """Validate inputs and delegate to the training backend."""
        settings = ctx.settings(self.settings_model)
        probe = BaseProbe.model_validate(ctx.catalog.read_json("base_probe"))
        if not probe.trainable:
            raise StageError(
                "train_sft",
                f"base model {probe.repo_id!r} is not trainable ({'; '.join(probe.notes)})",
            )

        train_entry = ctx.catalog.get("train")
        dataset_dir = str(Path(train_entry.path).parent)
        eval_entry = ctx.catalog.get("eval")
        eval_dir = str(Path(eval_entry.path).parent)
        output_dir = str(ctx.store.directory / settings.output_name)
        epochs = _epochs_for(train_entry.ref.rows, ctx.config.hardware, settings)

        request = TrainingRequest(
            base_model=probe.path or probe.repo_id,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            hardware=ctx.config.hardware,
            seed=ctx.config.seed,
            max_seq_len=ctx.config.hardware.max_seq_len,
            learning_rate=settings.learning_rate,
            epochs=epochs,
            lora_r=settings.lora_r,
            lora_alpha=settings.lora_alpha,
            lora_dropout=settings.lora_dropout,
            target_modules=settings.target_modules or probe.target_modules,
            target_parameters=probe.target_parameters,
            resume_from=settings.resume_from,
            max_steps=settings.max_steps,
            save_steps=settings.save_steps,
            eval_dir=eval_dir,
        )

        backend = ctx.run.backends.training()
        result: TrainingResult = await backend.train(request)
        ctx.emit_json("training_result", result)
        ctx.emit_json(
            "adapter",
            {
                "base_model": request.base_model,
                "adapter_dir": result.adapter_dir,
                "steps": result.steps,
                "train_loss": result.train_loss,
                "final_loss": result.final_loss,
                "checkpoints": result.checkpoints,
            },
        )
        ctx.emit_text(
            "report",
            "# Train SFT\n\n"
            f"- base: `{request.base_model}`\n"
            f"- steps: {result.steps}\n"
            f"- epochs: {epochs:g} (derived: at least {settings.min_steps} optimizer "
            f"steps over {train_entry.ref.rows} records)\n"
            f"- train loss: {result.train_loss:.4f}\n"
            f"- final loss: {result.final_loss:.4f}\n"
            f"- wall time: {result.wall_time_s:.1f}s\n"
            f"- adapter: `{result.adapter_dir}`\n",
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "steps": float(result.steps),
                "train_loss": result.train_loss,
                "final_loss": result.final_loss,
                "wall_time_s": result.wall_time_s,
            },
        )
