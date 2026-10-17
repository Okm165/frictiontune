"""``fetch_base`` — locate the base model and probe its trainability.

This is a prerequisite utility rather than a data step: it downloads the base
weights if permitted, then records architecture facts that let ``train_sft``
reject impossible jobs (GGUF weights, fused-expert LoRA) before hours of GPU
time are spent.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from frictiontune.backends.models import DEFAULT_TARGET_MODULES
from frictiontune.core.errors import StageError
from frictiontune.core.logging import get_logger
from frictiontune.core.retry import RetryPolicy, with_retry
from frictiontune.domain.base import BaseProbe
from frictiontune.llm.transport import is_transport_retryable
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings

if TYPE_CHECKING:
    from typing import Any

    from frictiontune.pipeline.context import StageContext

_log = get_logger(__name__)


class FetchBaseSettings(StageSettings):
    """Knobs for locating and probing the base model."""

    repo_id: str = ""
    weights_dir: Path | None = None
    allow_download: bool = False


def _first(config: dict[str, object], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = config.get(key)
        if value not in (None, "", 0):
            return value
    return None


def probe_base(repo_dir: Path, repo_id: str) -> BaseProbe:
    """Inspect a local model directory and return a trainability probe."""
    config_path = repo_dir / "config.json"
    if not config_path.is_file():
        raise StageError("fetch_base", f"no config.json in {repo_dir}")

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise StageError("fetch_base", f"unreadable config.json in {repo_dir}: {exc}") from exc
    if not isinstance(raw, dict):
        raise StageError("fetch_base", f"invalid config.json in {repo_dir}")

    architectures = raw.get("architectures") or []
    architecture = str(architectures[0]) if architectures else str(raw.get("model_type", ""))
    model_type = str(raw.get("model_type", ""))

    num_experts = int(_first(raw, ("num_experts", "num_local_experts", "n_routed_experts")) or 0)
    is_moe = num_experts > 0 or "moe" in model_type.lower() or "moe" in architecture.lower()

    quant_config = raw.get("quantization_config")
    quantization = ""
    if isinstance(quant_config, dict):
        quantization = str(quant_config.get("quant_method", "quantized"))
    has_gguf = any(path.suffix == ".gguf" for path in repo_dir.glob("*.gguf"))

    trainable = not has_gguf
    notes: list[str] = []
    if has_gguf:
        notes.append("GGUF weights are not QLoRA-trainable; provide a bf16 HF checkpoint.")
    if is_moe:
        notes.append(
            "MoE detected: keep target_parameters empty so LoRA does not attach to "
            "all fused experts."
        )

    return BaseProbe(
        repo_id=repo_id,
        path=str(repo_dir),
        architecture=architecture,
        model_type=model_type,
        hidden_size=int(_first(raw, ("hidden_size", "n_embd", "d_model")) or 0),
        num_layers=int(_first(raw, ("num_hidden_layers", "n_layer", "num_layers")) or 0),
        vocab_size=int(raw.get("vocab_size") or 0),
        context_length=int(_first(raw, ("max_position_embeddings", "n_ctx", "max_seq_len")) or 0),
        is_moe=is_moe,
        num_experts=num_experts,
        quantization=quantization,
        trainable=trainable,
        target_modules=list(DEFAULT_TARGET_MODULES),
        target_parameters=[],
        notes=notes,
    )


class FetchBase(Stage):
    """Download/locate the base model and emit its probe."""

    spec = StageSpec(
        name="fetch_base",
        tier=Tier.A,
        lead=Lead.CORE,
        roi="High",
        justification=(
            "A trainability guard: blocks invalid GGUF or fused-expert-MoE training "
            "before any GPU time is spent."
        ),
        evidence=["MoE fused-expert LoRA trap", "GGUF cannot be QLoRA-trained"],
        cost=CostProfile(compute="cpu", api_usd="none", maintenance="low"),
        inputs={},
        outputs={"base_probe": "json"},
        kill_criterion=(
            "Base probes become unnecessary if training is delegated to a hosted service."
        ),
    )
    settings_model = FetchBaseSettings

    def _resolve_dir(self, settings: FetchBaseSettings, ctx: StageContext) -> Path:
        if settings.weights_dir is not None:
            return settings.weights_dir
        if ctx.config.paths.base_weights_dir is not None:
            return ctx.config.paths.base_weights_dir
        slug = settings.repo_id.replace("/", "--") or "base"
        return Path(ctx.config.paths.cache_dir) / "base" / slug

    async def run(self, ctx: StageContext) -> StageResult:
        """Resolve the base weights and write the probe."""
        settings = ctx.settings(self.settings_model)
        if not settings.repo_id and settings.weights_dir is None:
            raise StageError("fetch_base", "set repo_id or weights_dir")

        target = self._resolve_dir(settings, ctx)
        if not target.is_dir():
            if not settings.allow_download:
                raise StageError(
                    "fetch_base",
                    f"base weights not found at {target}; set allow_download or weights_dir",
                )
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise StageError(
                    "fetch_base", "install huggingface-hub to enable allow_download"
                ) from exc
            target.mkdir(parents=True, exist_ok=True)
            ctx.logger.info("downloading base weights", repo_id=settings.repo_id)

            async def _download() -> object:
                return await asyncio.to_thread(
                    snapshot_download,
                    repo_id=settings.repo_id,
                    local_dir=str(target),
                )

            await with_retry(
                _download,
                policy=RetryPolicy(max_attempts=3),
                retry_on=is_transport_retryable,
                logger=_log,
            )

        probe = probe_base(target, settings.repo_id)
        ctx.emit_json("base_probe", probe)
        ctx.emit_text(
            "report",
            "# Base probe\n\n"
            f"- repo: `{probe.repo_id}`\n"
            f"- path: `{probe.path}`\n"
            f"- architecture: `{probe.architecture}`\n"
            f"- MoE: {probe.is_moe} (experts: {probe.num_experts})\n"
            f"- quantization: `{probe.quantization or 'none'}`\n"
            f"- trainable: {probe.trainable}\n"
            + "".join(f"- note: {note}\n" for note in probe.notes),
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "trainable": float(probe.trainable),
                "is_moe": float(probe.is_moe),
                "num_experts": float(probe.num_experts),
                "hidden_size": float(probe.hidden_size),
            },
            notes=probe.notes,
        )
