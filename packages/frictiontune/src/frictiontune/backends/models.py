"""Typed request/response contracts for pluggable backends.

Backends are the boundary between the language-agnostic pipeline and heavy,
environment-specific machinery (training stacks, sandboxes, embedding servers).
Everything crossing this boundary is a pydantic model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from frictiontune.config.models import HardwareConfig


class BackendModel(BaseModel):
    """Common configuration for backend contracts."""

    model_config = ConfigDict(extra="forbid")


class ProbeExample(BackendModel):
    """A single prompt/expected pair used for parity and capability checks."""

    prompt: str
    expected: str | None = None


DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
"""LoRA projection targets for a standard attention/MLP transformer.

One source of truth: ``fetch_base`` advertises these on the probe and
``TrainingRequest`` defaults to them, so a probe can never name a different
decomposition than training applies.
"""


class TrainingRequest(BackendModel):
    """Everything needed to run one supervised fine-tuning job."""

    base_model: str
    dataset_dir: str
    output_dir: str
    hardware: HardwareConfig
    seed: int = 0
    max_seq_len: int = 8192
    learning_rate: float = 2e-4
    epochs: float = 3.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    # MoE trap: an empty list prevents LoRA attaching to all experts.
    target_parameters: list[str] = Field(default_factory=list)
    resume_from: str | None = None
    max_steps: int = -1
    save_steps: int = 100
    eval_dir: str | None = None
    # The Gekhman arm (GAP-096): 1.0 keeps every record; below 1.0, the
    # (1 - known_fraction) records with the HIGHEST loss at base init are
    # dropped before training. Loss at init is the model's own answer to "did
    # it already know this?" — Gekhman et al. (EMNLP 2024) showed Unknown
    # examples fit slowly and learning them linearly increases hallucination.
    # The counterweight is Dataset Cartography (Swayamdipta et al., EMNLP
    # 2020): high-loss examples include the *ambiguous* ones that drive OOD
    # generalization, and hard-to-learn regions often mark labeling errors —
    # which is why this is a measured A/B knob, never a default.
    known_fraction: float = 1.0


class DpoRequest(BackendModel):
    """Everything needed to run one preference-optimization job (GAP-104).

    The job *continues* from the SFT adapter: the pairs teach avoidance on top
    of the behaviour the SFT stage taught, which is the standard SFT-then-DPO
    ordering. The adapter's LoRA configuration is loaded from the adapter
    itself, so the request carries no rank/alpha — those are the SFT run's
    decisions, not the preference run's.
    """

    base_model: str
    adapter_dir: str
    pairs_dir: str
    output_dir: str
    hardware: HardwareConfig
    seed: int = 0
    max_seq_len: int = 8192
    # DPO's learning rates run one to two orders below SFT's: the objective
    # pushes likelihoods apart rather than fitting text.
    learning_rate: float = 1e-5
    beta: float = 0.1
    epochs: float = 1.0
    max_steps: int = -1
    # 0 = derive from max_seq_len (prompt 3/4, completion 1/4).
    max_prompt_length: int = 0
    max_completion_length: int = 0


class TrainingResult(BackendModel):
    """Outcome of a fine-tuning job."""

    adapter_dir: str
    steps: int
    train_loss: float
    final_loss: float
    wall_time_s: float
    metrics: dict[str, float] = Field(default_factory=dict)
    checkpoints: list[str] = Field(default_factory=list)


class GrpoRequest(BackendModel):
    """Everything needed to run a GRPO job against a reward function."""

    base_model: str
    adapter_dir: str | None = None
    prompts_file: str
    output_dir: str
    hardware: HardwareConfig
    seed: int = 0
    learning_rate: float = 1e-5
    max_steps: int = -1
    group_size: int = 8
    reward_weights: dict[str, float] = Field(
        default_factory=lambda: {"judge": 1.0, "format": 0.5, "tool": 0.5}
    )
    max_new_tokens: int = 1_024


class ExportRequest(BackendModel):
    """Merge an adapter into its base or package a LoRA to serve."""

    base_model: str
    adapter_dir: str
    output_dir: str
    method: Literal["merge", "lora"] = "merge"
    probe: list[ProbeExample] = Field(default_factory=list)


class ParityMismatch(BackendModel):
    """One probe whose merged answer differed from the adapter's.

    The two texts are kept because a count alone cannot be read: a merge that
    rounded in bf16 and a merge that is broken both show up as a mismatch, and
    only the outputs say which one happened.
    """

    prompt: str
    expected: str
    actual: str


class ParityResult(BackendModel):
    """Agreement between two model endpoints on a probe set."""

    total: int
    matches: int
    mismatches: list[ParityMismatch] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every probe matched exactly."""
        return self.total > 0 and self.matches == self.total


class ExportResult(BackendModel):
    """Artifacts and parity outcome of an export."""

    served_dir: str
    merged: bool
    parity: ParityResult


class RolloutRequest(BackendModel):
    """A single rollout of a model against a task under a tool sandbox."""

    model: str
    adapter_dir: str | None = None
    task_prompt: str
    max_turns: int = 30
    sandbox: Literal["local", "container", "external"] = "local"
    seed: int = 0


class RolloutStep(BackendModel):
    """One model turn within a rollout."""

    turn: int
    assistant_text: str
    tool_name: str | None = None
    tool_args: dict[str, str] = Field(default_factory=dict)
    tool_result: str | None = None
    tool_error: bool = False


class RolloutResult(BackendModel):
    """A completed rollout transcript and its terminal status."""

    steps: list[RolloutStep]
    finished: bool
    outcome: str | None = None
