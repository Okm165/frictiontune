"""Configuration models.

Everything that varies by machine, model, or research choice is configuration.
The models here are strict: unknown keys are rejected so typos surface at load
time rather than mid-run.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from frictiontune.core.errors import ConfigError
from frictiontune.core.json import JsonObject
from frictiontune.core.retry import RetryPolicy


class BaseConfigModel(BaseModel):
    """Strict base for all configuration models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RoleName(StrEnum):
    """The narrow, typed LLM roles used across the pipeline."""

    TARGET = "target"
    ADAPTER = "adapter"
    ROUTER = "router"
    COMPACTOR = "compactor"
    SCHEMA_INFERER = "schema_inferer"
    FRICTION_ANALYST = "friction_analyst"
    RECOVERY_FINDER = "recovery_finder"
    ATTRIBUTOR = "attributor"
    RULE_COMPILER = "rule_compiler"
    RULE_VERIFIER = "rule_verifier"
    JUDGE = "judge"
    DEDUP = "dedup"
    TEACHER = "teacher"
    USER_SIMULATOR = "user_simulator"


class LlmRoleConfig(BaseConfigModel):
    """One provider-pluggable, typed LLM binding."""

    provider: str = "openai_compatible"
    base_url: str | None = None
    model: str
    api_key_env: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    concurrency: int = Field(default=4, ge=1)
    cache: bool = True
    timeout_s: float = Field(default=120.0, gt=0.0)
    response_format: Literal["json_object", "json_schema", "text"] = "json_object"
    retry: RetryPolicy = Field(default_factory=RetryPolicy)


class EmbeddingConfig(BaseConfigModel):
    """The embedding provider used for retrieval, calibration, and clustering."""

    provider: str = "openai_compatible"
    base_url: str | None = None
    model: str = "text-embedding-3-small"
    api_key_env: str | None = None
    dimensions: int | None = None
    cache: bool = True
    timeout_s: float = Field(default=60.0, gt=0.0)


class PathsConfig(BaseConfigModel):
    """Filesystem locations. Nothing here is machine-specific by default."""

    artifacts_dir: Path = Path("artifacts")
    cache_dir: Path = Path(".cache/frictiontune")
    corpus_db: Path | None = None
    base_weights_dir: Path | None = None


class HardwareConfig(BaseConfigModel):
    """Hardware/training knobs; no GPU count is ever hard-coded."""

    devices: int | Literal["auto"] = "auto"
    quant: Literal["nf4", "fp4", "none"] = "nf4"
    offload: Literal["none", "cpu", "zero3"] = "none"
    per_device_batch: int = Field(default=1, ge=1)
    grad_accum: int = Field(default=8, ge=1)
    max_seq_len: int = Field(default=8192, ge=128)
    gradient_checkpointing: bool = True
    dtype: Literal["bf16", "fp16"] = "bf16"


class TaxonomyConfig(BaseConfigModel):
    """The SWE-chat friction axes and our leaf labels."""

    forms: list[str]
    causes: list[str]
    costs: list[str]
    resolutions: list[str]
    leaves: dict[str, tuple[str, str]] = Field(default_factory=dict)

    @field_validator("leaves")
    @classmethod
    def _validate_leaves(
        cls, value: dict[str, tuple[str, str]], info: ValidationInfo
    ) -> dict[str, tuple[str, str]]:
        forms = info.data.get("forms", [])
        causes = info.data.get("causes", [])
        for leaf, (form, cause) in value.items():
            if not leaf or not form or not cause:
                raise ValueError("taxonomy leaves must be non-empty (leaf, form, cause)")
            if form not in forms:
                raise ValueError(f"taxonomy leaf {leaf!r} references unknown form {form!r}")
            if cause not in causes:
                raise ValueError(f"taxonomy leaf {leaf!r} references unknown cause {cause!r}")
        return value


class EvaluationConfig(BaseConfigModel):
    """Contamination and evaluation policy."""

    time_split_quantile: float = Field(default=0.8, gt=0.0, lt=1.0)
    t1_exact_ngram: bool = True
    t2_syntactic: bool = True
    t3_semantic: bool = True
    t4_task_level: bool = True
    ngram_size: int = Field(default=8, ge=1)
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    external_corpora: list[str] = Field(default_factory=list)


class RunConfig(BaseConfigModel):
    """The complete, validated configuration for a run."""

    project: str = "frictiontune"
    seed: int = 0
    paths: PathsConfig = Field(default_factory=PathsConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    llm: dict[RoleName, LlmRoleConfig] = Field(default_factory=dict)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    taxonomy: TaxonomyConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    stages: dict[str, JsonObject] = Field(default_factory=dict)

    def role(self, name: RoleName) -> LlmRoleConfig:
        """Return the configuration for ``name`` or raise a clear error.

        A missing role is a configuration error, not a bare ``KeyError``: it is
        raised as :class:`ConfigError` so a run fails with a typed, catchable
        message (and ``frictiontune validate`` can surface it) rather than an
        unhandled traceback in the middle of a stage.
        """
        try:
            return self.llm[name]
        except KeyError as exc:
            raise ConfigError(f"role {name.value!r} is not configured") from exc
