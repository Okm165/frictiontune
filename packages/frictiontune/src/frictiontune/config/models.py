"""Configuration models.

Everything that varies by machine, model, or research choice is configuration.
The models here are strict: unknown keys are rejected so typos surface at load
time rather than mid-run.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from frictiontune.core.errors import ConfigError
from frictiontune.core.json import JsonObject
from frictiontune.core.retry import RetryPolicy

if TYPE_CHECKING:
    from collections.abc import Collection


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
    HARVESTER = "harvester"
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
    # Provider-specific request-body fields that the OpenAI-compatible surface
    # does not standardise: chain-of-thought control
    # (``{"chat_template_kwargs": {"enable_thinking": false}}`` on vLLM/SGLang,
    # ``{"reasoning_effort": "low"}`` on OpenAI-style servers), grammar options,
    # or vendor flags. Merged verbatim into the request body and folded into the
    # cache key, so the same prompt sent with different provider knobs is a
    # different request. It lives in configuration — never hard-coded per
    # provider in the transport — because reasoning control is a research choice
    # with no portable spelling.
    extra_body: JsonObject = Field(default_factory=dict)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)


class EmbeddingConfig(BaseConfigModel):
    """The embedding provider used for retrieval, calibration, and clustering."""

    provider: str = "openai_compatible"
    base_url: str | None = None
    model: str = "text-embedding-3-small"
    api_key_env: str | None = None
    dimensions: int | None = None
    # Inputs longer than any embedder's context are a hard provider error (a
    # generic 400), not a silent truncation, so the backend bounds each text.
    # 8000 chars stays under an ~8192-token context even for ~1-token-per-char
    # scripts (punctuation, CJK, emoji); raise it for a known low-density corpus.
    max_input_chars: int = Field(default=8_000, ge=1)
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
    per_device_batch: int = Field(default=1, ge=1)
    grad_accum: int = Field(default=8, ge=1)
    max_seq_len: int = Field(default=8192, ge=128)
    gradient_checkpointing: bool = True
    dtype: Literal["bf16", "fp16"] = "bf16"


class LeafAxis(BaseConfigModel):
    """One friction leaf and the ``(form, cause)`` pair it maps to.

    ``description`` is the semantic definition injected into the analyst prompt.
    Without it the model would have to label from the bare leaf name, which lets
    the same situation drift across labels between runs.
    """

    form: str
    cause: str
    description: str = ""


class TaxonomyConfig(BaseConfigModel):
    """The SWE-chat friction axes and our leaf labels."""

    forms: list[str]
    causes: list[str]
    costs: list[str]
    resolutions: list[str]
    leaves: dict[str, LeafAxis] = Field(default_factory=dict)

    @field_validator("leaves")
    @classmethod
    def _validate_leaves(
        cls, value: dict[str, LeafAxis], info: ValidationInfo
    ) -> dict[str, LeafAxis]:
        forms = info.data.get("forms", [])
        causes = info.data.get("causes", [])
        for leaf, axis in value.items():
            if not leaf or not axis.form or not axis.cause:
                raise ValueError("taxonomy leaves must be non-empty (leaf, form, cause)")
            if axis.form not in forms:
                raise ValueError(f"taxonomy leaf {leaf!r} references unknown form {axis.form!r}")
            if axis.cause not in causes:
                raise ValueError(f"taxonomy leaf {leaf!r} references unknown cause {axis.cause!r}")
        return value


class EvaluationConfig(BaseConfigModel):
    """Contamination and evaluation policy."""

    time_split_quantile: float = Field(default=0.8, gt=0.0, lt=1.0)
    # The pre-registered minimal important difference (GAP-102): the smallest
    # rubric delta the pipeline will treat as a real effect. Declaring it here
    # — rather than after seeing a result — is what makes a powered
    # measurement conclusive; the dashboard flags deltas below it as "measured,
    # below the declared bar". 0.5 is half a rubric point: with the judge's
    # measured compression onto {4,5} (GAP-099), smaller deltas are within the
    # instrument's own wobble. Changing this re-keys the evaluation stages,
    # which is correct: it is a change of instrument, not of tuning.
    min_important_delta: float = Field(default=0.5, ge=0.0)
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

    def semantic_fingerprint(
        self,
        *,
        stage: str | None = None,
        roles: Collection[RoleName] | None = None,
        uses_evaluation: bool = True,
    ) -> JsonObject:
        """The part of the configuration that can change what a stage produces.

        A run id must be stable across *phases* of one pipeline. Every phase
        starts a different set of servers — the data phase runs the in-pod
        llama.cpp replicas, training stops them all to take the GPUs, the replay
        phase serves the base and adapter — so the resolved config differs by
        phase even though nothing a stage *decides* has changed. Fingerprinting
        the whole config made every phase re-key every stage, so the stage cache
        could never hit across a phase boundary and each phase rebuilt the
        pipeline it was supposed to consume.

        ``stage`` narrows the stage settings to that stage's own block,
        ``roles`` to the roles that stage calls (``None`` keeps every role, the
        conservative default for a stage that does not declare them), and
        ``uses_evaluation`` includes the evaluation policy only for the stages
        that read it — leaving it global meant changing a *measurement* policy
        re-keyed the whole data phase, so the one experiment that improves the
        instrument cost a full rebuild.

        Three kinds of field are excluded because they cannot change a decision:
        where a model is served (``base_url``, ``api_key_env``), how many calls a
        role may have in flight (``concurrency`` — a scheduling knob, and the one
        that differs whenever a phase runs against a different number of
        replicas), and the embedding endpoint. Everything that *can* change a
        decision is kept: the stage's own settings, each declared role's model,
        sampling knobs, token budget, cache flag and timeout, the taxonomy, the
        evaluation policy, the seed, and the hardware.
        """
        transport = {"base_url", "api_key_env", "concurrency"}
        selected = sorted(self.llm, key=lambda role: role.value) if roles is None else sorted(roles)
        return {
            "project": self.project,
            "seed": self.seed,
            "hardware": self.hardware.model_dump(mode="json"),
            "taxonomy": self.taxonomy.model_dump(mode="json"),
            "evaluation": self.evaluation.model_dump(mode="json") if uses_evaluation else {},
            "stages": {stage: self.stages.get(stage, {})}
            if stage is not None
            else dict(self.stages),
            "llm": {
                name.value: self.llm[name].model_dump(mode="json", exclude=transport)
                for name in selected
                if name in self.llm
            },
            "embedding": self.embedding.model_dump(
                mode="json", exclude={"base_url", "api_key_env"}
            ),
        }
