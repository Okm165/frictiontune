"""Tests for the layered configuration loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from frictiontune.config.loader import (
    expand_env,
    load_config,
    load_env_file,
    parse_set_overrides,
)
from frictiontune.config.models import LeafAxis, RoleName, TaxonomyConfig
from frictiontune.core.errors import ConfigError
from frictiontune.pipeline.registry import default_registry, validate_stage_settings
from pydantic import ValidationError


def test_load_env_file_reads_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / ".env"
    path.write_text("FRICTIONTUNE_TEST_LOADED=hello\n", encoding="utf-8")
    monkeypatch.delenv("FRICTIONTUNE_TEST_LOADED", raising=False)

    assert load_env_file(path) == path
    assert os.environ["FRICTIONTUNE_TEST_LOADED"] == "hello"


def test_load_env_file_does_not_override_real_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ".env"
    path.write_text("FRICTIONTUNE_TEST_KEEP=from-file\n", encoding="utf-8")
    monkeypatch.setenv("FRICTIONTUNE_TEST_KEEP", "from-env")

    load_env_file(path)

    assert os.environ["FRICTIONTUNE_TEST_KEEP"] == "from-env"


def test_load_env_file_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_env_file(tmp_path / "absent.env") is None


def test_load_config_with_hardware_preset(config_dir: Path) -> None:
    config = load_config(config_dir, hardware_preset="3x24g")

    assert config.hardware.devices == 3
    assert config.hardware.quant == "nf4"
    assert RoleName.JUDGE in config.llm
    assert RoleName.TARGET in config.llm
    assert RoleName.ADAPTER in config.llm


def test_taxonomy_axes_and_leaves_are_loaded(config_dir: Path) -> None:
    config = load_config(config_dir)

    assert "instruction_misread" in config.taxonomy.leaves
    assert len(config.taxonomy.leaves) >= 15
    axis = config.taxonomy.leaves["instruction_misread"]
    assert axis.form in config.taxonomy.forms
    assert axis.cause in config.taxonomy.causes
    assert axis.description


def _taxonomy(**leaves: tuple[str, str]) -> TaxonomyConfig:
    return TaxonomyConfig(
        forms=["known-form"],
        causes=["known-cause"],
        costs=[],
        resolutions=[],
        leaves={name: LeafAxis(form=form, cause=cause) for name, (form, cause) in leaves.items()},
    )


def test_taxonomy_leaf_must_reference_a_declared_form() -> None:
    with pytest.raises(ValidationError, match="unknown form"):
        _taxonomy(x=("mystery-form", "known-cause"))


def test_taxonomy_leaf_must_reference_a_declared_cause() -> None:
    with pytest.raises(ValidationError, match="unknown cause"):
        _taxonomy(x=("known-form", "mystery-cause"))


def test_taxonomy_consistent_leaves_are_accepted() -> None:
    taxonomy = _taxonomy(x=("known-form", "known-cause"))

    assert taxonomy.leaves == {"x": LeafAxis(form="known-form", cause="known-cause")}


def test_empty_env_expansion_drops_key(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRICTIONTUNE_CORPUS_DB", raising=False)

    config = load_config(config_dir)

    assert config.paths.corpus_db is None
    assert config.paths.base_weights_dir is None


def test_set_env_var_becomes_a_path(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRICTIONTUNE_CORPUS_DB", "/tmp/sessions.db")

    config = load_config(config_dir)

    assert config.paths.corpus_db == Path("/tmp/sessions.db")


def test_provided_env_does_not_override_explicit_override(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FRICTIONTUNE_CORPUS_DB", "/tmp/from-env.db")

    config = load_config(config_dir, overrides={"paths": {"corpus_db": "/tmp/explicit.db"}})

    assert config.paths.corpus_db == Path("/tmp/explicit.db")


def test_unknown_key_is_rejected(config_dir: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(config_dir, overrides={"not_a_real_key": 1})


def test_hardware_can_be_overridden(config_dir: Path) -> None:
    config = load_config(
        config_dir, hardware_preset="1x24g", overrides={"hardware": {"devices": 2}}
    )
    assert config.hardware.devices == 2


def test_parse_set_overrides_nests_and_coerces_types() -> None:
    parsed = parse_set_overrides(
        [
            "hardware.devices=2",
            "evaluation.time_split_quantile=0.7",
            "stages.curate.near_dedup=false",
        ]
    )

    assert parsed == {
        "hardware": {"devices": 2},
        "evaluation": {"time_split_quantile": 0.7},
        "stages": {"curate": {"near_dedup": False}},
    }


def test_parse_set_overrides_rejects_malformed() -> None:
    with pytest.raises(ConfigError):
        parse_set_overrides(["no_equals_sign"])


def test_missing_role_raises_config_error(config_dir: Path) -> None:
    config = load_config(config_dir)
    trimmed = config.model_copy(update={"llm": {}})

    with pytest.raises(ConfigError, match="not configured"):
        trimmed.role(RoleName.JUDGE)


def test_env_default_applies_when_variable_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRICTIONTUNE_TEST_URL", "")

    assert expand_env("${FRICTIONTUNE_TEST_URL:-http://fallback}") == "http://fallback"


def test_env_value_wins_over_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRICTIONTUNE_TEST_URL", "http://set")

    assert expand_env("${FRICTIONTUNE_TEST_URL:-http://fallback}") == "http://set"


def test_env_without_default_is_empty_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRICTIONTUNE_TEST_URL", raising=False)

    assert expand_env("${FRICTIONTUNE_TEST_URL}") == ""


def test_stage_settings_accept_known_knobs(config_dir: Path) -> None:
    config = load_config(config_dir, overrides={"stages": {"curate": {"near_dedup": False}}})

    validate_stage_settings(config, default_registry())


def test_stage_settings_reject_unknown_knob(config_dir: Path) -> None:
    config = load_config(config_dir, overrides={"stages": {"curate": {"near_dedup_typo": True}}})

    with pytest.raises(ConfigError, match="invalid config for stage 'curate'"):
        validate_stage_settings(config, default_registry())


@pytest.mark.parametrize(
    ("stage", "knob"),
    [
        ("curate", "embed_batch"),
        ("decontaminate", "embed_batch"),
        ("reduce", "chunk_turns"),
        ("build_sft", "max_seq_len"),
    ],
)
def test_zero_valued_loop_bounds_are_rejected_up_front(
    config_dir: Path, stage: str, knob: str
) -> None:
    config = load_config(config_dir, overrides={"stages": {stage: {knob: 0}}})

    with pytest.raises(ConfigError, match=f"invalid config for stage '{stage}'"):
        validate_stage_settings(config, default_registry())


def test_semantic_fingerprint_ignores_where_models_are_served(config_dir: Path) -> None:
    base = load_config(config_dir)
    moved = base.model_copy(
        deep=True,
        update={
            "embedding": base.embedding.model_copy(
                update={"base_url": "http://elsewhere.invalid:9998/v1"}
            )
        },
    )
    for name, role in base.llm.items():
        moved.llm[name] = role.model_copy(update={"base_url": "http://elsewhere.invalid:9999/v1"})

    # Each phase of a pipeline starts a different set of servers, so endpoints
    # differ by phase while nothing a stage decides has changed. If they were in
    # the fingerprint, every phase would re-key every stage and the stage cache
    # could never be hit across a phase boundary.
    assert moved.semantic_fingerprint() == base.semantic_fingerprint()


def test_semantic_fingerprint_notices_a_change_that_can_move_a_decision(
    config_dir: Path,
) -> None:
    base = load_config(config_dir)
    retuned = base.model_copy(deep=True)
    retuned.llm[RoleName.JUDGE] = retuned.llm[RoleName.JUDGE].model_copy(
        update={"temperature": 0.9}
    )
    uncapped = base.model_copy(deep=True)
    uncapped.llm[RoleName.JUDGE] = uncapped.llm[RoleName.JUDGE].model_copy(
        update={"max_tokens": 4096}
    )

    assert retuned.semantic_fingerprint() != base.semantic_fingerprint()
    assert uncapped.semantic_fingerprint() != base.semantic_fingerprint()
