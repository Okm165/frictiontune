"""Tests for the layered configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from frictiontune.config.loader import load_config, parse_set_overrides
from frictiontune.config.models import RoleName
from frictiontune.core.errors import ConfigError
from frictiontune.pipeline.registry import default_registry, validate_stage_settings


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
    form, cause = config.taxonomy.leaves["instruction_misread"]
    assert form in config.taxonomy.forms
    assert cause in config.taxonomy.causes


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


def test_stage_settings_accept_known_knobs(config_dir: Path) -> None:
    config = load_config(config_dir, overrides={"stages": {"curate": {"near_dedup": False}}})

    validate_stage_settings(config, default_registry())


def test_stage_settings_reject_unknown_knob(config_dir: Path) -> None:
    config = load_config(config_dir, overrides={"stages": {"curate": {"near_dedup_typo": True}}})

    with pytest.raises(ConfigError, match="invalid config for stage 'curate'"):
        validate_stage_settings(config, default_registry())
