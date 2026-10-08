"""Layered configuration loading.

Resolution order (later wins), all shallow-merging into one tree:

1. ``configs/base.yaml``               — defaults
2. ``configs/roles.yaml``              — LLM role bindings
3. ``configs/friction_taxonomy.yaml``  — friction axes and leaves
4. ``configs/hardware/<preset>.yaml``  — hardware preset
5. ``configs/environments/<env>.yaml`` — optional environment overlay
6. ``configs/stages/*.yaml``           — per-stage knobs
7. explicit overrides (e.g. ``--set``)

Strings may reference environment variables as ``${VAR}`` or ``${VAR:-default}``
so that no machine-specific values are baked into the repo.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import yaml

from frictiontune.config.models import RunConfig
from frictiontune.core.errors import ConfigError

if TYPE_CHECKING:
    from pathlib import Path

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path`` or raise :class:`ConfigError`."""
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"expected a mapping at the top level of {path}")
    return raw


def expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` references inside strings."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(
            lambda match: os.environ.get(match.group(1), match.group(2) or ""), value
        )
    if isinstance(value, Mapping):
        return {str(key): expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


def deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into a copy of ``base``."""
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _collect_stage_configs(config_dir: Path) -> dict[str, Any]:
    stages_dir = config_dir / "stages"
    if not stages_dir.is_dir():
        return {}
    collected: dict[str, Any] = {}
    for path in sorted(stages_dir.glob("*.yaml")):
        collected[path.stem] = load_yaml(path)
    return collected


def _prune_empty(value: Any) -> Any:
    """Drop mapping entries whose value is empty or ``None``.

    This is what makes ``${VAR:-}`` behave like an absent setting rather than an
    empty string, so optional paths (like ``corpus_db``) keep their ``None``
    default instead of collapsing to ``.``.
    """
    if isinstance(value, Mapping):
        pruned: dict[str, Any] = {}
        for key, item in value.items():
            cleaned = _prune_empty(item)
            if cleaned is None or cleaned == "":
                continue
            pruned[str(key)] = cleaned
        return pruned
    if isinstance(value, list):
        return [_prune_empty(item) for item in value]
    return value


def load_config(
    config_dir: Path,
    *,
    hardware_preset: str | None = None,
    environment: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    extra_files: list[Path] | None = None,
) -> RunConfig:
    """Build a validated :class:`RunConfig` from the layered config tree."""
    if not config_dir.is_dir():
        raise ConfigError(f"config directory not found: {config_dir}")

    tree: dict[str, Any] = {}

    base = config_dir / "base.yaml"
    if base.is_file():
        tree = deep_merge(tree, load_yaml(base))

    roles_path = config_dir / "roles.yaml"
    if roles_path.is_file():
        roles = load_yaml(roles_path)
        tree = deep_merge(tree, {"llm": roles.get("llm", roles)})

    taxonomy_path = config_dir / "friction_taxonomy.yaml"
    if taxonomy_path.is_file():
        tree = deep_merge(tree, {"taxonomy": load_yaml(taxonomy_path)})

    if hardware_preset:
        preset_path = config_dir / "hardware" / f"{hardware_preset}.yaml"
        tree = deep_merge(tree, {"hardware": load_yaml(preset_path)})

    if environment:
        env_path = config_dir / "environments" / f"{environment}.yaml"
        tree = deep_merge(tree, load_yaml(env_path))

    stage_configs = _collect_stage_configs(config_dir)
    if stage_configs:
        tree = deep_merge(tree, {"stages": stage_configs})

    for extra in extra_files or []:
        tree = deep_merge(tree, load_yaml(extra))

    if overrides:
        tree = deep_merge(tree, overrides)

    tree.pop("environment", None)
    tree = _prune_empty(expand_env(tree))

    try:
        return RunConfig.model_validate(tree)
    except Exception as exc:  # pydantic ValidationError and friends
        raise ConfigError(f"invalid configuration: {exc}") from exc


def parse_set_overrides(items: list[str]) -> dict[str, Any]:
    """Parse ``a.b.c=value`` strings into a nested override mapping.

    Values are parsed as YAML scalars, so ``stages.curate.near_dedup=false``
    yields a boolean and ``stages.judge.temperature=0.9`` yields a float.
    """
    root: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        if not keys or any(not key for key in keys):
            raise ConfigError(f"invalid --set key: {dotted!r}")
        cursor = root
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
            if not isinstance(cursor, dict):
                raise ConfigError(f"--set key conflict at {key!r}")
        cursor[keys[-1]] = yaml.safe_load(raw)
    return root
