"""Stage registry: the single catalogue of executable stages."""

from __future__ import annotations

from functools import lru_cache
from typing import TypeVar

from frictiontune.core.errors import ConfigError
from frictiontune.pipeline.stage import Stage

StageT = TypeVar("StageT", bound=type[Stage])


class StageRegistry:
    """Holds every registered stage instance, keyed by name."""

    def __init__(self) -> None:
        """Create an empty stage registry."""
        self._stages: dict[str, Stage] = {}

    def register(self, stage_cls: StageT) -> StageT:
        """Register a stage class (instantiating it) and return the class."""
        instance = stage_cls()
        name = instance.name
        if name in self._stages:
            raise ConfigError(f"duplicate stage registration: {name!r}")
        self._stages[name] = instance
        return stage_cls

    def get(self, name: str) -> Stage:
        """Return a registered stage by name or raise."""
        try:
            return self._stages[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._stages)) or "(none)"
            raise ConfigError(f"unknown stage {name!r}; known: {known}") from exc

    def names(self) -> list[str]:
        """Every registered stage name."""
        return sorted(self._stages)

    def all(self) -> list[Stage]:
        """Every registered stage instance, ordered by name."""
        return [self._stages[name] for name in self.names()]

    def core(self) -> list[Stage]:
        """Every stage belonging to the default core run."""
        return [stage for stage in self.all() if stage.spec.core]


@lru_cache(maxsize=1)
def default_registry() -> StageRegistry:
    """Return the process-wide stage registry, importing built-in stages once."""
    registry = StageRegistry()
    _load_builtin_stages(registry)
    return registry


def register(stage_cls: StageT) -> StageT:
    """Decorator: register ``stage_cls`` on the default registry."""
    return default_registry().register(stage_cls)


def _load_builtin_stages(registry: StageRegistry) -> None:
    """Import the built-in stage modules for their registration side effects."""
    from frictiontune.stages import register_core_stages

    register_core_stages(registry)
