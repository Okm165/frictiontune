"""Plugin loading via entry points.

Workspace members (e.g. ``frictiontune-train``) register extra stages and
backends by exposing a module with a ``register(registry, backends)`` function
under the ``frictiontune.plugins`` entry-point group. Loading is best-effort and
never required for the core pipeline.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from frictiontune.core.logging import get_logger

if TYPE_CHECKING:
    from frictiontune.backends.registry import BackendRegistry
    from frictiontune.pipeline.registry import StageRegistry

_log = get_logger(__name__)

PLUGIN_GROUP = "frictiontune.plugins"


def load_plugins(registry: StageRegistry, backends: BackendRegistry) -> list[str]:
    """Load every installed plugin, returning the names that loaded."""
    loaded: list[str] = []
    for entry_point in entry_points(group=PLUGIN_GROUP):
        try:
            module = entry_point.load()
            register = getattr(module, "register", None)
            if not callable(register):
                _log.warning("plugin has no register()", plugin=entry_point.name)
                continue
            register(registry, backends)
            loaded.append(entry_point.name)
            _log.debug("plugin loaded", plugin=entry_point.name)
        except Exception as exc:  # a broken plugin must not kill the pipeline
            _log.error("plugin failed to load", plugin=entry_point.name, error=str(exc))
    return loaded
