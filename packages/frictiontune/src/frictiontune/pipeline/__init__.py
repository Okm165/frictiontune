"""Pipeline: stage contracts, registry, DAG, context, orchestrator, plugins."""

from frictiontune.pipeline.context import (
    ArtifactCatalog,
    CatalogEntry,
    RunContext,
    StageContext,
)
from frictiontune.pipeline.dag import (
    CORE_EDGES,
    EXTENSION_EDGES,
    DependencyGraph,
    core_graph,
    full_graph,
)
from frictiontune.pipeline.manifest import (
    RunManifest,
    StageManifest,
    code_revision,
    utc_now,
)
from frictiontune.pipeline.orchestrator import Orchestrator
from frictiontune.pipeline.plugins import load_plugins
from frictiontune.pipeline.registry import StageRegistry, default_registry, register
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings

__all__ = [
    "CORE_EDGES",
    "EXTENSION_EDGES",
    "ArtifactCatalog",
    "CatalogEntry",
    "CostProfile",
    "DependencyGraph",
    "Lead",
    "Orchestrator",
    "RunContext",
    "RunManifest",
    "Stage",
    "StageContext",
    "StageManifest",
    "StageRegistry",
    "StageResult",
    "StageSettings",
    "StageSpec",
    "Tier",
    "code_revision",
    "core_graph",
    "default_registry",
    "full_graph",
    "load_plugins",
    "register",
    "utc_now",
]
