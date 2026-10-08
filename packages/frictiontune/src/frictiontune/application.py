"""Application container: wires configuration into a runnable pipeline.

Building the application is the one place objects are constructed: logging, the
content cache, the LLM client, the prompt registry, the gateway, the embedder,
the backend registry, and plugin discovery. Everything else receives them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from frictiontune.backends.embedding import OpenAiEmbedder
from frictiontune.backends.registry import BackendRegistry
from frictiontune.core.cache import ContentCache
from frictiontune.core.logging import LogFormat, LogLevel, configure_logging, get_logger
from frictiontune.llm.client import LlmClient
from frictiontune.llm.gateway import LlmGateway
from frictiontune.llm.prompts.registry import default_registry
from frictiontune.pipeline.context import RunContext
from frictiontune.pipeline.plugins import load_plugins
from frictiontune.pipeline.registry import StageRegistry, validate_stage_settings
from frictiontune.pipeline.registry import default_registry as stage_registry

if TYPE_CHECKING:
    from frictiontune.config.models import RunConfig


@dataclass(slots=True)
class Application:
    """A fully-wired frictiontune process."""

    config: RunConfig
    cache: ContentCache
    client: LlmClient
    gateway: LlmGateway
    backends: BackendRegistry
    stages: StageRegistry
    plugins: list[str] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        config: RunConfig,
        *,
        stages: StageRegistry | None = None,
        log_level: LogLevel = "info",
        log_format: LogFormat = "console",
    ) -> Application:
        """Construct the application from a validated configuration."""
        configure_logging(level=log_level, log_format=log_format)
        logger = get_logger("frictiontune.application")
        cache = ContentCache(config.paths.cache_dir, enabled=True)
        client = LlmClient(cache)
        registry = default_registry()
        gateway = LlmGateway(client, registry)
        backends = BackendRegistry()
        backends.set_embedder(OpenAiEmbedder(config.embedding, cache))
        stage_reg = stages or stage_registry()
        plugins = load_plugins(stage_reg, backends)
        validate_stage_settings(config, stage_reg)
        logger.debug("application built", plugins=plugins, stages=len(stage_reg.names()))
        return cls(
            config=config,
            cache=cache,
            client=client,
            gateway=gateway,
            backends=backends,
            stages=stage_reg,
            plugins=plugins,
        )

    def run_context(self, pipeline: str) -> RunContext:
        """Create a :class:`RunContext` for a named pipeline."""
        from frictiontune.pipeline.context import ArtifactCatalog

        return RunContext(
            config=self.config,
            pipeline=pipeline,
            cache=self.cache,
            gateway=self.gateway,
            backends=self.backends,
            logger=get_logger("frictiontune.run"),
            artifacts_root=self.config.paths.artifacts_dir,
            catalog=ArtifactCatalog(),
        )

    async def aclose(self) -> None:
        """Close network clients."""
        await self.client.aclose()
        embedder = self.backends.embedder_or_none()
        if isinstance(embedder, OpenAiEmbedder):
            await embedder.aclose()
