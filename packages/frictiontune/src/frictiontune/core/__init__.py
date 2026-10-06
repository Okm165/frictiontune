"""Core primitives: the deterministic plumbing shared by every stage.

Nothing in this package makes a semantic judgment. It handles IO, parsing,
hashing, caching, retrying, logging, and machine artifacts.
"""

from frictiontune.core.artifacts import ArtifactStore, DatasetRef
from frictiontune.core.cache import ContentCache
from frictiontune.core.errors import (
    ArtifactError,
    BackendUnavailableError,
    ConfigError,
    FrictionTuneError,
    GateError,
    IngestError,
    LlmError,
    LlmSchemaError,
    StageError,
    UnsupportedOperationError,
    ValidationError,
)
from frictiontune.core.hashing import (
    combine_hashes,
    compute_run_id,
    content_hash,
    file_sha256,
    short,
)
from frictiontune.core.retry import RetryPolicy, with_retry

__all__ = [
    "ArtifactError",
    "ArtifactStore",
    "BackendUnavailableError",
    "ConfigError",
    "ContentCache",
    "DatasetRef",
    "FrictionTuneError",
    "GateError",
    "IngestError",
    "LlmError",
    "LlmSchemaError",
    "RetryPolicy",
    "StageError",
    "UnsupportedOperationError",
    "ValidationError",
    "combine_hashes",
    "compute_run_id",
    "content_hash",
    "file_sha256",
    "short",
    "with_retry",
]
