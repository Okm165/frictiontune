"""frictiontune — friction-aware agent fine-tuning.

A data-centric pipeline that mines real agent chat logs, detects friction with
LLM/agent evaluators, compiles runtime rules, and builds curated SFT data.

Public entry point: the ``frictiontune`` CLI (see :mod:`frictiontune.cli`).
"""

from frictiontune.version import __version__

__all__ = ["__version__"]
