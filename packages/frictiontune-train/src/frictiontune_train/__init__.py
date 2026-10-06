"""Heavy, GPU-side implementations for frictiontune.

This package is the *only* place that imports the ML stack (torch, transformers,
peft, trl, bitsandbytes, unsloth). The data pipeline depends on it solely through
the backend protocols, so a CPU log box never needs any of it.

The entry point ``frictiontune_train.plugin:register`` wires the concrete
backends into a :class:`~frictiontune.backends.registry.BackendRegistry`.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
