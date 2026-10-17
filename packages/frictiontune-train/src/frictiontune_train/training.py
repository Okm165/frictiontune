"""The :class:`TrainingBackend` implementation.

The trainer runs in a child process, never in the pipeline process, so the
trainer's CUDA context and memory stay out of the pipeline. The *topology* of
that child is not fixed: it is whatever the hardware preset names.

* ``devices: 1`` — one process on one card.
* ``devices: N`` — a torchrun data-parallel group across ``N`` cards, each rank
  holding a full model replica and syncing LoRA gradients.

The world size is read from the hardware preset, never hard-coded, so a run
declares its parallelism the same way it declares its quantisation.

The child is also where the native build environment is fixed up: TRL's fused
LM head JIT-compiles a Triton kernel, and Triton's AMD backend builds a small C
extension against the Python C API. That build needs ``Python.h``, which this
module points the compiler at.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from frictiontune.backends.models import TrainingResult
from frictiontune.core.errors import StageError
from frictiontune.core.logging import get_logger

if TYPE_CHECKING:
    from frictiontune.backends.models import DpoRequest, GrpoRequest, TrainingRequest

_log = get_logger("frictiontune_train.training")

_STAGE = {"sft": "train_sft", "dpo": "train_dpo", "grpo": "train_grpo"}
_LOCAL_INCLUDE = Path(".cache/frictiontune/python-include/usr/include")


class LocalTrainingBackend:
    """Runs SFT and GRPO jobs on the local machine's accelerators."""

    async def train(self, request: TrainingRequest) -> TrainingResult:
        """Run a supervised QLoRA job in a child process."""
        return await asyncio.to_thread(_run_in_child, request, "sft")

    async def train_grpo(self, request: GrpoRequest) -> TrainingResult:
        """Run a GRPO job in a child process."""
        return await asyncio.to_thread(_run_in_child, request, "grpo")

    async def train_dpo(self, request: DpoRequest) -> TrainingResult:
        """Run a DPO preference job in a child process (GAP-104)."""
        return await asyncio.to_thread(_run_in_child, request, "dpo")


def _run_in_child(request: Any, kind: str) -> TrainingResult:
    """Serialize ``request``, train in a child process, return its result."""
    with tempfile.TemporaryDirectory(prefix="frictiontune-train-") as tmp:
        request_path = Path(tmp) / "request.json"
        result_path = Path(tmp) / "result.json"
        request_path.write_text(request.model_dump_json())
        environment = {**os.environ}
        _expose_python_headers(environment, _STAGE[kind])
        _stabilise_allocator(environment)
        command = _build_command(request, kind, request_path, result_path, environment)
        world = _world_size(request)
        _log.info("launching trainer", kind=kind, world_size=world)
        try:
            subprocess.run(command, check=True, env=environment, cwd=Path.cwd())
        except subprocess.CalledProcessError as exc:
            raise StageError(_STAGE[kind], f"trainer exited with code {exc.returncode}") from exc
        if not result_path.is_file():
            raise StageError(_STAGE[kind], "trainer produced no result")
        return TrainingResult.model_validate_json(result_path.read_text())


def _world_size(request: Any) -> int:
    """Resolve the training world size from the hardware preset.

    The preset is the single source of truth: ``devices: 3`` means three
    processes across three cards. ``"auto"`` defers to the number of visible
    accelerators. A non-positive or unreadable value degrades to one process.
    """
    devices = getattr(request.hardware, "devices", 1)
    if isinstance(devices, int):
        return max(1, devices)
    return max(1, _visible_devices())


def _visible_devices() -> int:
    """Count accelerators, or ``0`` when torch is unavailable."""
    try:
        torch = __import__("torch")
        return int(torch.cuda.device_count())
    except Exception:  # pragma: no cover - torch missing is a packaging error
        return 0


def _build_command(
    request: Any, kind: str, request_path: Path, result_path: Path, environment: dict[str, str]
) -> list[str]:
    """Return the child argv and stamp the environment for the chosen topology.

    ``devices: 1`` pins one card and runs the runner directly. Several devices
    run the runner under ``torchrun`` so every rank gets its ``LOCAL_RANK``, with
    ``CUDA_VISIBLE_DEVICES`` narrowed to exactly the declared cards so the group
    size and the preset can never disagree.
    """
    world = _world_size(request)
    module = ["-m", "frictiontune_train.runner", kind, str(request_path), str(result_path)]
    if world == 1:
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        return [sys.executable, *module]

    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(world))
    # Consumer Radeons have no working PCIe peer access, so RCCL's default
    # P2P path fails collective init with ``ncclUnhandledCudaError``. Disabling
    # P2P (and IB, absent on these boxes) is the documented topology for them and
    # costs nothing where P2P is unavailable anyway. Data-center fabrics can
    # re-enable it through the environment.
    environment.setdefault("NCCL_P2P_DISABLE", os.environ.get("FRICTIONTUNE_NCCL_P2P_DISABLE", "1"))
    environment.setdefault("NCCL_IB_DISABLE", os.environ.get("FRICTIONTUNE_NCCL_IB_DISABLE", "1"))
    # Unsloth is a single-card loader; force the plain transformers path under DDP.
    environment["FRICTIONTUNE_USE_UNSLOTH"] = "0"
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world}",
        "--master_port",
        str(_free_port()),
        *module,
    ]


def _free_port() -> int:
    """Return a currently free TCP port for torchrun's rendezvous."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stabilise_allocator(environment: dict[str, str]) -> None:
    """Ask the allocator to use expandable segments where the build supports it.

    Long-context QLoRA holds a large, churning working set (4-bit weights, the
    vocabulary-wide logits, and the backward buffers). The default caching
    allocator can fragment that set and fail to satisfy a large contiguous
    request even though the free blocks would sum to enough. Expandable segments
    let the allocator grow a block in place instead. The HIP build does not
    support the option (it warns and ignores it), so it is offered only to
    CUDA; ``FRICTIONTUNE_ALLOC_CONF`` sets it explicitly on either vendor. Both
    the current and the historical variable name are written, because the
    spelling moved between torch releases.
    """
    value = os.environ.get("FRICTIONTUNE_ALLOC_CONF")
    if value is None and not _on_hip():
        value = "expandable_segments:True"
    if value:
        environment.setdefault("PYTORCH_ALLOC_CONF", value)
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", value)


def _on_hip() -> bool:
    """Whether the torch in this environment is the ROCm build (cheap, no init)."""
    try:
        from torch.version import hip
    except Exception:  # torch may be absent from the launcher environment
        return False
    return bool(hip)


def _expose_python_headers(environment: dict[str, str], stage: str) -> None:
    """Add a ``Python.h`` directory to ``C_INCLUDE_PATH`` for the child.

    Triton's ROCm backend JIT-compiles a C stub against the Python C API the
    first time TRL's fused LM head runs. On a host without the Python
    development package that compile fails deep inside Triton with a bare
    ``Python.h: No such file or directory``. The interpreter's own include dir
    is used when present; otherwise a repo-local copy can be supplied through
    ``FRICTIONTUNE_PYTHON_INCLUDE``. A missing header is raised as a named
    stage error rather than mis-attributed to the model or the data.
    """
    include = _python_include()
    if include is None:
        raise StageError(
            stage,
            "Python development headers (Python.h) are required to build Triton's "
            "runtime support; install the Python development package or set "
            "FRICTIONTUNE_PYTHON_INCLUDE to a directory containing Python.h",
        )
    environment["C_INCLUDE_PATH"] = os.pathsep.join(
        [str(include), *filter(None, [environment.get("C_INCLUDE_PATH")])]
    )


def _python_include() -> Path | None:
    """Return a directory that contains ``Python.h``, or ``None``."""
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        os.environ.get("FRICTIONTUNE_PYTHON_INCLUDE"),
        sysconfig.get_paths()["include"],
        str(_LOCAL_INCLUDE / version),
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "Python.h").is_file():
            return Path(candidate).resolve()
    return None
