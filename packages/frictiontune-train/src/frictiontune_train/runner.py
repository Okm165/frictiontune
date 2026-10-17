"""Child-process entry point for one training job.

The pipeline never trains in-process: it serialises a request onto disk and runs
this module in a child (pinned to one card, or one *rank* of a torchrun group;
see :mod:`frictiontune_train.training`), then reads the result JSON back. Under
data parallelism every rank runs this module, but only rank 0 writes the result
so the ranks cannot race on one file.

Usage: ``python -m frictiontune_train.runner <sft|grpo> <request.json> <result.json>``
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from frictiontune_train.distributed import is_main_process

if TYPE_CHECKING:
    from frictiontune.backends.models import TrainingResult


def main(argv: list[str]) -> int:
    """Run one training job and write its result JSON on the main process."""
    if len(argv) != 3:
        raise SystemExit("usage: runner <sft|grpo> <request.json> <result.json>")
    kind, request_path, result_path = argv
    result = _run(kind, Path(request_path))
    if is_main_process():
        Path(result_path).write_text(result.model_dump_json())
    return 0


def _run(kind: str, path: Path) -> TrainingResult:
    """Load the request for ``kind`` and dispatch to its trainer."""
    from frictiontune.backends.models import GrpoRequest, TrainingRequest

    from frictiontune_train.grpo import run_grpo
    from frictiontune_train.sft import run_sft

    if kind == "sft":
        return run_sft(TrainingRequest.model_validate_json(path.read_text()))
    return run_grpo(GrpoRequest.model_validate_json(path.read_text()))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
