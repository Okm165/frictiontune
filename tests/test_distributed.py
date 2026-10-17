"""Tests for data-parallel training topology and the child launch command.

The world size comes from the hardware preset, and the launcher, the loader, and
the rank-0-only artifact writes all read it through one module. These lock the
mapping so a preset that says ``devices: 2`` cannot silently train on one card,
and so the RCCL knobs consumer cards need are always stamped.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from frictiontune.config.models import HardwareConfig
from frictiontune_train import distributed
from frictiontune_train.training import _build_command, _world_size


def _request(devices: Any) -> Any:
    return SimpleNamespace(hardware=HardwareConfig(devices=devices))


def test_topology_defaults_to_one_process(monkeypatch: Any) -> None:
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)

    assert distributed.world_size() == 1
    assert distributed.rank() == 0
    assert distributed.local_rank() == 0
    assert distributed.is_distributed() is False
    assert distributed.is_main_process() is True
    assert distributed.device_map() == "auto"
    assert distributed.config_kwargs() == {}


def test_topology_reads_the_torchrun_environment(monkeypatch: Any) -> None:
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")

    assert distributed.world_size() == 3
    assert distributed.is_distributed() is True
    assert distributed.is_main_process() is False
    # Each rank holds a whole replica on its own card, not a shard.
    assert distributed.device_map() == {"": 2}
    # The unused-parameter scan is off by default; the environment can bring it
    # back for architectures that do skip parameters.
    assert distributed.config_kwargs() == {"ddp_find_unused_parameters": False}
    monkeypatch.setenv("FRICTIONTUNE_DDP_FIND_UNUSED", "1")
    assert distributed.config_kwargs() == {"ddp_find_unused_parameters": True}


def test_world_size_comes_from_the_hardware_preset() -> None:
    assert _world_size(_request(1)) == 1
    assert _world_size(_request(2)) == 2
    assert _world_size(_request(3)) == 3


def test_single_card_launch_runs_the_runner_directly() -> None:
    environment: dict[str, str] = {}

    command = _build_command(
        _request(1), "sft", Path("/tmp/r.json"), Path("/tmp/out.json"), environment
    )

    assert command[0] == sys.executable
    assert "torch.distributed.run" not in command
    assert command[-5:] == [
        "-m",
        "frictiontune_train.runner",
        "sft",
        "/tmp/r.json",
        "/tmp/out.json",
    ]
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert "NCCL_P2P_DISABLE" not in environment


def test_multi_card_launch_runs_torchrun_with_rccl_fixes() -> None:
    environment: dict[str, str] = {}

    command = _build_command(
        _request(3), "sft", Path("/tmp/r.json"), Path("/tmp/out.json"), environment
    )

    assert "torch.distributed.run" in command
    assert "--standalone" in command
    assert "--nproc_per_node=3" in command
    # The declared topology and the visible devices can never disagree.
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2"
    # Consumer Radeons need the P2P path off or RCCL init fails.
    assert environment["NCCL_P2P_DISABLE"] == "1"
    assert environment["NCCL_IB_DISABLE"] == "1"
    # Unsloth is a single-card loader; DDP must use the transformers path.
    assert environment["FRICTIONTUNE_USE_UNSLOTH"] == "0"
    assert command[-5:] == [
        "-m",
        "frictiontune_train.runner",
        "sft",
        "/tmp/r.json",
        "/tmp/out.json",
    ]
