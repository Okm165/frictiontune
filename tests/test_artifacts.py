"""Tests for the artifact store: naming, kinds, round-tripping, streaming."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from frictiontune.core.artifacts import ArtifactStore
from frictiontune.core.errors import ArtifactError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path, pipeline="test_pipeline", run_id="abc123")


def test_logical_name_round_trips_without_suffix(store: ArtifactStore) -> None:
    jsonl = store.write_jsonl("messages", [{"a": 1}])
    assert jsonl.name == "messages"
    assert jsonl.path.endswith("messages.jsonl")
    assert jsonl.kind == "jsonl"

    rules = store.write_json("rules", {"items": []})
    assert rules.name == "rules"
    assert rules.path.endswith("rules.json")
    assert rules.kind == "json"

    report = store.write_text("report", "# hi")
    assert report.name == "report"
    assert report.path.endswith("report.md")
    assert report.kind == "text"


def test_write_does_not_double_suffix(store: ArtifactStore) -> None:
    ref = store.write_jsonl("messages.jsonl", [{"a": 1}])
    assert ref.path.endswith("messages.jsonl")
    assert not ref.path.endswith("messages.jsonl.jsonl")


def test_jsonl_round_trip(store: ArtifactStore) -> None:
    store.write_jsonl("rows", [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    rows = list(store.iter_jsonl("rows"))
    assert rows == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


def test_missing_artifact_raises(store: ArtifactStore) -> None:
    with pytest.raises(ArtifactError):
        store.read_json("absent")
    with pytest.raises(ArtifactError):
        list(store.iter_jsonl("absent"))


def test_digest_links_manifest(store: ArtifactStore) -> None:
    ref = store.write_jsonl("rows", [{"a": 1}])
    assert store.digest_of("rows") == ref.sha256
    assert store.digest_of("rows.jsonl") == ref.sha256

    json_ref = store.write_json("rules", {"items": []})
    assert store.digest_of("rules") == json_ref.sha256

    text_ref = store.write_text("report", "# hi")
    assert store.digest_of("report") == text_ref.sha256
