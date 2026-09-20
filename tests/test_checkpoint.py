"""Offline tests for hev.checkpoint: Hub reference parsing and run resolution."""
from pathlib import Path

import pytest

import hev.checkpoint as checkpoint
from hev.checkpoint import parse_hub_reference, resolve_run


def test_parse_with_revision():
    assert parse_hub_reference("hf://OWNER/hev-0.6b@seed-0") == ("OWNER/hev-0.6b", "seed-0")


def test_parse_without_revision():
    assert parse_hub_reference("hf://OWNER/hev-0.6b") == ("OWNER/hev-0.6b", None)


@pytest.mark.parametrize("ref", [
    "OWNER/hev-0.6b",
    "hf://hev-0.6b",
    "hf://a/b/c",
    "hf:///name",
    "hf://owner/",
    "hf://owner/name@",
])
def test_parse_rejects_malformed(ref):
    with pytest.raises(ValueError):
        parse_hub_reference(ref)


def test_existing_local_path_wins(tmp_path):
    assert resolve_run(tmp_path) == tmp_path


def test_missing_local_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_run(tmp_path / "nope")


def test_hub_reference_downloads(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot_download(repo_id, revision=None):
        calls.append((repo_id, revision))
        return tmp_path / "snapshot"

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    resolved = resolve_run("hf://OWNER/hev-0.6b@seed-0")
    assert resolved == tmp_path / "snapshot"
    assert calls == [("OWNER/hev-0.6b", "seed-0")]
