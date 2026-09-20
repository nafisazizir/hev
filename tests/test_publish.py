"""Offline tests for hev.publish: bundle validation and dry-run never touch the Hub."""
import json
from pathlib import Path

import pytest
import torch

import hev.publish as publish


def fake_run(tmp_path, head="pointer", seed=0, test_evaluated=False):
    run = tmp_path / "m9-pointer-s0"
    run.mkdir()
    (run / "adapter_config.json").write_text("{}")
    (run / "adapter_model.safetensors").write_bytes(b"weights")
    (run / "tokenizer.json").write_text("{}")
    torch.save({"base": "fake-base", "base_revision": "rev", "suite_sha256": "suite-sha",
                "training_config_sha256": "cfg-sha", "args": {"seed": seed},
                "readout": {"head_kind": head}}, run / "readout.pt")
    config = {"status": "configured", "args": {"seed": seed}, "config_sha256": "cfg-sha"}
    (run / "training_config.json").write_text(json.dumps(config))
    (run / "training_metrics.json").write_text(json.dumps({"status": "success"}))
    (run / "result.json").write_text(json.dumps(
        {"status": "success", "test_evaluated": test_evaluated, "head": head,
         "base": "fake-base", "base_revision": "rev", "training_suite_sha256": "suite-sha",
         "training_config": config, "temperature_fit": {"temperature": 1.0}}))
    return run


@pytest.fixture
def card(tmp_path, monkeypatch):
    card = tmp_path / "MODEL_CARD.md"
    card.write_text("see https://huggingface.co/OWNER/hev-0.6b")
    monkeypatch.setattr(publish, "DEFAULT_CARD", card)
    return card


def bundle_files(tmp):
    return sorted(p.name for p in Path(tmp.name).iterdir())


def test_build_bundle_manifest(tmp_path, card):
    run = fake_run(tmp_path)
    tmp, manifest = publish.build_bundle(run, "me/hev-0.6b", "seed-0")
    with tmp:
        assert manifest["schema_version"] == 1
        assert manifest["repo"] == "me/hev-0.6b" and manifest["revision"] == "seed-0"
        assert manifest["run"] == run.name and manifest["seed"] == 0
        assert manifest["base"] == "fake-base" and manifest["head"] == "pointer"
        assert manifest["test_evaluated"] is False
        assert set(manifest["files"]) == {"adapter_config.json", "adapter_model.safetensors",
                                          "readout.pt", "result.json", "training_config.json",
                                          "training_metrics.json", "tokenizer.json"}
        staged = bundle_files(tmp)
        assert "README.md" in staged and "MANIFEST.json" in staged


def test_default_card_staged_and_substituted(tmp_path, card):
    run = fake_run(tmp_path)
    tmp, _ = publish.build_bundle(run, "me/hev-0.6b", "seed-0")
    with tmp:
        text = (Path(tmp.name) / "README.md").read_text()
        assert "me/hev-0.6b" in text and "OWNER/hev-0.6b" not in text
        assert "OWNER/hev-0.6b" in card.read_text()


def test_explicit_card_substituted(tmp_path, card):
    run = fake_run(tmp_path)
    custom = tmp_path / "card.md"
    custom.write_text("model: OWNER/hev-0.6b")
    tmp, _ = publish.build_bundle(run, "me/hev-0.6b", "seed-0", card=custom)
    with tmp:
        assert (Path(tmp.name) / "README.md").read_text() == "model: me/hev-0.6b"


def test_missing_card_rejects(tmp_path):
    run = fake_run(tmp_path)
    with pytest.raises(FileNotFoundError, match="model card"):
        publish.build_bundle(run, "me/hev-0.6b", "seed-0", card=tmp_path / "nope.md")


def test_dry_run_no_network(tmp_path, monkeypatch, capsys, card):
    run = fake_run(tmp_path)
    monkeypatch.setattr("sys.argv", ["hev.publish", "--run", str(run),
                                   "--repo", "me/hev-0.6b", "--revision", "seed-0", "--dry-run"])
    publish.main()
    out = json.loads(capsys.readouterr().out)
    assert out["repo"] == "me/hev-0.6b" and out["revision"] == "seed-0"
    assert out["run"] == run.name
    assert "MANIFEST.json" in out["files"] and "result.json" in out["files"]


def test_rejects_test_evaluated(tmp_path, card):
    run = fake_run(tmp_path, test_evaluated=True)
    with pytest.raises(ValueError, match="test"):
        publish.build_bundle(run, "me/hev-0.6b", "seed-0")


def test_rejects_seed_mismatch(tmp_path, card):
    run = fake_run(tmp_path, seed=0)
    with pytest.raises(ValueError, match="seed"):
        publish.build_bundle(run, "me/hev-0.6b", "seed-1")


def test_rejects_non_seed_revision(tmp_path, card):
    run = fake_run(tmp_path)
    with pytest.raises(ValueError, match="seed-N"):
        publish.build_bundle(run, "me/hev-0.6b", "v1.0")


def test_rejects_altered_training_config(tmp_path, card):
    run = fake_run(tmp_path)
    config = json.loads((run / "training_config.json").read_text())
    config["args"]["lr"] = 9
    (run / "training_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="training config"):
        publish.build_bundle(run, "me/hev-0.6b", "seed-0")


def test_rejects_metadata_seed_mismatch(tmp_path, card):
    run = fake_run(tmp_path, seed=0)
    metadata = torch.load(run / "readout.pt", map_location="cpu", weights_only=True)
    metadata["args"]["seed"] = 1
    torch.save(metadata, run / "readout.pt")
    with pytest.raises(ValueError, match="seed"):
        publish.build_bundle(run, "me/hev-0.6b", "seed-0")


def test_rejects_set_head(tmp_path, card):
    run = fake_run(tmp_path, head="set")
    with pytest.raises(ValueError, match="pointer"):
        publish.build_bundle(run, "me/hev-0.6b", "seed-0")
