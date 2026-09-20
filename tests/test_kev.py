"""Offline tests for hev.kev (KevPredictor) and the vendored hev.kev_model.

The fake run mirrors a released kev checkpoint directory; the backbone, tokenizer and PEFT adapter are injected
by monkeypatching module-level names (no seam was added to the vendored code). The `hub` test loads the real
jaredpalmer/kev-0.6b snapshot from the local HF cache and is skipped unless HEV_HUB_TESTS=1.
"""
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import hev.kev as kev
import hev.kev_model as kev_model
from hev.data import materialize
from hev.kev import KevPredictor
from hev.kev_model import KEV_COMMIT, KEV_MODEL_SHA256, VENDOR_MARKER, PointerHead
from hev.model import encode as hev_encode
from hev.suite import load_split

ROOT = Path(__file__).resolve().parents[1]
KEV_CLONE = Path("/Users/nafis/Documents/personal/kev")
KEV_0_6B = Path("~/.cache/huggingface/hub/models--jaredpalmer--kev-0.6b/snapshots/83e05fabf7ef08e343bb4daf144e08777f99713d").expanduser()
KEV_0_6B_SUITE = "1b33e566d114f9eafeff55b36c221fadb2a4ae358a1b9cc68006e82c7cfad8f1"


def request():
    return {
        "state": "the shoes arrived late and in the wrong size",
        "questions": {
            "team": {"type": "choice", "instructions": "which team?",
                     "criteria": {"returns": None, "shipping": None, "billing": None}, "label": "returns", "src": "fixture"},
            "urgent": {"type": "noul", "instructions": "urgent?", "label": True, "src": "fixture"},
        },
    }


def fake_kev_run(tmp_path, **head_extra):
    run = tmp_path / "kev-run"
    run.mkdir()
    torch.save({"head": PointerHead(64, 256).state_dict(), "base": "fake-base", "base_revision": "rev", "lora": 16,
                "holdout": [], "args": {"seed": 0, "lr": 2e-4}, "suite_sha256": "suite-sha", **head_extra}, run / "head.pt")
    (run / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "fake-base", "r": 16,
                                                         "trainable_token_indices": None}))
    (run / "adapter_model.safetensors").write_bytes(b"weights")
    (run / "provenance.json").write_text("{}")
    (run / "result.json").write_text("{}")
    (run / "training_config.json").write_text("{}")
    return run


@pytest.fixture
def patched(monkeypatch, tiny_backbone, tok):
    """Route kev's loading path onto the offline fixtures without touching the vendored class."""
    monkeypatch.setattr(kev, "load_tokenizer", lambda name, revision=None: tok)
    monkeypatch.setattr(kev_model, "AutoModelForCausalLM",
                        SimpleNamespace(from_pretrained=lambda name, **kw: SimpleNamespace(model=copy.deepcopy(tiny_backbone))))
    monkeypatch.setattr(kev, "load_adapter", lambda lm, run, device: lm)
    return tok


# ---------------------------------------------------------------- vendoring


def test_vendored_text_matches_upstream_hash():
    text = (ROOT / "hev" / "kev_model.py").read_text(encoding="utf-8")
    tail = text[text.rindex(VENDOR_MARKER) + len(VENDOR_MARKER):]
    assert hashlib.sha256(tail.encode("utf-8")).hexdigest() == KEV_MODEL_SHA256
    assert len(KEV_COMMIT) == 40


def test_upstream_clone_agrees_with_recorded_hash():
    if not (KEV_CLONE / ".git").exists():
        pytest.skip("sibling kev clone not present")
    upstream = subprocess.run(["git", "-C", str(KEV_CLONE), "show", f"{KEV_COMMIT}:kev/model.py"],
                              capture_output=True, check=True).stdout
    assert hashlib.sha256(upstream).hexdigest() == KEV_MODEL_SHA256


# ---------------------------------------------------------------- packing: why the vendored path exists


def test_kev_packing_puts_decide_after_options(tok):
    rec = materialize(request())
    k, h = kev_model.encode(tok, rec), hev_encode(tok, rec)
    assert k["ids"] != h["ids"]
    assert sorted(k["ids"]) == sorted(h["ids"])  # same multiset of tokens, different order
    for d, oi in zip(k["decide_idx"], k["opt_idx"]):
        assert d > max(oi)
    for d, oi in zip(h["decide_idx"], h["opt_idx"]):
        assert d < min(oi)


def test_plain_kev_positions_depend_on_option_order(tok):
    a = {"state": "s", "questions": [{"instr": "which?", "options": ["a", "bbbb", "cc"], "label": 0}]}
    b = {"state": "s", "questions": [{"instr": "which?", "options": ["bbbb", "a", "cc"], "label": 1}]}
    ea, eb = kev_model.encode(tok, a), kev_model.encode(tok, b)
    # sequential positions: option "a" ends at a different position once "bbbb" precedes it
    assert ea["pos"][ea["opt_idx"][0][0]] != eb["pos"][eb["opt_idx"][0][1]]
    assert ea["pos"] == list(range(len(ea["ids"])))


def test_isolated_kev_positions_share_an_option_start(tok):
    rec = {"state": "s", "questions": [{"instr": "which?", "options": ["a", "bbbb", "cc"], "label": 0}]}
    enc = kev_model.encode(tok, rec, option_isolation=True)
    o_id = tok.convert_tokens_to_ids(kev_model.SPECIAL[2])
    starts = [enc["pos"][i] for i, t in enumerate(enc["ids"]) if t == o_id]
    assert len(starts) == 3 and len(set(starts)) == 1
    assert enc["option_isolation"] is True and enc["opt"][enc["decide_idx"][0]] == kev_model.OPT_DECIDE


# ---------------------------------------------------------------- predictor


def test_predictor_probabilities_and_latency(tmp_path, patched):
    predictor = KevPredictor(fake_kev_run(tmp_path), "cpu")
    probabilities, latency = predictor(request())
    assert [len(p) for p in probabilities] == [3, 2]
    for p in probabilities:
        assert isinstance(p, np.ndarray) and p.dtype == np.float32
        assert abs(float(p.sum()) - 1) < 1e-5 and (p >= 0).all()
    assert isinstance(latency, float) and 0 < latency < 60
    assert not hasattr(predictor.model, "level")
    assert predictor.run == tmp_path / "kev-run" and predictor.device == "cpu"


def test_predictor_metadata(tmp_path, patched):
    predictor = KevPredictor(fake_kev_run(tmp_path), "cpu")
    meta = predictor.metadata
    assert set(meta) == {"base", "base_revision", "suite_sha256", "readout", "head_dim", "option_isolation",
                         "special_embeddings", "lora", "training_args", "kev_source", "checkpoint"}
    assert meta["readout"] == {"head_kind": "kev-pointer"}
    assert meta["base"] == "fake-base" and meta["base_revision"] == "rev" and meta["suite_sha256"] == "suite-sha"
    assert meta["head_dim"] == 256 and meta["option_isolation"] is False and meta["special_embeddings"] is False
    assert meta["lora"] == 16 and meta["training_args"] == {"seed": 0, "lr": 2e-4}
    assert meta["kev_source"] == {"commit": KEV_COMMIT, "model_py_sha256": KEV_MODEL_SHA256}
    checkpoint = meta["checkpoint"]
    assert checkpoint["reference"] == str(tmp_path / "kev-run") and checkpoint["hub_commit"] is None
    assert set(checkpoint["file_hashes"]) == {"adapter_config.json", "adapter_model.safetensors", "head.pt",
                                              "provenance.json", "result.json", "training_config.json"}
    assert checkpoint["file_hashes"]["adapter_model.safetensors"] == hashlib.sha256(b"weights").hexdigest()


def test_predictor_rejects_special_embeddings(tmp_path, patched):
    with pytest.raises(ValueError, match="special_embeddings"):
        KevPredictor(fake_kev_run(tmp_path, special_embeddings=True), "cpu")


def test_predictor_rejects_inconsistent_head_dim(tmp_path, patched):
    with pytest.raises(ValueError, match="head_dim"):
        KevPredictor(fake_kev_run(tmp_path, head_dim=128), "cpu")


def test_predictor_rejects_bundled_tokenizer_mismatch(tmp_path, patched, monkeypatch):
    run = fake_kev_run(tmp_path)
    (run / "tokenizer.json").write_text("{}")
    shifted = copy.copy(patched)
    shifted.special = {t: i + 2 for i, t in enumerate(kev_model.SPECIAL)}
    monkeypatch.setattr(kev, "load_tokenizer", lambda name, revision=None: shifted if name == str(run) else patched)
    with pytest.raises(ValueError, match="delimiter ids"):
        KevPredictor(run, "cpu")


def test_predictor_refuses_overlong_requests(tmp_path, patched):
    predictor = KevPredictor(fake_kev_run(tmp_path), "cpu")
    long = request()
    long["state"] = "x" * 300
    long["questions"]["team"]["criteria"] = {f"option {i}": "y" * 200 for i in range(12)}
    long["questions"]["team"]["label"] = "option 0"
    with pytest.raises(ValueError):
        predictor(long)


# ---------------------------------------------------------------- real checkpoint


@pytest.mark.hub
def test_released_kev_0_6b_scores_a_development_record():
    if not KEV_0_6B.exists():
        pytest.skip("jaredpalmer/kev-0.6b snapshot not in the local HF cache")
    predictor = KevPredictor(KEV_0_6B, "cpu")
    assert predictor.metadata["suite_sha256"] == KEV_0_6B_SUITE
    assert predictor.metadata["option_isolation"] is False
    assert predictor.metadata["readout"]["head_kind"] == "kev-pointer"
    assert predictor.metadata["checkpoint"]["hub_commit"] == KEV_0_6B.name
    assert not hasattr(predictor.model, "level")
    record = min(load_split(ROOT / "evals" / "decision-v2", "development"), key=lambda r: len(json.dumps(r)))
    probabilities, latency = predictor(record)
    assert len(probabilities) == len(record["questions"])
    for p in probabilities:
        assert np.isfinite(p).all() and abs(float(p.sum()) - 1) < 1e-5 and (p >= 0).all()
    assert isinstance(latency, float) and latency > 0
