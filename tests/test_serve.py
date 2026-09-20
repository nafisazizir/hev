"""Offline tests for hev.serve: fakes stand in for the predictor, no hub or checkpoint loading."""
import json
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

import hev.serve as serve
from hev.evaluate import scaled_probabilities

BODY = {"state": {"ticket": "hi"}, "model": "hev-latest", "questions": {
    "n": {"type": "noul", "instructions": "yes?"},
    "c": {"type": "choice", "instructions": "which", "criteria": {"a": "A", "b": "B"}},
    "s": {"type": "score", "instructions": "how", "criteria": ["low", "mid", "high"]}}}


def fake_predictor(tok, probs, max_pos=4096):
    model = SimpleNamespace(lm=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=max_pos)),
                            level=SimpleNamespace(num_embeddings=64))
    model.probs = lambda enc: [torch.tensor(p) for p in probs(enc)]
    return SimpleNamespace(tokenizer=tok, device="cpu", model=model,
                           metadata={"base": "fake-base", "readout": {"head_kind": "pointer"}})


def even_probs(enc):
    return [[1.0 / len(o)] * len(o) for o in enc["opt_idx"]]


@pytest.fixture
def state():
    saved = dict(serve.STATE)
    yield serve.STATE
    serve.STATE.clear()
    serve.STATE.update(saved)


@pytest.fixture
def client(tok, state):
    return TestClient(serve.app)


def load(predictor, temperature=1.0, run="runs/fake"):
    serve.STATE.update({"run": run, "predictor": predictor, "temperature": temperature})


def test_systemone_answer_shapes(client, tok):
    load(fake_predictor(tok, even_probs))
    r = client.post("/v1/systemone", json=BODY)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "hev-latest"
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}
    n, c, s = body["answers"]["n"], body["answers"]["c"], body["answers"]["s"]
    assert n["type"] == "noul" and 0.0 <= n["noul"] <= 1.0
    assert c["type"] == "choice" and c["choice"] in {"a", "b"}
    assert set(c["probabilities"]) == {"a", "b"} and "confidence" in c
    assert s["type"] == "score" and s["legend"] == {"0": "low", "1": "mid", "2": "high"}
    assert set(s["probabilities"]) == {"0", "1", "2"} and "confidence" in s


def test_calibration_temperature_is_applied(client, tok):
    load(fake_predictor(tok, lambda enc: [[0.8, 0.2]]), temperature=2.0)
    body = {"state": "x", "questions": {"n": {"type": "noul", "instructions": "yes?"}}}
    r = client.post("/v1/systemone", json=body)
    expected = round(float(scaled_probabilities([0.8, 0.2], 2.0)[1]), 2)
    assert r.status_code == 200
    assert r.json()["answers"]["n"]["noul"] == expected != 0.2


def test_models_endpoint(client, tok):
    load(fake_predictor(tok, even_probs), temperature=1.5)
    m = client.get("/v1/models").json()["models"][0]
    assert m["id"] == "hev-latest" and "jev-latest" in m["aliases"]
    assert m["run"] == "runs/fake" and m["base"] == "fake-base"
    assert m["head"] == "pointer" and m["temperature"] == 1.5


def test_unloaded_returns_503(client):
    serve.STATE.update({"run": None, "predictor": None, "temperature": None})
    r = client.post("/v1/systemone", json={"state": "x", "questions": {"n": {"type": "noul", "instructions": "y?"}}})
    assert r.status_code == 503


def test_oversized_context_returns_422(client, tok):
    load(fake_predictor(tok, even_probs, max_pos=4))
    r = client.post("/v1/systemone", json=BODY)
    assert r.status_code == 422


def test_score_levels_beyond_embedding_returns_422(client, tok):
    load(fake_predictor(tok, even_probs))
    body = {"state": "x", "questions": {"s": {"type": "score", "instructions": "rate",
                                             "criteria": [str(i) for i in range(65)]}}}
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 422


def test_encode_value_error_returns_422(client, tok, monkeypatch):
    load(fake_predictor(tok, even_probs))
    monkeypatch.setattr(serve, "encode", lambda *a, **k: (_ for _ in ()).throw(ValueError("branch too long")))
    r = client.post("/v1/systemone", json=BODY)
    assert r.status_code == 422


def result_json(tmp_path, **overrides):
    result = {"status": "success", "test_evaluated": False, "head": "pointer",
              "base": "fake-base", "base_revision": "rev", "training_suite_sha256": "suite-sha",
              "training_config": {"config_sha256": "cfg-sha"},
              "temperature_fit": {"temperature": 1.5}}
    result.update(overrides)
    (tmp_path / "result.json").write_text(json.dumps(result))


def stub_local_predictor(monkeypatch):
    metadata = {"base": "fake-base", "base_revision": "rev", "suite_sha256": "suite-sha",
                "training_config_sha256": "cfg-sha", "readout": {"head_kind": "pointer"}}
    predictor = SimpleNamespace(metadata=metadata)
    monkeypatch.setattr(serve, "LocalPredictor", lambda run, device: predictor)
    return predictor


def test_load_run_accepts_matching_provenance(state, tmp_path, monkeypatch):
    predictor = stub_local_predictor(monkeypatch)
    result_json(tmp_path)
    serve.load_run(tmp_path, "cpu")
    assert serve.STATE["predictor"] is predictor
    assert serve.STATE["run"] == str(tmp_path) and serve.STATE["temperature"] == 1.5


def test_load_run_rejects_base_mismatch(state, tmp_path, monkeypatch):
    stub_local_predictor(monkeypatch)
    result_json(tmp_path, base="other-base")
    with pytest.raises(ValueError, match="provenance"):
        serve.load_run(tmp_path, "cpu")


def test_load_run_rejects_config_mismatch(state, tmp_path, monkeypatch):
    stub_local_predictor(monkeypatch)
    result_json(tmp_path, training_config={"config_sha256": "other"})
    with pytest.raises(ValueError, match="training config"):
        serve.load_run(tmp_path, "cpu")
