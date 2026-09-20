"""Offline tests for evaluating outside the run directory: --out, --allow-cross-suite and --checkpoint-kind kev.

Everything runs against a tiny fake suite built in tmp_path and stub predictors; no backbone is loaded.
"""
import json
import sys
import types
from pathlib import Path

import pytest
import torch

import hev.evaluate as evaluate
from hev.data import EVAL_ONLY, TRAINABLE
from hev.suite import digest, object_digest, source_hashes, write_json
from hev.train import validate_training_records

BASE, REVISION = "fake-base", "rev"


def choice_record(identifier, source, label="a", task=None, **meta):
    return {
        "state": f"state {identifier}",
        "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {"a": None, "b": None, "c": None},
                            "label": label, "src": task or source}},
        "_meta": {"id": identifier, "group_id": identifier, "source": source, "variant": "clean", **meta},
    }


def write_suite(directory, splits, eval_only=False, eval_only_sources=(), trainable_sources=(), **extra):
    directory.mkdir(parents=True)
    files = {}
    for split in ("train", "calibration", "development"):
        path = directory / f"{split}.jsonl"
        records = splits.get(split, [])
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        files[path.name] = {"sha256": digest(path), "records": len(records)}
    write_json(directory / "manifest.json", {
        "version": 99, "files": files, "base_revisions": {BASE: REVISION}, "eval_only": eval_only,
        "eval_only_sources": list(eval_only_sources), "trainable_sources": list(trainable_sources), **extra,
    })
    return directory


@pytest.fixture
def suites(tmp_path):
    decision = write_suite(tmp_path / "decision", {
        "train": [choice_record("t1", "banking77")],
        "calibration": [choice_record("c1", "banking77"), choice_record("c2", "boolq", label="b")],
        "development": [choice_record("d1", "banking77"), choice_record("d2", "boolq", label="b")],
    }, trainable_sources=["banking77", "boolq"])
    transfer = write_suite(tmp_path / "transfer", {
        "development": [choice_record("h1", "legacy_holdout"),
                        choice_record("h2", "composition_holdout", task="composition_held_and_or")],
    }, eval_only=True, eval_only_sources=["legacy_holdout", "composition_holdout"],
       excluded_training_states_from="evals/decision")
    return decision, transfer


def fake_run(tmp_path, suite_sha256, recorded_sources):
    run = tmp_path / "run"
    run.mkdir()
    config = {"status": "configured", "args": {"seed": 0}, "provenance": {"source_hashes": recorded_sources}}
    config["config_sha256"] = object_digest(config)
    write_json(run / "training_config.json", config)
    write_json(run / "training_metrics.json", {"status": "success"})
    torch.save({"base": BASE, "base_revision": REVISION, "suite_sha256": suite_sha256,
                "training_config_sha256": config["config_sha256"], "provenance": {"source_hashes": recorded_sources},
                "readout": {"head_kind": "pointer"}}, run / "readout.pt")
    return run


class StubPredictor:
    """Duck-types LocalPredictor: metadata from readout.pt, semantic (order-invariant) answers."""
    def __init__(self, run, device):
        self.run, self.device = Path(run), device
        self.metadata = torch.load(self.run / "readout.pt", map_location="cpu", weights_only=True)
        self.model = torch.nn.Module()
        self.model.level = torch.nn.Embedding(2, 2)

    def __call__(self, request):
        weights = {"a": 0.7, "b": 0.2, "c": 0.1}
        return [[weights[key] for key in question["criteria"]] for question in request["questions"].values()], 0.0


def args(**overrides):
    return evaluate.parser().parse_args([
        "--run", str(overrides.pop("run")), "--suite", str(overrides.pop("suite")),
        "--transfer", str(overrides.pop("transfer")), "--device", "cpu", "--permutations", "2",
        "--bootstrap-samples", "5", *overrides.pop("extra", []),
    ])


def snapshot(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def test_out_refuses_existing_directory_and_leaves_run_untouched(tmp_path, suites, monkeypatch):
    decision, transfer = suites
    run = fake_run(tmp_path, digest(decision / "manifest.json"), source_hashes())
    monkeypatch.setattr(evaluate, "LocalPredictor", StubPredictor)
    before = snapshot(run)
    existing = tmp_path / "taken"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="existing output directory"):
        evaluate.evaluate(args(run=run, suite=decision, transfer=transfer, extra=["--out", str(existing)]))
    assert snapshot(run) == before and list(existing.iterdir()) == []


def test_out_directory_receives_every_artifact(tmp_path, suites, monkeypatch):
    decision, transfer = suites
    run = fake_run(tmp_path, digest(decision / "manifest.json"), source_hashes())
    monkeypatch.setattr(evaluate, "LocalPredictor", StubPredictor)
    before = snapshot(run)
    out = tmp_path / "fresh" / "nested"
    result = evaluate.evaluate(args(run=run, suite=decision, transfer=transfer, extra=["--out", str(out)]))
    assert snapshot(run) == before
    assert {p.name for p in out.iterdir()} == {"result.json", "calibration_rows.json", "decision_rows.json", "transfer_rows.json"}
    assert result["schema_version"] == 2 and result["checkpoint_kind"] == "hev" and result["cross_suite"] is None
    assert result["output_dir"] == str(out) and result["evaluation_flags"]["out"] == str(out)
    assert result["evaluation_suite"] == {"path": str(decision), "manifest_sha256": digest(decision / "manifest.json")}
    assert result["provenance"]["gates"] == {"training_config": "enforced", "source_hashes": "enforced"}
    assert result["provenance"]["source_hashes_match"] and result["provenance"]["changed_files"] == []
    assert result["artifact_hashes"]["decision"]["path"] == "decision_rows.json"
    assert result["evaluation_config"] == {"seed": 1, "permutations": 2, "bootstrap_samples": 5, "level_zero_ablation": False}


def test_cross_suite_without_flag_raises_existing_error(tmp_path, suites, monkeypatch):
    decision, transfer = suites
    run = fake_run(tmp_path, "some-other-suite", source_hashes())
    monkeypatch.setattr(evaluate, "LocalPredictor", StubPredictor)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="checkpoint and training suite hashes differ"):
        evaluate.evaluate(args(run=run, suite=decision, transfer=transfer, extra=["--out", str(out)]))
    assert json.loads((out / "evaluation_failure.json").read_text())["error_type"] == "ValueError"
    assert not (run / "evaluation_failure.json").exists() and not (run / "result.json").exists()


def test_changed_source_without_flag_raises_existing_error(tmp_path, suites, monkeypatch):
    decision, transfer = suites
    stale = {**source_hashes(), "hev/evaluate.py": "0" * 64}
    run = fake_run(tmp_path, digest(decision / "manifest.json"), stale)
    monkeypatch.setattr(evaluate, "LocalPredictor", StubPredictor)
    with pytest.raises(ValueError, match="source changed after training"):
        evaluate.evaluate(args(run=run, suite=decision, transfer=transfer))
    assert (run / "evaluation_failure.json").exists() and not (run / "result.json").exists()


def test_allow_cross_suite_requires_out(tmp_path, suites, monkeypatch, capsys):
    decision, transfer = suites
    with pytest.raises(ValueError, match="--allow-cross-suite requires --out"):
        evaluate.evaluate(args(run=tmp_path / "run", suite=decision, transfer=transfer, extra=["--allow-cross-suite"]))
    assert not (tmp_path / "run").exists()
    monkeypatch.setattr(sys, "argv", ["evaluate", "--run", "r", "--suite", "s", "--allow-cross-suite"])
    with pytest.raises(SystemExit) as exit_info:
        evaluate.main()  # argparse-level rejection, before any directory or model is touched
    assert exit_info.value.code == 2 and "--allow-cross-suite requires --out" in capsys.readouterr().err


def test_allow_cross_suite_records_suite_and_source_mismatch(tmp_path, suites, monkeypatch):
    decision, transfer = suites
    current = source_hashes()
    stale = {**current, "hev/evaluate.py": "0" * 64, "hev/removed.py": "1" * 64}
    run = fake_run(tmp_path, "training-suite-sha", stale)
    monkeypatch.setattr(evaluate, "LocalPredictor", StubPredictor)
    out = tmp_path / "cross"
    result = evaluate.evaluate(args(run=run, suite=decision, transfer=transfer, extra=["--out", str(out), "--allow-cross-suite"]))
    assert result["cross_suite"] == {
        "training_suite_sha256": "training-suite-sha",
        "evaluation_suite_sha256": digest(decision / "manifest.json"),
        "transfer_suite_sha256": digest(transfer / "manifest.json"),
        "suite_differs": True,
        "allowed_by_flag": True,
    }
    assert result["training_suite_sha256"] == "training-suite-sha"
    assert result["transfer_suite"] == {"path": str(transfer), "manifest_sha256": digest(transfer / "manifest.json")}
    provenance = result["provenance"]
    assert provenance["kind"] == "hev" and provenance["source_hashes_match"] is False
    assert provenance["changed_files"] == ["hev/evaluate.py", "hev/removed.py"]
    assert provenance["training_source_hashes"] == stale and provenance["evaluation_source_hashes"] == current
    assert provenance["gates"]["source_hashes"] == "relaxed by --allow-cross-suite"
    assert result["evaluation_flags"]["allow_cross_suite"] is True
    assert result["transfer"]["coverage"]["evaluated_questions"] == 2
    assert not (run / "result.json").exists()
    assert json.loads((out / "result.json").read_text())["cross_suite"]["allowed_by_flag"] is True


def test_allow_cross_suite_keeps_internal_consistency_gates(tmp_path, suites, monkeypatch):
    decision, transfer = suites
    run = fake_run(tmp_path, "training-suite-sha", source_hashes())
    config = json.loads((run / "training_config.json").read_text())
    config["args"]["seed"] = 7  # tamper after hashing
    write_json(run / "training_config.json", config)
    monkeypatch.setattr(evaluate, "LocalPredictor", StubPredictor)
    with pytest.raises(ValueError, match="training config hash mismatch"):
        evaluate.evaluate(args(run=run, suite=decision, transfer=transfer, extra=["--out", str(tmp_path / "o"), "--allow-cross-suite"]))


class FakeKevPredictor:
    instances = []

    def __init__(self, reference, device):
        self.reference, self.device = reference, device
        self.metadata = {"base": BASE, "base_revision": REVISION, "suite_sha256": FakeKevPredictor.suite_sha256,
                         "checkpoint": {"repo": reference, "revision": "abc"}, "kev_source": {"commit": "20fa626"},
                         "training_args": {"seed": 0, "lr": 2e-4},
                         "option_isolation": False, "special_embeddings": False, "head_dim": 256, "lora": 16}
        FakeKevPredictor.instances.append(self)

    def __call__(self, request):
        weights = {"a": 0.6, "b": 0.3, "c": 0.1}
        return [[weights[key] for key in question["criteria"]] for question in request["questions"].values()], 0.0


@pytest.fixture
def fake_kev(monkeypatch, suites):
    decision, _ = suites
    FakeKevPredictor.suite_sha256 = digest(decision / "manifest.json")
    FakeKevPredictor.instances = []
    module = types.ModuleType("hev.kev")
    module.KevPredictor = FakeKevPredictor
    monkeypatch.setitem(sys.modules, "hev.kev", module)
    return FakeKevPredictor


def test_kev_requires_out_and_refuses_level_zero_before_any_load(tmp_path, suites, fake_kev):
    decision, transfer = suites
    with pytest.raises(ValueError, match="kev requires --out"):
        evaluate.evaluate(args(run="hf://jaredpalmer/kev-0.6b", suite=decision, transfer=transfer, extra=["--checkpoint-kind", "kev"]))
    with pytest.raises(ValueError, match="level-zero-ablation"):
        evaluate.evaluate(args(run="hf://jaredpalmer/kev-0.6b", suite=decision, transfer=transfer,
                               extra=["--checkpoint-kind", "kev", "--out", str(tmp_path / "o"), "--level-zero-ablation"]))
    assert fake_kev.instances == [] and not (tmp_path / "o").exists()


def test_kev_evaluates_through_kev_predictor_and_records_provenance(tmp_path, suites, fake_kev):
    decision, transfer = suites
    out = tmp_path / "kev-out"
    reference = "hf://jaredpalmer/kev-0.6b"
    result = evaluate.evaluate(args(run=reference, suite=decision, transfer=transfer, extra=["--checkpoint-kind", "kev", "--out", str(out)]))
    assert [(p.reference, p.device) for p in fake_kev.instances] == [(reference, "cpu")]
    assert result["run"] == reference and result["checkpoint_kind"] == "kev" and result["head"] == "kev"
    assert result["base"] == BASE and result["base_revision"] == REVISION
    assert result["training_config"] is None and result["training_metrics"] is None and result["cross_suite"] is None
    assert result["provenance"] == {
        "kind": "kev",
        "gates": {"training_config": "not applicable: external checkpoint",
                  "source_hashes": "not applicable: external checkpoint"},
        "checkpoint": {"repo": reference, "revision": "abc"},
        "kev_source": {"commit": "20fa626"},
        "training_args": {"seed": 0, "lr": 2e-4},
        # An order-sensitivity number is only interpretable against the packing that produced it,
        # so the scored checkpoint's architecture flags must survive into the artifact.
        "option_isolation": False,
        "special_embeddings": False,
        "head_dim": 256,
        "lora": 16,
    }
    assert result["training_suite_sha256"] == digest(decision / "manifest.json")
    assert result["decision"]["coverage"]["evaluated_questions"] == 2
    assert result["transfer"]["coverage"]["evaluated_questions"] == 2
    assert (out / "result.json").exists() and (out / "transfer_rows.json").exists()


def test_kev_suite_gate_still_applies(tmp_path, suites, fake_kev):
    decision, transfer = suites
    fake_kev.suite_sha256 = "kev-was-trained-elsewhere"
    with pytest.raises(ValueError, match="checkpoint and training suite hashes differ"):
        evaluate.evaluate(args(run="local/kev", suite=decision, transfer=transfer, extra=["--checkpoint-kind", "kev", "--out", str(tmp_path / "a")]))
    result = evaluate.evaluate(args(run="local/kev", suite=decision, transfer=transfer,
                                    extra=["--checkpoint-kind", "kev", "--out", str(tmp_path / "b"), "--allow-cross-suite"]))
    assert result["cross_suite"]["training_suite_sha256"] == "kev-was-trained-elsewhere"
    assert result["cross_suite"]["suite_differs"] and result["cross_suite"]["allowed_by_flag"]


def test_transfer_manifest_declaration_plus_non_trainability_is_the_rule(tmp_path, suites):
    decision, transfer = suites
    records = evaluate.validate_transfer_suite(decision, transfer, BASE, REVISION)
    assert {record["_meta"]["source"] for record in records} == {"legacy_holdout", "composition_holdout"}
    undeclared = write_suite(tmp_path / "undeclared", {"development": [choice_record("x", "legacy_holdout")]},
                             eval_only=True, eval_only_sources=["composition_holdout"])
    with pytest.raises(ValueError, match="legacy_holdout"):
        evaluate.validate_transfer_suite(decision, undeclared, BASE, REVISION)
    leaked = write_suite(tmp_path / "leaked", {"development": [choice_record("x", "banking77")]},
                         eval_only=True, eval_only_sources=["banking77"])
    with pytest.raises(ValueError, match="banking77"):
        evaluate.validate_transfer_suite(decision, leaked, BASE, REVISION)


def test_v4_holdouts_are_eval_only_and_training_refuses_them():
    assert EVAL_ONLY[-2:] == ("legacy_holdout", "composition_holdout")
    assert not set(EVAL_ONLY) & set(TRAINABLE)
    for source in ("legacy_holdout", "composition_holdout"):
        with pytest.raises(ValueError, match="eval-only"):
            validate_training_records([{"_meta": {"source": source}}])
        with pytest.raises(ValueError, match="eval-only"):
            validate_training_records([{"_meta": {"source": "boolq"}, "questions": {"q": {"src": source}}}])
