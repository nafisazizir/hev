"""Offline tests for the two-seed M3 replication aggregate: synthetic runs, stubbed loader and bootstrap."""
import json

import pytest

import hev.replicate as replicate
from hev.replicate import (CRITICAL_SOURCES, EVALUATE_HASHES, TOLERANCE, aggregate, mechanism_checks,
                           validate_compatibility, validate_evaluation_protocol)


def suite_report(accuracy=0.8, ece=0.03, brier=0.27, spread=1e-6, flips=0.0, packed=1e-6,
                 requested=10, evaluated=10, questions=30, answered=30, rejected=0, truncated=0):
    return {
        "raw": {"overall": {"accuracy": accuracy}},
        "calibrated": {"overall": {"ece": ece, "brier": brier}},
        "permutation": {"orders": 6, "argmax_flip_rate": flips, "p90_correct_probability_spread": spread},
        "bootstrap_raw": {"samples": 10000},
        "bootstrap_calibrated": {"samples": 10000},
        "packed_vs_separate": {"max_abs_probability_difference": packed},
        "coverage": {"requested_records": requested, "evaluated_records": evaluated,
                     "requested_questions": questions, "evaluated_questions": answered,
                     "rejected_records": rejected, "truncated_records": truncated},
    }


def result(decision_accuracy=0.8, transfer_accuracy=0.6, ece=0.03, brier=0.27, temperature=1.9,
           source_hashes=None, evaluate_hash=None, **decision_overrides):
    hashes = source_hashes or {name: f"hash-{name}" for name in CRITICAL_SOURCES}
    hashes = {**hashes, "hev/evaluate.py": evaluate_hash or EVALUATE_HASHES["seed0"]}
    return {
        "base": "Qwen/pinned",
        "base_revision": "rev",
        "training_suite_sha256": "train-sha",
        "transfer_suite_sha256": "transfer-sha",
        "temperature_fit": {"temperature": temperature},
        "evaluation_config": {"seed": 1, "permutations": 6, "bootstrap_samples": 10000,
                              "level_zero_ablation": False},
        "training_config": {"args": {"seed": 0},
                            "provenance": {"source_hashes": hashes}},
        "decision": suite_report(accuracy=decision_accuracy, ece=ece, brier=brier, **decision_overrides),
        "transfer": suite_report(accuracy=transfer_accuracy, packed=None),
    }


def run(tmp_path, name, res, rows=None):
    directory = tmp_path / name
    directory.mkdir()
    result_path = directory / "result.json"
    result_path.write_text(json.dumps(res))
    return {"path": directory, "result_path": result_path, "result": res,
            "rows": rows or {"decision": [{"seed": name}], "transfer": [{"seed": name}]}}


def test_mechanism_checks_tolerance_boundary_and_coverage():
    checks = mechanism_checks(result(spread=TOLERANCE, packed=TOLERANCE))
    assert all(checks.values()) and len(checks) == 8
    for name in ("decision", "transfer"):
        assert {f"{name}_coverage", f"{name}_zero_flips", f"{name}_p90_spread", f"{name}_packed"} <= set(checks)
    failed = mechanism_checks(result(spread=TOLERANCE * 2))
    assert not failed["decision_p90_spread"] and failed["transfer_p90_spread"]
    incomplete = mechanism_checks(result(evaluated=9))
    assert not incomplete["decision_coverage"] and incomplete["transfer_coverage"]


def test_validate_evaluation_protocol_orders_bootstrap_and_config():
    validate_evaluation_protocol(result())
    validate_evaluation_protocol(result(), require_config=True)
    wrong_orders = result()
    wrong_orders["decision"]["permutation"]["orders"] = 4
    with pytest.raises(ValueError, match="permutation"):
        validate_evaluation_protocol(wrong_orders)
    wrong_samples = result()
    wrong_samples["transfer"]["bootstrap_calibrated"]["samples"] = 1000
    with pytest.raises(ValueError, match="bootstrap"):
        validate_evaluation_protocol(wrong_samples)
    wrong_seed = result()
    wrong_seed["evaluation_config"]["seed"] = 0
    validate_evaluation_protocol(wrong_seed)
    with pytest.raises(ValueError, match="protocol"):
        validate_evaluation_protocol(wrong_seed, require_config=True)


def test_validate_compatibility_provenance_and_critical_sources(tmp_path):
    seed0 = run(tmp_path, "s0", result(evaluate_hash=EVALUATE_HASHES["seed0"]))
    seed1 = run(tmp_path, "s1", result(evaluate_hash=EVALUATE_HASHES["seed1"]))
    validate_compatibility(seed0, seed1)
    unrecognized = run(tmp_path, "sx", result(evaluate_hash="unaudited"))
    with pytest.raises(ValueError, match="metadata-only"):
        validate_compatibility(seed0, unrecognized)
    changed = run(tmp_path, "s2", result(source_hashes={**{name: "h" for name in CRITICAL_SOURCES},
                                                        "hev/model.py": "different"},
                                         evaluate_hash=EVALUATE_HASHES["seed1"]))
    with pytest.raises(ValueError, match="hev/model.py"):
        validate_compatibility(seed0, changed)
    drifted = run(tmp_path, "s3", {**result(evaluate_hash=EVALUATE_HASHES["seed1"]), "base_revision": "other"})
    with pytest.raises(ValueError, match="provenance"):
        validate_compatibility(seed0, drifted)


def stubbed_aggregate(monkeypatch, tmp_path, res1=None, calls=None):
    seed0 = run(tmp_path, "seed0", result(decision_accuracy=0.8, transfer_accuracy=0.6, ece=0.04, brier=0.28,
                                          evaluate_hash=EVALUATE_HASHES["seed0"]),
                rows={"decision": [{"s": 0, "split": "d"}], "transfer": [{"s": 0, "split": "t"}]})
    seed1 = run(tmp_path, "seed1", res1 or result(decision_accuracy=0.9, transfer_accuracy=0.7, ece=0.02,
                                                  brier=0.26, temperature=2.1,
                                                  evaluate_hash=EVALUATE_HASHES["seed1"]),
                rows={"decision": [{"s": 1, "split": "d"}], "transfer": [{"s": 1, "split": "t"}]})
    runs = {"seed0": seed0, "seed1": seed1}

    def loader(path, head, expected_seed=0):
        assert head == "pointer"
        return runs["seed0" if expected_seed == 0 else "seed1"]

    monkeypatch.setattr(replicate, "load_hev_run", loader)
    monkeypatch.setattr(replicate, "paired_bootstrap",
                        lambda left, right, temperature, samples, seed:
                        calls.append((left, right, temperature, samples, seed)) or {"nll": {"estimate": 0.0}})
    return aggregate("ignored-s0", "ignored-s1", samples=7, seed=42)


def test_aggregate_seeds_means_ranges_and_paired_direction(monkeypatch, tmp_path):
    calls = []
    out = stubbed_aggregate(monkeypatch, tmp_path, calls=calls)
    assert out["status"] == "success" and out["test_evaluated"] is False
    assert out["seeds"]["seed0"]["seed"] == 0 and out["seeds"]["seed1"]["seed"] == 1
    assert out["seeds"]["seed1"]["temperature"] == 2.1
    expected = {"decision_accuracy": (0.85, [0.8, 0.9]), "transfer_accuracy": (0.65, [0.6, 0.7]),
                "decision_calibrated_ece": (0.03, [0.02, 0.04]),
                "decision_calibrated_brier": (0.27, [0.26, 0.28])}
    for name, (mean, bounds) in expected.items():
        assert out["across_seed"][name]["mean"] == pytest.approx(mean)
        assert out["across_seed"][name]["range"] == bounds
    assert out["protocol"]["paired_direction"] == "seed1 minus seed0; descriptive only"
    assert out["protocol"]["replication_seed"] == 1
    assert out["provenance"]["evaluate_source_exception"]["hashes"] == EVALUATE_HASHES
    assert out["replication_passed"] is True
    assert set(out["paired_seed1_minus_seed0_raw"]) == {"decision", "transfer"}
    for (left, right, temperature, samples, seed), split in zip(calls, ("decision", "transfer")):
        assert left == [{"s": 1, "split": split[0]}] and right == [{"s": 0, "split": split[0]}]
        assert temperature == 1.0 and samples == 7 and seed == 42


def test_seed1_mechanism_failure_reports_success_but_not_passed(monkeypatch, tmp_path):
    calls = []
    broken = result(decision_accuracy=0.9, transfer_accuracy=0.7, ece=0.02, brier=0.26,
                    temperature=2.1, spread=1e-2, evaluate_hash=EVALUATE_HASHES["seed1"])
    out = stubbed_aggregate(monkeypatch, tmp_path, res1=broken, calls=calls)
    assert out["status"] == "success"
    assert out["seeds"]["seed1"]["mechanism_passed"] is False
    assert out["seeds"]["seed0"]["mechanism_passed"] is True
    assert out["replication_passed"] is False
