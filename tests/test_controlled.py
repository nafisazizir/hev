"""Offline tests for the M4b controlled aggregate (D15): three same-suite Hev seeds against the released kev."""
import json

import numpy as np
import pytest

import hev.controlled as controlled
import hev.released as released
from hev.controlled import (EVALUATION_CONFIG, EXPECTED_OPTIMIZER_STEPS, HEV_SEEDS, aggregate, main,
                            three_seed_summary, validate_hev_run)
from hev.released import DECISION_SUITE_SHA256, KEV_HUB_COMMIT, TRANSFER_SUITE_SHA256, verdict
from hev.suite import digest
from hev.train import KEV_V4_RECIPE
from tests.test_released import GROUPS, clean_accuracy, kev_style_result, make_rows, permutation_summary


def hev_result(seed):
    return {
        "head": "pointer",
        "cross_suite": None,
        "evaluation_config": dict(EVALUATION_CONFIG),
        "training_config": {
            "suite_sha256": DECISION_SUITE_SHA256,
            "args": {"seed": seed, "batch": 1, "accum": 8, "p_none_pair": 0.25, "recipe": "kev-v4"},
            "train_partition": {"path": "evals/decision-v4/train.jsonl", "sha256": "cb55b79e", "records": 10896},
            "recipe": {"name": "kev-v4", "matched": dict(KEV_V4_RECIPE["matched"]),
                       "accepted_deviations": [{"knob": "precision", "kev": "bf16 autocast", "hev": "fp32"}]},
            "provenance": {"device": "mps", "dtype": "fp32", "gpu": None, "git": {"commit": "abc123", "dirty": False}},
        },
        "training_metrics": {"status": "success", "optimizer_steps": EXPECTED_OPTIMIZER_STEPS, "none_pair_records": 2600,
                             "wall_seconds": 4000.0},
    }


def write_eval_dir(tmp_path, name, kind, rows, temperature=1.5, flips=0.0, spread=1e-6, packed=1e-6, seed=0, mutate=None):
    directory = tmp_path / name
    directory.mkdir()
    result = {
        "schema_version": 1, "status": "success", "test_evaluated": False,
        "training_suite_sha256": DECISION_SUITE_SHA256, "transfer_suite_sha256": TRANSFER_SUITE_SHA256,
        "temperature_fit": {"temperature": temperature, "n": 10},
        "decision": {"permutation": permutation_summary(flips, spread),
                     "packed_vs_separate": {"n_records": 2, "n_questions": 4, "max_abs_probability_difference": packed}},
        "transfer": {"permutation": permutation_summary(flips, spread),
                     "packed_vs_separate": {"n_records": 0, "n_questions": 0, "max_abs_probability_difference": None}},
    }
    if kind == "kev":
        result["provenance"] = {"kind": "kev", "option_isolation": False,
                                "checkpoint": {"hub_commit": KEV_HUB_COMMIT, "resolved_path": "/cache/snap"},
                                "training_args": {"batch": 8, "accum": 1, "dtype": "bf16", "p_none_pair": 0.25}}
    else:
        result.update(hev_result(seed))
    if mutate:
        mutate(result)
    result["artifact_hashes"] = {}
    for split, split_rows in rows.items():
        path = directory / f"{split}_rows.json"
        path.write_text(json.dumps(split_rows))
        result["artifact_hashes"][split] = {"path": path.name, "sha256": digest(path), "rows": len(split_rows)}
    (directory / "result.json").write_text(json.dumps(result))
    return directory


@pytest.fixture
def world(tmp_path, monkeypatch):
    rows = {"kev": {"calibration": make_rows("decision", 0.9, 1), "decision": make_rows("decision", 0.95, 2),
                    "transfer": make_rows("transfer", 0.7, 3)}}
    for index, name in enumerate(HEV_SEEDS):
        rows[name] = {"calibration": make_rows("decision", 0.8, 10 + index),
                      "decision": make_rows("decision", 0.6 + 0.1 * index, 20 + index),
                      "transfer": make_rows("transfer", 0.65, 30 + index)}
    dirs = {"kev": write_eval_dir(tmp_path, "kev", "kev", rows["kev"], temperature=1.2)}
    for index, name in enumerate(HEV_SEEDS):
        dirs[name] = write_eval_dir(tmp_path, f"hev{index}", "hev", rows[name], temperature=1.5 + index / 10, seed=index)
    hub = tmp_path / "hub_result.json"
    hub.write_text(json.dumps(kev_style_result(clean_accuracy(rows["kev"]["decision"]),
                                               clean_accuracy(rows["kev"]["transfer"]), 0)))
    seed1 = tmp_path / "seed1.json"
    seed1.write_text(json.dumps(kev_style_result(0.80, 0.60, 1)))
    seed2 = tmp_path / "seed2.json"
    seed2.write_text(json.dumps(kev_style_result(0.82, 0.58, 2)))
    monkeypatch.setattr(released, "KEV_FILE_HASHES", {"hub": digest(hub), "seed1": digest(seed1), "seed2": digest(seed2)})
    return {"rows": rows, "dirs": dirs, "hub": hub, "seed1": seed1, "seed2": seed2, "tmp": tmp_path}


def run_aggregate(world, samples=50):
    return aggregate(world["dirs"]["kev"], [world["dirs"][name] for name in HEV_SEEDS], world["hub"],
                     world["seed1"], world["seed2"], samples=samples, seed=7)


def cli_args(world, out):
    return ["--kev", str(world["dirs"]["kev"]), "--hev-seed0", str(world["dirs"]["hev_seed0"]),
            "--hev-seed1", str(world["dirs"]["hev_seed1"]), "--hev-seed2", str(world["dirs"]["hev_seed2"]),
            "--kev-hub-result", str(world["hub"]), "--kev-seed1", str(world["seed1"]),
            "--kev-seed2", str(world["seed2"]), "--out", str(out), "--bootstrap-samples", "40", "--seed", "3"]


def replace_hev(world, name, mutate, **kwargs):
    index = HEV_SEEDS.index(name)
    world["dirs"][name] = write_eval_dir(world["tmp"], f"{name}-{len(list(world['tmp'].iterdir()))}", "hev",
                                         world["rows"][name], seed=index, mutate=mutate, **kwargs)


def test_hev_validation_requires_same_suite_recipe_seed_and_evaluation_settings(world):
    ok = validate_hev_run(released.load_evaluation(world["dirs"]["hev_seed2"], "hev_seed2"), 2)
    assert ok["training_seed"] == 2 and ok["recipe"] == "kev-v4" and ok["optimizer_steps"] == EXPECTED_OPTIMIZER_STEPS
    assert ok["training_suite_sha256"] == DECISION_SUITE_SHA256 and ok["dtype"] == "fp32"
    assert ok["accepted_deviations"][0]["knob"] == "precision" and ok["git_commit"] == "abc123"
    cases = [
        (lambda r: r.update(cross_suite={"allowed_by_flag": True}), "must not use --allow-cross-suite"),
        (lambda r: r.update(head="set"), "expected 'pointer'"),
        (lambda r: r["training_config"].update(suite_sha256="0" * 64), "not decision-v4"),
        (lambda r: r["training_config"]["recipe"].update(name="none"), "not the kev-v4 recipe"),
        (lambda r: r["training_config"]["recipe"]["matched"].update(lr=1e-4), "differ from kev's published trial"),
        (lambda r: r["training_config"]["args"].update(seed=5), "expected 2"),
        (lambda r: r["training_config"]["train_partition"].update(records=3432), "expected 10896"),
        (lambda r: r["training_metrics"].update(status="failed_loss_gate"), "expected 'success'"),
        (lambda r: r["training_metrics"].update(optimizer_steps=2723), "expected 2724"),
        (lambda r: r["evaluation_config"].update(permutations=2), "is not D13's"),
    ]
    for mutate, message in cases:
        replace_hev(world, "hev_seed2", mutate)
        with pytest.raises(ValueError, match=message):
            run_aggregate(world)


def test_three_seed_summary_is_mean_and_range_without_pooled_interval():
    entry = lambda estimate, lo, hi: {"estimate": estimate, "ci95": [lo, hi], "ci90": [lo + 0.01, hi - 0.01]}
    paired = {name: {split: {"raw": {"accuracy": entry(value, value - 0.03, value + 0.03), "n": 5},
                             "calibrated": {"nll": entry(-value, -0.2, 0.0)}} for split in ("decision", "transfer")}
              for name, value in zip(HEV_SEEDS, (0.04, 0.02, 0.06))}
    summary = three_seed_summary(paired)
    accuracy = summary["decision"]["raw"]["accuracy"]
    assert accuracy["mean_point_delta"] == pytest.approx(0.04)
    assert accuracy["min_point_delta"] == 0.02 and accuracy["max_point_delta"] == 0.06
    assert accuracy["range_point_delta"] == pytest.approx(0.04)
    assert accuracy["seed_point_deltas"] == [0.04, 0.02, 0.06]
    assert accuracy["pooled_interval"] is None
    assert accuracy["intervals"]["hev_seed2"]["ci95"] == pytest.approx([0.03, 0.09])
    assert summary["transfer"]["calibrated"]["nll"]["mean_point_delta"] == pytest.approx(-0.04)
    assert "no seed selected" in summary["note"] and "n" not in summary["decision"]["raw"]


def test_aggregate_artifact_schema_and_contents(world):
    result = run_aggregate(world, samples=60)
    expected_keys = {"schema_version", "protocol", "status", "test_evaluated", "inputs", "suites", "provenance",
                     "bootstrap", "populations", "population_note", "sanity_gate", "mechanism_checks_passed", "models",
                     "paired_kev_minus_hev", "three_seed_summary", "seed_spread", "verdict_rule", "margins", "verdicts",
                     "order_study", "kev_seeds_point_only", "training_data", "caveats"}
    assert expected_keys <= set(result)
    assert result["protocol"] == "D15" and result["status"] == "success" and result["test_evaluated"] is False
    assert result["suites"]["hev_training_suite_sha256"] == DECISION_SUITE_SHA256
    assert result["mechanism_checks_passed"] is True
    assert set(result["models"]) == {"kev", *HEV_SEEDS} and set(result["verdicts"]) == set(HEV_SEEDS)
    assert set(result["provenance"]) == {"kev", *HEV_SEEDS}
    assert [result["provenance"][name]["training_seed"] for name in HEV_SEEDS] == [0, 1, 2]
    for name in HEV_SEEDS:
        assert result["models"][name]["decision"]["primary"]["n_clean"] == 2 * GROUPS
        for split in ("decision", "transfer"):
            raw = result["paired_kev_minus_hev"][name][split]["raw"]
            assert raw["n"] == 2 * GROUPS and raw["direction"] == "kev minus Hev"
            assert result["verdicts"][name][split] == verdict(raw["accuracy"]["ci95"], raw["accuracy"]["ci90"],
                                                             released.MARGINS[split])
    kev_primary = clean_accuracy(released.partition(world["rows"]["kev"]["decision"], "decision", "primary"))
    hev1_primary = clean_accuracy(released.partition(world["rows"]["hev_seed1"]["decision"], "decision", "primary"))
    assert result["paired_kev_minus_hev"]["hev_seed1"]["decision"]["raw"]["accuracy"]["estimate"] == pytest.approx(
        kev_primary - hev1_primary)
    summary = result["three_seed_summary"]["decision"]["raw"]["accuracy"]
    deltas = [result["paired_kev_minus_hev"][name]["decision"]["raw"]["accuracy"]["estimate"] for name in HEV_SEEDS]
    assert summary["seed_point_deltas"] == deltas and summary["mean_point_delta"] == pytest.approx(np.mean(deltas))
    assert summary["max_point_delta"] == max(deltas) and summary["pooled_interval"] is None

    seed_spread = result["seed_spread"]["decision"]
    hev_all = [clean_accuracy(world["rows"][name]["decision"]) for name in HEV_SEEDS]
    assert seed_spread["hev"]["values"] == pytest.approx(hev_all)
    assert seed_spread["hev"]["range"] == pytest.approx(max(hev_all) - min(hev_all))
    assert seed_spread["kev"]["values"] == pytest.approx([clean_accuracy(world["rows"]["kev"]["decision"]), 0.80, 0.82])
    assert seed_spread["mean_difference_kev_minus_hev"] == pytest.approx(np.mean(seed_spread["kev"]["values"]) - np.mean(hev_all))
    assert "no interval and no verdict" in result["seed_spread"]["note"]

    order = result["order_study"]
    checks = order["hev_seed0"]["mechanism_check"]["checks"]
    assert checks["decision_packed_vs_separate"] and checks["transfer_packed_vs_separate"] and checks["decision_zero_flips"]
    assert order["kev"]["mechanism_check"] is None
    training = result["training_data"]
    assert training["shared"]["records"] == 10896 and training["shared"]["minimal_pairs"] is True
    assert training["shared"]["recipe"] == KEV_V4_RECIPE["matched"]
    assert training["hev"]["recorded_provenance"][2]["git_commit"] == "abc123"
    assert training["hev"]["batch"] == 1 and training["kev"]["batch"] == 8
    assert result["kev_seeds_point_only"]["seeds"]["seed2"]["decision_clean_accuracy"] == 0.82


def test_mechanism_check_fails_on_packed_difference_or_flips(world):
    replace_hev(world, "hev_seed1", None, packed=5e-4)
    result = run_aggregate(world)
    check = result["order_study"]["hev_seed1"]["mechanism_check"]
    assert not check["passed"] and not check["checks"]["decision_packed_vs_separate"]
    assert result["mechanism_checks_passed"] is False and result["status"] == "success"
    replace_hev(world, "hev_seed1", None, flips=1 / 12)
    result = run_aggregate(world)
    assert not result["order_study"]["hev_seed1"]["mechanism_check"]["checks"]["decision_zero_flips"]
    assert result["order_study"]["hev_seed0"]["mechanism_check"]["passed"]


def test_cli_writes_artifacts_and_refuses_existing_out(world):
    out = world["tmp"] / "aggregate"
    result = main(cli_args(world, out))
    assert result["status"] == "success"
    written = json.loads((out / "result.json").read_text())
    assert written["protocol"] == "D15" and written["bootstrap"]["samples"] == 40
    assert written["artifact_hashes"]["summary"]["sha256"] == digest(out / "summary.md")
    summary = (out / "summary.md").read_text()
    assert "| hev_seed2 |" in summary and "Three-seed summary" in summary and "Seed spread" in summary
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(cli_args(world, out))
    assert written == json.loads((out / "result.json").read_text())


def test_cli_sanity_gate_failure_keeps_artifact_and_exits_nonzero(world):
    hub = json.loads(world["hub"].read_text())
    hub["clean"]["acc"] -= 0.02
    world["hub"].write_text(json.dumps(hub))
    released.KEV_FILE_HASHES["hub"] = digest(world["hub"])
    out = world["tmp"] / "failed-gate"
    with pytest.raises(SystemExit) as info:
        main(cli_args(world, out))
    assert info.value.code == 1
    written = json.loads((out / "result.json").read_text())
    assert written["status"] == "failed" and written["verdicts"] is None and "FAILED" in (out / "summary.md").read_text()


def test_cli_records_failure_json_on_error(world):
    replace_hev(world, "hev_seed0", lambda result: result.update(status="running"))
    out = world["tmp"] / "broken"
    with pytest.raises(ValueError, match="expected 'success'"):
        main(cli_args(world, out))
    failure = json.loads((out / "failure.json").read_text())
    assert failure["error_type"] == "ValueError" and not (out / "result.json").exists()
