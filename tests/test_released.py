"""Offline tests for the M4 released-kev aggregate: synthetic evaluation dirs, sanity gate, verdicts, populations."""
import json

import numpy as np
import pytest

import hev.released as released
from hev.released import (DECISION_SUITE_SHA256, HEV_TRAINING_SUITE_SHA256, KEV_HUB_COMMIT, TRANSFER_SUITE_SHA256,
                          aggregate, main, partition, sanity_gate, two_seed_summary, verdict)
from hev.suite import digest

SOURCES = {"decision": ("banking77", "agnews", "legacy_policy"), "transfer": ("mmlu", "sciq", "legacy_holdout")}
GROUPS = 6
KEYS = ["a", "b", "c"]


def make_rows(split, hit_rate, seed):
    """One clean choice row per (source, group) plus one permuted row per source; correctness follows hit_rate."""
    rng = np.random.default_rng(seed)
    rows = []
    for source in SOURCES[split]:
        for index in range(GROUPS + 1):
            variant = "clean" if index < GROUPS else "permuted"
            label = index % 3
            winner = label if rng.random() < hit_rate else (label + 1) % 3
            probabilities = np.full(3, 0.1)
            probabilities[winner] = 0.8
            rows.append({
                "id": f"{source}/dev/{index}", "group": f"{source}/dev/{index}", "question": "label",
                "source": source, "task": source, "type": "choice", "variant": variant, "pair_id": None,
                "sibling": None, "parent": f"{source}/dev/{index}", "keys": list(KEYS), "label": label,
                "probabilities": probabilities.tolist(), "predicted_key": KEYS[winner],
                "correct": bool(winner == label), "confidence": 0.8,
            })
    return rows


def permutation_summary(flips=0.0, spread=1e-6, n=12):
    return {"n": n, "orders": 6, "argmax_flip_rate": flips, "mean_correct_probability_spread": spread,
            "p90_correct_probability_spread": spread, "max_correct_probability_spread": spread}


def write_eval_dir(tmp_path, name, kind, rows, temperature=1.5, flips=0.0, spread=1e-6, mutate=None):
    directory = tmp_path / name
    directory.mkdir()
    result = {
        "schema_version": 1, "status": "success", "test_evaluated": False,
        "training_suite_sha256": DECISION_SUITE_SHA256, "transfer_suite_sha256": TRANSFER_SUITE_SHA256,
        "temperature_fit": {"temperature": temperature, "n": 10},
        "decision": {"permutation": permutation_summary(flips, spread),
                     "packed_vs_separate": {"n_records": 2, "n_questions": 4, "max_abs_probability_difference": 1e-6}},
        "transfer": {"permutation": permutation_summary(flips, spread),
                     "packed_vs_separate": {"n_records": 0, "n_questions": 0, "max_abs_probability_difference": None}},
    }
    if kind == "kev":
        result["provenance"] = {"kind": "kev", "option_isolation": False,
                                "checkpoint": {"hub_commit": KEV_HUB_COMMIT, "resolved_path": "/cache/snap"},
                                "training_args": {"batch": 8, "accum": 1, "dtype": "bf16", "p_none_pair": 0.25}}
    else:
        result["head"] = "pointer"
        result["cross_suite"] = {"allowed_by_flag": True, "training_suite_sha256": HEV_TRAINING_SUITE_SHA256}
        result["training_config"] = {"suite_sha256": HEV_TRAINING_SUITE_SHA256,
                                     "args": {"seed": 0, "batch": 1, "accum": 8},
                                     "provenance": {"device": "mps", "dtype": "float32", "gpu": None}}
    if mutate:
        mutate(result)
    result["artifact_hashes"] = {}
    for split, split_rows in rows.items():
        path = directory / f"{split}_rows.json"
        path.write_text(json.dumps(split_rows))
        result["artifact_hashes"][split] = {"path": path.name, "sha256": digest(path), "rows": len(split_rows)}
    (directory / "result.json").write_text(json.dumps(result))
    return directory


def kev_style_result(decision_acc, transfer_acc, seed):
    return {"test_evaluated": False, "clean": {"acc": decision_acc, "n": 12},
            "transfer": {"clean": {"acc": transfer_acc, "n": 12}, "suite_sha256": TRANSFER_SUITE_SHA256},
            "provenance": {"suite_sha256": DECISION_SUITE_SHA256, "config": {"seed": seed}}}


def clean_accuracy(rows):
    clean = [row for row in rows if row["variant"] == "clean"]
    return sum(row["correct"] for row in clean) / len(clean)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Three evaluation dirs, a matching fake Hub result and two fake kev seed files, with hashes pinned."""
    rows = {
        "kev": {"calibration": make_rows("decision", 0.9, 1), "decision": make_rows("decision", 0.95, 2),
                "transfer": make_rows("transfer", 0.7, 3)},
        "hev_seed0": {"calibration": make_rows("decision", 0.8, 4), "decision": make_rows("decision", 0.6, 5),
                      "transfer": make_rows("transfer", 0.65, 6)},
        "hev_seed1": {"calibration": make_rows("decision", 0.8, 7), "decision": make_rows("decision", 0.7, 8),
                      "transfer": make_rows("transfer", 0.6, 9)},
    }
    dirs = {"kev": write_eval_dir(tmp_path, "kev", "kev", rows["kev"], temperature=1.2),
            "hev_seed0": write_eval_dir(tmp_path, "hev0", "hev", rows["hev_seed0"], temperature=1.9),
            "hev_seed1": write_eval_dir(tmp_path, "hev1", "hev", rows["hev_seed1"], temperature=2.1)}
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
    return aggregate(world["dirs"]["kev"], world["dirs"]["hev_seed0"], world["dirs"]["hev_seed1"],
                     world["hub"], world["seed1"], world["seed2"], samples=samples, seed=7)


def cli_args(world, out):
    return ["--kev", str(world["dirs"]["kev"]), "--hev-seed0", str(world["dirs"]["hev_seed0"]),
            "--hev-seed1", str(world["dirs"]["hev_seed1"]), "--kev-hub-result", str(world["hub"]),
            "--kev-seed1", str(world["seed1"]), "--kev-seed2", str(world["seed2"]),
            "--out", str(out), "--bootstrap-samples", "40", "--seed", "3"]


def test_verdict_rule_on_constructed_intervals():
    equivalent = verdict([-0.015, 0.012], [-0.01, 0.008], 0.02)
    assert equivalent["verdict"] == "equivalent" and equivalent["direction"] is None and equivalent["within_margin"]
    better = verdict([0.03, 0.08], [0.035, 0.075], 0.02)
    assert better["verdict"] == "kev_better" and better["direction"] == "kev_better" and not better["within_margin"]
    worse = verdict([-0.08, -0.03], [-0.075, -0.035], 0.03)
    assert worse["verdict"] == "hev_better" and worse["direction"] == "hev_better"
    inconclusive = verdict([-0.01, 0.05], [-0.005, 0.045], 0.02)
    assert inconclusive["verdict"] == "inconclusive" and inconclusive["direction"] is None
    assert not inconclusive["within_margin"]
    both = verdict([0.002, 0.018], [0.004, 0.016], 0.02)
    assert both["direction"] == "kev_better" and both["within_margin"] is True
    assert both["verdict"] == "kev_better_within_margin"
    boundary = verdict([-0.01, 0.025], [-0.02, 0.02], 0.02)
    assert boundary["within_margin"] and boundary["verdict"] == "equivalent"


def test_partition_primary_secondary_all_counts():
    rows = make_rows("decision", 0.9, 0)
    primary = partition(rows, "decision", "primary")
    secondary = partition(rows, "decision", "secondary")
    everything = partition(rows, "decision", "all")
    assert len(primary) == 2 * (GROUPS + 1) and {row["source"] for row in primary} == {"banking77", "agnews"}
    assert len(secondary) == GROUPS + 1 and {row["source"] for row in secondary} == {"legacy_policy"}
    assert len(everything) == len(rows) == 3 * (GROUPS + 1)
    transfer = make_rows("transfer", 0.9, 0)
    assert {row["source"] for row in partition(transfer, "transfer", "secondary")} == {"legacy_holdout"}
    with pytest.raises(ValueError, match="outside the declared populations"):
        partition(rows + [{**rows[0], "source": "mystery"}], "decision", "all")


def test_two_seed_summary_is_mean_of_point_deltas_without_pooled_interval():
    entry = lambda estimate, lo, hi: {"estimate": estimate, "ci95": [lo, hi], "ci90": [lo + 0.01, hi - 0.01]}
    paired = {
        "hev_seed0": {split: {"raw": {"accuracy": entry(0.04, 0.0, 0.08), "n": 5},
                              "calibrated": {"nll": entry(-0.1, -0.2, 0.0)}} for split in ("decision", "transfer")},
        "hev_seed1": {split: {"raw": {"accuracy": entry(0.02, -0.02, 0.06), "n": 5},
                              "calibrated": {"nll": entry(-0.3, -0.4, -0.2)}} for split in ("decision", "transfer")},
    }
    summary = two_seed_summary(paired)
    accuracy = summary["decision"]["raw"]["accuracy"]
    assert accuracy["mean_point_delta"] == pytest.approx(0.03)
    assert accuracy["seed_point_deltas"] == [0.04, 0.02]
    assert accuracy["pooled_interval"] is None
    assert accuracy["intervals"]["hev_seed0"]["ci95"] == [0.0, 0.08]
    assert accuracy["intervals"]["hev_seed1"]["ci90"] == pytest.approx([-0.01, 0.05])
    assert summary["transfer"]["calibrated"]["nll"]["mean_point_delta"] == pytest.approx(-0.2)
    assert "n" not in summary["decision"]["raw"]


def test_sanity_gate_pass_and_fail(world):
    kev = released.load_evaluation(world["dirs"]["kev"], "kev")
    hub = json.loads(world["hub"].read_text())
    passed = sanity_gate(kev, hub)
    assert passed["passed"] and passed["tolerance"] == 0.01
    assert passed["hev_measured"]["decision"] == pytest.approx(hub["clean"]["acc"])
    assert passed["kev_reported"]["transfer"] == hub["transfer"]["clean"]["acc"]
    assert "variant == 'clean'" in passed["n_definition"]
    hub["transfer"]["clean"]["acc"] += 0.05
    failed = sanity_gate(kev, hub)
    assert not failed["passed"] and failed["checks"]["decision"]["passed"] and not failed["checks"]["transfer"]["passed"]
    assert failed["checks"]["transfer"]["abs_difference"] == pytest.approx(0.05)


def test_alignment_mismatch_is_an_error(world):
    rows = world["rows"]["hev_seed1"]
    shifted = {**rows, "decision": [{**row, "label": (row["label"] + 1) % 3} if row["id"].endswith("/2") else row
                                    for row in rows["decision"]]}
    world["dirs"]["hev_seed1"] = write_eval_dir(world["tmp"], "hev1-bad-label", "hev", shifted)
    with pytest.raises(ValueError, match="decision rows of hev_seed1 do not align"):
        run_aggregate(world)
    dropped = {**rows, "transfer": rows["transfer"][1:]}
    world["dirs"]["hev_seed1"] = write_eval_dir(world["tmp"], "hev1-dropped", "hev", dropped)
    with pytest.raises(ValueError, match="transfer rows of hev_seed1 do not align"):
        run_aggregate(world)


def test_validation_names_missing_or_wrong_keys(world):
    rows = world["rows"]["hev_seed0"]
    world["dirs"]["hev_seed0"] = write_eval_dir(world["tmp"], "hev0-noflag", "hev", rows,
                                                mutate=lambda result: result.pop("cross_suite"))
    with pytest.raises(ValueError, match="cross_suite.allowed_by_flag"):
        run_aggregate(world)

    def wrong_training(result):
        result["cross_suite"]["training_suite_sha256"] = "0" * 64
        result["training_config"]["suite_sha256"] = "0" * 64
    world["dirs"]["hev_seed0"] = write_eval_dir(world["tmp"], "hev0-wrongsuite", "hev", rows, mutate=wrong_training)
    with pytest.raises(ValueError, match="not decision-v2"):
        run_aggregate(world)
    world["dirs"]["hev_seed0"] = world["tmp"] / "hev0"

    def isolated(result):
        result["provenance"]["option_isolation"] = True
    world["dirs"]["kev"] = write_eval_dir(world["tmp"], "kev-isolated", "kev", world["rows"]["kev"], mutate=isolated)
    with pytest.raises(ValueError, match="option_isolation"):
        run_aggregate(world)

    def wrong_suite(result):
        result["transfer_suite_sha256"] = "1" * 64
    world["dirs"]["kev"] = write_eval_dir(world["tmp"], "kev-wrongsuite", "kev", world["rows"]["kev"], mutate=wrong_suite)
    with pytest.raises(ValueError, match="not transfer-v4"):
        run_aggregate(world)

    def failed(result):
        result["status"] = "failed"
    world["dirs"]["kev"] = write_eval_dir(world["tmp"], "kev-failed", "kev", world["rows"]["kev"], mutate=failed)
    with pytest.raises(ValueError, match="expected 'success'"):
        run_aggregate(world)


def test_kev_seed_files_are_hash_verified(world, monkeypatch):
    monkeypatch.setattr(released, "KEV_FILE_HASHES", {**released.KEV_FILE_HASHES, "seed2": "f" * 64})
    with pytest.raises(ValueError, match="kev seed2 result hash mismatch"):
        run_aggregate(world)


def test_aggregate_artifact_schema_and_contents(world):
    result = run_aggregate(world, samples=60)
    expected_keys = {"schema_version", "protocol", "status", "test_evaluated", "inputs", "suites", "provenance",
                     "bootstrap", "populations", "sanity_gate", "models", "paired_kev_minus_hev", "two_seed_summary",
                     "verdict_rule", "margins", "verdicts", "order_study", "kev_seeds_point_only", "training_data",
                     "caveats"}
    assert expected_keys <= set(result)
    assert result["protocol"] == "D13" and result["status"] == "success" and result["test_evaluated"] is False
    assert result["inputs"]["kev"]["result_sha256"] == digest(world["dirs"]["kev"] / "result.json")
    assert result["bootstrap"] == {"samples": 60, "seed": 7, "unit": "(source, group_id) clusters",
                                   "stratification": "source",
                                   "intervals": "percentile 95% and 90% from the same draws"}
    assert result["populations"]["primary"]["decision"] == sorted(released.PRIMARY_SOURCES["decision"])
    assert result["populations"]["secondary"]["transfer"] == ["composition_holdout", "legacy_holdout"]
    for name in ("kev", "hev_seed0", "hev_seed1"):
        model = result["models"][name]
        assert model["decision"]["primary"]["n_clean"] == 2 * GROUPS
        assert model["decision"]["secondary"]["n_clean"] == GROUPS
        assert model["decision"]["all"]["n_clean"] == 3 * GROUPS
        assert model["decision"]["all"]["n_rows"] == 3 * (GROUPS + 1)
        for population in ("primary", "secondary", "all"):
            report = model["transfer"][population]
            assert set(report["raw"]) >= {"overall", "by_source", "by_task", "by_type"}
            assert set(report["bootstrap_calibrated"]) >= {"accuracy", "nll", "brier", "ece"}
            assert report["none_diagnostics"]["none_present"]["n"] == 0
            assert report["contrastive"] is None
    assert result["models"]["kev"]["temperature"] == 1.2 and result["models"]["hev_seed1"]["temperature"] == 2.1
    kev_rows = world["rows"]["kev"]["decision"]
    assert result["models"]["kev"]["decision"]["all"]["raw"]["overall"]["accuracy"] == pytest.approx(clean_accuracy(kev_rows))

    paired = result["paired_kev_minus_hev"]
    for seed in ("hev_seed0", "hev_seed1"):
        for split in ("decision", "transfer"):
            raw, calibrated = paired[seed][split]["raw"], paired[seed][split]["calibrated"]
            assert raw["n"] == 2 * GROUPS and raw["direction"] == "kev minus Hev"
            assert raw["accuracy"]["ci95"][0] <= raw["accuracy"]["estimate"] <= raw["accuracy"]["ci95"][1]
            assert raw["accuracy"]["ci90"][0] >= raw["accuracy"]["ci95"][0]
            assert raw["accuracy"]["ci90"][1] <= raw["accuracy"]["ci95"][1]
            assert calibrated["temperatures"] == {"kev": 1.2, "hev": result["models"][seed]["temperature"]}
            assert {"nll", "brier"} <= set(calibrated)
    seed0_kev = clean_accuracy(partition(kev_rows, "decision", "primary"))
    seed0_hev = clean_accuracy(partition(world["rows"]["hev_seed0"]["decision"], "decision", "primary"))
    assert paired["hev_seed0"]["decision"]["raw"]["accuracy"]["estimate"] == pytest.approx(seed0_kev - seed0_hev)
    summary = result["two_seed_summary"]["decision"]["raw"]["accuracy"]
    assert summary["mean_point_delta"] == pytest.approx(np.mean([
        paired[seed]["decision"]["raw"]["accuracy"]["estimate"] for seed in ("hev_seed0", "hev_seed1")]))
    assert summary["pooled_interval"] is None

    assert result["verdict_rule"] == released.VERDICT_RULE and result["margins"] == {"decision": 0.02, "transfer": 0.03}
    for seed in ("hev_seed0", "hev_seed1"):
        for split in ("decision", "transfer"):
            entry = result["verdicts"][seed][split]
            assert entry == verdict(paired[seed][split]["raw"]["accuracy"]["ci95"],
                                    paired[seed][split]["raw"]["accuracy"]["ci90"], released.MARGINS[split])
    assert result["verdicts"]["hev_seed0"]["decision"]["verdict"] == "kev_better"

    order = result["order_study"]
    assert order["hev_seed0"]["mechanism_check"]["passed"] and order["kev"]["mechanism_check"] is None
    assert order["kev"]["decision"]["permutation"]["flip_count"] == 0
    assert order["kev"]["decision"]["permutation"]["denominator"] == 12
    assert order["hev_seed1"]["transfer"]["packed_vs_separate"]["max_abs_probability_difference"] is None

    context = result["kev_seeds_point_only"]
    assert context["seeds"]["seed1"]["label"] == released.KEV_SEED_LABEL
    assert context["seeds"]["seed1"]["sha256"] == digest(world["seed1"])
    assert context["seeds"]["seed2"]["decision_clean_accuracy"] == 0.82
    hub_decision = clean_accuracy(kev_rows)
    assert context["seeds"]["hub"]["decision_clean_accuracy"] == pytest.approx(hub_decision)
    assert context["seed_spread"]["decision_clean_accuracy"]["max"] == pytest.approx(max(hub_decision, 0.82))
    assert context["seed_spread"]["decision_clean_accuracy"]["min"] == pytest.approx(min(hub_decision, 0.80))
    assert context["seed_spread"]["transfer_clean_accuracy"]["min"] == pytest.approx(
        min(clean_accuracy(world["rows"]["kev"]["transfer"]), 0.58))

    training = result["training_data"]
    assert training["hev"]["records"] == 3432 and training["kev"]["records"] == 10896
    assert training["kev"]["recorded_training_args"]["p_none_pair"] == 0.25
    assert training["hev"]["recorded_training_args"][0]["accum"] == 8
    assert "docs/KEV.md" in training["kev"]["source"]
    assert result["provenance"]["kev"]["hub_commit"] == KEV_HUB_COMMIT
    assert result["provenance"]["hev_seed0"]["training_suite_sha256"] == HEV_TRAINING_SUITE_SHA256


def test_hev_mechanism_check_fails_on_flips(world):
    world["dirs"]["hev_seed1"] = write_eval_dir(world["tmp"], "hev1-flips", "hev", world["rows"]["hev_seed1"], flips=1 / 12)
    result = run_aggregate(world)
    check = result["order_study"]["hev_seed1"]["mechanism_check"]
    assert not check["passed"] and not check["checks"]["decision_zero_flips"]
    assert result["order_study"]["hev_seed1"]["decision"]["permutation"]["flip_count"] == 1
    assert result["order_study"]["hev_seed0"]["mechanism_check"]["passed"]


def test_cli_writes_artifacts_and_refuses_existing_out(world):
    out = world["tmp"] / "aggregate"
    result = main(cli_args(world, out))
    assert result["status"] == "success"
    written = json.loads((out / "result.json").read_text())
    assert written["protocol"] == "D13" and written["bootstrap"]["samples"] == 40
    assert written["artifact_hashes"]["summary"]["sha256"] == digest(out / "summary.md")
    summary = (out / "summary.md").read_text()
    assert "| kev |" in summary and "kev_better" in summary and "no pooled interval" in summary
    assert not (out / "result.json.tmp").exists() and not (out / "failure.json").exists()
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
    assert written["status"] == "failed" and "sanity gate" in written["failure_reason"]
    assert not written["sanity_gate"]["passed"] and written["verdicts"] is None
    assert "paired_kev_minus_hev" in written and (out / "summary.md").exists()
    assert "FAILED" in (out / "summary.md").read_text()


def test_cli_records_failure_json_on_error(world):
    world["dirs"]["hev_seed1"] = write_eval_dir(world["tmp"], "hev1-nostatus", "hev", world["rows"]["hev_seed1"],
                                                mutate=lambda result: result.update(status="running"))
    out = world["tmp"] / "broken"
    with pytest.raises(ValueError, match="expected 'success'"):
        main(cli_args(world, out))
    failure = json.loads((out / "failure.json").read_text())
    assert failure["error_type"] == "ValueError" and not (out / "result.json").exists()
