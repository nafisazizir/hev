"""Build the comprehensive M3 Hev, kev, and hosted Jev development comparison ledger."""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .compare import kev_summary, load_hev_run
from .evaluate import (aligned_rows, bootstrap_draw_values, clean_rows, cluster_index, fit_temperature,
                       metric_report, report_values)
from .suite import digest, write_json


def load_json(path):
    return json.loads(Path(path).read_text())


def canonical_kev_metrics(values):
    names = {
        "acc": "accuracy",
        "mean_conf": "mean_confidence",
        "score_mae": "score_mae_levels",
    }
    return {names.get(key, key): value for key, value in values.items()}


def canonical_kev_report(overall, tasks=None):
    return {
        "overall": canonical_kev_metrics(overall),
        "by_task": {name: canonical_kev_metrics(values) for name, values in sorted((tasks or {}).items())},
        "by_source": None,
        "by_type": None,
    }


def metric_aggregate(values):
    if not values:
        raise ValueError("cannot aggregate empty metrics")
    if len({value.get("n") for value in values}) != 1:
        raise ValueError("seed metric populations differ")
    keys = sorted(set.intersection(*(set(value) for value in values)) - {"n"})
    result = {"n": values[0].get("n")}
    for key in keys:
        observed = [value[key] for value in values]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in observed):
            result[key] = {"mean": float(np.mean(observed)), "range": [float(min(observed)), float(max(observed))]}
    return result


def report_aggregate(reports):
    sections = {"overall": metric_aggregate([report["overall"] for report in reports])}
    for section in ("by_task", "by_source", "by_type"):
        groups = [report.get(section) for report in reports]
        if any(group is None for group in groups):
            sections[section] = None
            continue
        if len({tuple(group) for group in groups}) != 1:
            raise ValueError(f"seed {section} populations differ")
        sections[section] = {
            name: metric_aggregate([group[name] for group in groups])
            for name in groups[0]
        }
    return sections


def hev_summary(run):
    result = run["result"]
    return {
        "path": str(run["path"]),
        "result_sha256": digest(run["result_path"]),
        "temperature": result["temperature_fit"]["temperature"],
        "decision": {
            "raw": result["decision"]["raw"],
            "calibrated": result["decision"]["calibrated"],
            "variants": result["decision"]["variants"],
            "none_diagnostics": result["decision"]["none_of_the_above"],
            "contrastive": result["decision"]["contrastive"],
            "permutation": result["decision"]["permutation"],
            "packed_vs_separate": result["decision"]["packed_vs_separate"],
            "coverage": result["decision"]["coverage"],
            "latency_seconds": result["decision"]["latency_seconds"],
        },
        "transfer": {
            "raw": result["transfer"]["raw"],
            "calibrated": result["transfer"]["calibrated"],
            "variants": result["transfer"]["variants"],
            "none_diagnostics": result["transfer"]["none_of_the_above"],
            "contrastive": result["transfer"]["contrastive"],
            "permutation": result["transfer"]["permutation"],
            "packed_vs_separate": result["transfer"]["packed_vs_separate"],
            "coverage": result["transfer"]["coverage"],
            "latency_seconds": result["transfer"]["latency_seconds"],
        },
    }


def full_kev_summary(path, label):
    verified = kev_summary(path, label)
    result = load_json(path)
    return {
        "path": str(path),
        "result_sha256": verified["sha256"],
        "temperature": result["temperature"],
        "decision": {
            "raw": canonical_kev_report(result["clean"], result["tasks"]),
            "calibrated": canonical_kev_report(result["calibrated_clean"]),
            "variants": {name: canonical_kev_metrics(values) for name, values in result["variants"].items()},
            "contrastive": result["paired_flip"],
            "permutation": result["permutation"],
            "coverage": result["coverage"],
            "latency_ms": result["latency_ms"],
            "metric_policy": result["metric_policy"],
            "mechanism_checks": result["mechanism_checks"],
        },
        "transfer": {
            "raw": canonical_kev_report(result["transfer"]["clean"], result["transfer"]["tasks"]),
            "calibrated": None,
            "variants": {name: canonical_kev_metrics(values) for name, values in result["transfer"]["variants"].items()},
            "contrastive": result["transfer"]["paired_flip"],
            "permutation": result["transfer"]["permutation"],
            "coverage": result["transfer"]["coverage"],
            "latency_ms": None,
            "metric_policy": result["metric_policy"],
        },
        "provenance": result["provenance"],
    }


def load_jev_run(path, expected_split):
    path = Path(path)
    result_path = path / "result.json"
    result = load_json(result_path)
    if result.get("status") != "success" or result.get("test_evaluated") is not False:
        raise ValueError(f"{path} is not a successful development-only Jev result")
    if result.get("split") != expected_split:
        raise ValueError(f"expected Jev {expected_split}, got {result.get('split')}")
    coverage = result["coverage"]
    if (
        coverage["requested_records"] != coverage["evaluated_records"]
        or coverage["requested_questions"] != coverage["evaluated_questions"]
        or coverage["rejected_records"]
        or coverage["truncated_records"]
    ):
        raise ValueError(f"{path} has incomplete Jev coverage")
    artifact = result["artifact_hashes"]["rows"]
    rows_path = (path / artifact["path"]).resolve()
    if not rows_path.is_relative_to(path.resolve()) or digest(rows_path) != artifact["sha256"]:
        raise ValueError(f"{path} Jev rows hash mismatch")
    rows = load_json(rows_path)
    if len(rows) != artifact["rows"]:
        raise ValueError(f"{path} Jev row count mismatch")
    versions = result["provider"]["model_versions"]
    if len(versions) != 1 or re.fullmatch(r"jev-\d+(?:\.\d+)+", versions[0]) is None:
        raise ValueError(f"{path} lacks one versioned Jev model")
    return {"path": path, "result_path": result_path, "result": result, "rows": rows, "version": versions[0]}


def paired_bootstrap_temperatures(left, right, left_temperature, right_temperature, samples, seed):
    left, right = clean_rows(left), clean_rows(right)
    a, b = aligned_rows(left, right)
    units = cluster_index(list(a.values()))
    key_for = {id(row): key for key, row in a.items()}
    right_units = {
        source: [[b[key_for[id(row)]] for row in unit] for unit in source_units]
        for source, source_units in units.items()
    }
    observed_left = report_values(list(a.values()), left_temperature)
    observed_right = report_values(list(b.values()), right_temperature)
    names = sorted(observed_left.keys() & observed_right.keys())
    left_draws, right_draws = bootstrap_draw_values(
        [units, right_units], [left_temperature, right_temperature], samples, seed
    )
    return {
        name: {
            "estimate": float(observed_left[name] - observed_right[name]),
            "ci95": np.quantile(left_draws[name] - right_draws[name], [0.025, 0.975]).tolist(),
        }
        for name in names
    } | {
        "samples": samples,
        "seed": seed,
        "direction": "left minus right",
        "temperatures": {"left": left_temperature, "right": right_temperature},
        "unit": "paired source-stratified (source, group_id) clusters",
    }


def point_delta(left, right):
    return {
        key: float(left[key] - right[key])
        for key in sorted(left.keys() & right.keys())
        if key != "n" and isinstance(left[key], (int, float)) and isinstance(right[key], (int, float))
    }


def nll_floor_sensitivity(rows):
    selected = clean_rows(rows)
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["task"]].append(row)
    result = {}
    for floor in (1e-3, 1e-6, 1e-9):
        losses = [
            -math.log(max(float(row["probabilities"][row["label"]]), floor))
            for row in selected
        ]
        task_losses = {
            task: np.mean([
                -math.log(max(float(row["probabilities"][row["label"]]), floor))
                for row in task_rows
            ])
            for task, task_rows in grouped.items()
        }
        result[str(floor)] = {
            "overall_nll": float(np.mean(losses)),
            "macro_task_nll": float(np.mean(list(task_losses.values()))),
        }
    return result


def validate_compatibility(hev_runs, kev_runs, jev_runs):
    decision_hash = hev_runs[0]["result"]["training_suite_sha256"]
    transfer_hash = hev_runs[0]["result"]["transfer_suite_sha256"]
    if any(run["result"]["training_suite_sha256"] != decision_hash for run in hev_runs):
        raise ValueError("Hev decision suite hashes differ")
    if any(run["result"]["transfer_suite_sha256"] != transfer_hash for run in hev_runs):
        raise ValueError("Hev transfer suite hashes differ")
    if any(run["decision_suite_sha256"] != decision_hash for run in kev_runs):
        raise ValueError("kev and Hev decision suite hashes differ")
    if any(run["transfer_suite_sha256"] != transfer_hash for run in kev_runs):
        raise ValueError("kev and Hev transfer suite hashes differ")
    calibration, decision, transfer = jev_runs
    if calibration["result"]["suite_manifest_sha256"] != decision_hash:
        raise ValueError("Jev calibration suite hash differs")
    if decision["result"]["suite_manifest_sha256"] != decision_hash:
        raise ValueError("Jev decision suite hash differs")
    if transfer["result"]["suite_manifest_sha256"] != transfer_hash:
        raise ValueError("Jev transfer suite hash differs")
    if len({run["version"] for run in jev_runs}) != 1:
        raise ValueError("Jev model version changed across the comparison")


def compare(hev_seed0_path, hev_seed1_path, hev_set_path, kev_seed0_path, kev_seed1_path,
            jev_calibration_path, jev_decision_path, jev_transfer_path, samples=10000, seed=20260920):
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    hev0 = load_hev_run(hev_seed0_path, "pointer", expected_seed=0)
    hev1 = load_hev_run(hev_seed1_path, "pointer", expected_seed=1)
    hev_set = load_hev_run(hev_set_path, "set", expected_seed=0)
    kev_verified = [kev_summary(kev_seed0_path, "seed0"), kev_summary(kev_seed1_path, "seed1")]
    kev = [full_kev_summary(kev_seed0_path, "seed0"), full_kev_summary(kev_seed1_path, "seed1")]
    jev_calibration = load_jev_run(jev_calibration_path, "calibration")
    jev_decision = load_jev_run(jev_decision_path, "development")
    jev_transfer = load_jev_run(jev_transfer_path, "development")
    validate_compatibility([hev0, hev1, hev_set], kev_verified, [jev_calibration, jev_decision, jev_transfer])

    hev = [hev_summary(hev0), hev_summary(hev1)]
    set_summary = hev_summary(hev_set)
    jev_fit = fit_temperature(jev_calibration["rows"])
    jev_temperature = jev_fit["temperature"]
    jev = {
        "model_version": jev_decision["version"],
        "temperature_fit": jev_fit,
        "calibration": {
            "path": str(jev_calibration["path"]),
            "result_sha256": digest(jev_calibration["result_path"]),
            "raw": jev_calibration["result"]["raw"],
        },
        "decision": {
            "path": str(jev_decision["path"]),
            "result_sha256": digest(jev_decision["result_path"]),
            "raw": metric_report(clean_rows(jev_decision["rows"])),
            "calibrated": metric_report(clean_rows(jev_decision["rows"]), jev_temperature),
            "variants": jev_decision["result"]["variants"],
            "none_diagnostics": jev_decision["result"]["none_diagnostics"],
            "contrastive": jev_decision["result"]["contrastive"],
            "permutation": jev_decision["result"]["permutation"],
            "coverage": jev_decision["result"]["coverage"],
            "latency_ms": jev_decision["result"]["latency_ms"],
            "metric_policy": jev_decision["result"]["metric_policy"],
            "nll_floor_sensitivity": nll_floor_sensitivity(jev_decision["rows"]),
        },
        "transfer": {
            "path": str(jev_transfer["path"]),
            "result_sha256": digest(jev_transfer["result_path"]),
            "raw": metric_report(clean_rows(jev_transfer["rows"])),
            "calibrated": metric_report(clean_rows(jev_transfer["rows"]), jev_temperature),
            "variants": jev_transfer["result"]["variants"],
            "none_diagnostics": jev_transfer["result"]["none_diagnostics"],
            "contrastive": jev_transfer["result"]["contrastive"],
            "permutation": jev_transfer["result"]["permutation"],
            "coverage": jev_transfer["result"]["coverage"],
            "latency_ms": jev_transfer["result"]["latency_ms"],
            "metric_policy": jev_transfer["result"]["metric_policy"],
            "nll_floor_sensitivity": nll_floor_sensitivity(jev_transfer["rows"]),
        },
        "provider": {
            "calibration": jev_calibration["result"]["provider"],
            "decision": jev_decision["result"]["provider"],
            "transfer": jev_transfer["result"]["provider"],
            "total_estimated_usd": sum(run["result"]["provider"]["estimated_usd"] for run in (jev_calibration, jev_decision, jev_transfer)),
        },
    }

    families = {
        "hev_pointer": {
            "decision_raw": report_aggregate([run["decision"]["raw"] for run in hev]),
            "decision_calibrated": report_aggregate([run["decision"]["calibrated"] for run in hev]),
            "transfer_raw": report_aggregate([run["transfer"]["raw"] for run in hev]),
            "transfer_calibrated": report_aggregate([run["transfer"]["calibrated"] for run in hev]),
        },
        "kev": {
            "decision_raw": report_aggregate([run["decision"]["raw"] for run in kev]),
            "decision_calibrated": report_aggregate([run["decision"]["calibrated"] for run in kev]),
            "transfer_raw": report_aggregate([run["transfer"]["raw"] for run in kev]),
            "transfer_calibrated": None,
        },
    }

    paired = {}
    for label, run, summary in (("hev_pointer_seed0", hev0, hev[0]), ("hev_pointer_seed1", hev1, hev[1]),
                                ("hev_set_seed0", hev_set, set_summary)):
        paired[label] = {}
        for split, jev_run in (("decision", jev_decision), ("transfer", jev_transfer)):
            paired[label][split] = {
                "raw_jev_minus_hev": paired_bootstrap_temperatures(
                    jev_run["rows"], run["rows"][split], 1.0, 1.0, samples, seed
                ),
                "calibrated_jev_minus_hev": paired_bootstrap_temperatures(
                    jev_run["rows"], run["rows"][split], jev_temperature, summary["temperature"], samples, seed
                ),
            }

    point_only = {
        "reason": "checked-in kev v2 artifacts contain no per-example rows; paired intervals against kev are impossible",
        "seed_matched_hev_pointer_minus_kev": {},
        "jev_minus_kev": {},
    }
    for index, name in enumerate(("seed0", "seed1")):
        point_only["seed_matched_hev_pointer_minus_kev"][name] = {
            "decision_raw": point_delta(hev[index]["decision"]["raw"]["overall"], kev[index]["decision"]["raw"]["overall"]),
            "decision_calibrated": point_delta(hev[index]["decision"]["calibrated"]["overall"], kev[index]["decision"]["calibrated"]["overall"]),
            "transfer_raw": point_delta(hev[index]["transfer"]["raw"]["overall"], kev[index]["transfer"]["raw"]["overall"]),
        }
        point_only["jev_minus_kev"][name] = {
            "decision_raw": point_delta(jev["decision"]["raw"]["overall"], kev[index]["decision"]["raw"]["overall"]),
            "decision_calibrated": point_delta(jev["decision"]["calibrated"]["overall"], kev[index]["decision"]["calibrated"]["overall"]),
            "transfer_raw": point_delta(jev["transfer"]["raw"]["overall"], kev[index]["transfer"]["raw"]["overall"]),
        }

    return {
        "schema_version": 1,
        "status": "success",
        "test_evaluated": False,
        "protocol": {
            "suites": ["decision-v2 development", "transfer-v2 development"],
            "jev_calibration": "decision-v2 calibration only; fitted temperature applied unchanged to both development suites",
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
            "primary_direction": "Jev minus Hev; higher accuracy is better, lower loss/calibration metrics are better",
            "seed_policy": "report both predeclared Hev Pointer and kev seeds plus means/ranges; do not select the better seed",
        },
        "models": {
            "hev_pointer": {"seed0": hev[0], "seed1": hev[1]},
            "hev_set_ablation": {"seed0": set_summary, "replicated": False},
            "kev": {"seed0": kev[0], "seed1": kev[1]},
            "jev": jev,
        },
        "family_aggregates": families,
        "paired_jev_minus_hev": paired,
        "point_only_comparisons": point_only,
        "order_protocol": {
            "hev": "dedicated six-order study over every eligible Choice question; exact-by-construction claim evaluated at 1e-4",
            "kev_and_jev": "one fixed clean-versus-permuted suite variant on 72 decision and 36 transfer Choice questions",
            "warning": "flip rates and probability movement are descriptive but not numerically identical protocols",
        },
        "availability": {
            "by_task": "available for every model",
            "by_type": "available for Hev and Jev; unavailable for kev because its checked-in v2 artifacts have no rows or type summary",
            "kev_transfer_calibration": "unavailable in the checked-in v2 artifacts",
            "latency": "reported but not hardware-comparable: Hev local MPS, kev H100, Jev hosted network",
        },
        "caveats": [
            "Development and calibration splits only; the locked test split was not accessed.",
            "Jev is a hosted alias snapshot resolved to one observed version, not a reproducible checkpoint.",
            "Jev rounds many probabilities to zero; NLL uses a 1e-9 floor and must be read with the included floor sensitivity.",
            "Each model family has different training data and architecture; this is a same-evaluation comparison, not a controlled training comparison.",
            "Hev and kev have two training seeds; Jev has one hosted snapshot, so no Jev seed-variance estimate exists.",
            "Calibration temperatures are fit per model on decision-v2 calibration; transfer calibration is descriptive only.",
            "SetHead is a seed-0 Hev ablation, not the selected or replicated M3 model.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hev-seed0", required=True)
    parser.add_argument("--hev-seed1", required=True)
    parser.add_argument("--hev-set", required=True)
    parser.add_argument("--kev-seed0", required=True)
    parser.add_argument("--kev-seed1", required=True)
    parser.add_argument("--jev-calibration", required=True)
    parser.add_argument("--jev-decision", required=True)
    parser.add_argument("--jev-transfer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing three-way comparison: {output}")
    output.mkdir(parents=True)
    try:
        result = compare(
            args.hev_seed0, args.hev_seed1, args.hev_set, args.kev_seed0, args.kev_seed1,
            args.jev_calibration, args.jev_decision, args.jev_transfer,
            args.bootstrap_samples, args.seed,
        )
        write_json(output / "result.json", result)
        print(json.dumps({
            "hev_pointer": result["family_aggregates"]["hev_pointer"],
            "kev": result["family_aggregates"]["kev"],
            "jev": {
                "model_version": result["models"]["jev"]["model_version"],
                "decision_raw": result["models"]["jev"]["decision"]["raw"]["overall"],
                "transfer_raw": result["models"]["jev"]["transfer"]["raw"]["overall"],
            },
        }, indent=2), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
