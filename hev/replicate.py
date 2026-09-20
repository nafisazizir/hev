"""Validate and aggregate the predeclared two-seed M3 PointerHead replication."""
import argparse
import json
from pathlib import Path

import numpy as np

from .compare import M2_RECIPE, load_hev_run
from .evaluate import paired_bootstrap, write_atomic
from .suite import digest, write_json

TOLERANCE = 1e-4
CRITICAL_SOURCES = ("hev/data.py", "hev/model.py", "hev/suite.py", "hev/train.py")
EVALUATE_HASHES = {
    "seed0": "1867e1638059fa16dcc671000cdf5ded61fca8e03b227431b1ae9b8439f5aad8",
    "seed1": "355a5375b33076d1d6856e0562a8d94afc9b12bac95f7b095ddafeaa07d43f4c",
}


def mechanism_checks(result):
    checks = {}
    for name in ("decision", "transfer"):
        report = result[name]
        coverage = report["coverage"]
        packed = report["packed_vs_separate"]["max_abs_probability_difference"]
        checks[f"{name}_coverage"] = (
            coverage["requested_records"] == coverage["evaluated_records"]
            and coverage["requested_questions"] == coverage["evaluated_questions"]
            and coverage["rejected_records"] == 0
            and coverage["truncated_records"] == 0
        )
        checks[f"{name}_zero_flips"] = report["permutation"]["argmax_flip_rate"] == 0
        checks[f"{name}_p90_spread"] = report["permutation"]["p90_correct_probability_spread"] <= TOLERANCE
        checks[f"{name}_packed"] = packed is None or packed <= TOLERANCE
    return checks


def headline(result):
    return {
        "decision_accuracy": result["decision"]["raw"]["overall"]["accuracy"],
        "transfer_accuracy": result["transfer"]["raw"]["overall"]["accuracy"],
        "decision_calibrated_ece": result["decision"]["calibrated"]["overall"]["ece"],
        "decision_calibrated_brier": result["decision"]["calibrated"]["overall"]["brier"],
    }


def seed_summary(run, seed):
    result = run["result"]
    checks = mechanism_checks(result)
    return {
        "seed": seed,
        "path": str(run["path"]),
        "result_sha256": digest(run["result_path"]),
        "temperature": result["temperature_fit"]["temperature"],
        "headline": headline(result),
        "mechanism_checks": checks,
        "mechanism_passed": all(checks.values()),
    }


def validate_evaluation_protocol(result, require_config=False):
    for name in ("decision", "transfer"):
        report = result[name]
        if report["permutation"]["orders"] != 6:
            raise ValueError(f"{name} did not use six permutation orders")
        if report["bootstrap_raw"]["samples"] != 10000 or report["bootstrap_calibrated"]["samples"] != 10000:
            raise ValueError(f"{name} did not use 10,000 bootstrap draws")
    if require_config:
        expected = {"seed": 1, "permutations": 6, "bootstrap_samples": 10000, "level_zero_ablation": False}
        if result.get("evaluation_config") != expected:
            raise ValueError("seed-1 evaluation does not match the predeclared M3 protocol")


def validate_compatibility(seed0, seed1):
    left, right = seed0["result"], seed1["result"]
    fields = ("base", "base_revision", "training_suite_sha256", "transfer_suite_sha256")
    if any(left[field] != right[field] for field in fields):
        raise ValueError("replication run provenance differs")
    left_sources = left["training_config"]["provenance"]["source_hashes"]
    right_sources = right["training_config"]["provenance"]["source_hashes"]
    changed = [name for name in CRITICAL_SOURCES if left_sources.get(name) != right_sources.get(name)]
    if changed:
        raise ValueError(f"replication-critical source changed between seeds: {changed}")
    evaluate_hashes = {"seed0": left_sources.get("hev/evaluate.py"), "seed1": right_sources.get("hev/evaluate.py")}
    if evaluate_hashes != EVALUATE_HASHES:
        raise ValueError("evaluate.py drift is not the audited M3 metadata-only change")
    validate_evaluation_protocol(left)
    validate_evaluation_protocol(right, require_config=True)


def aggregate(seed0_path, seed1_path, samples=10000, seed=20260920):
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    seed0 = load_hev_run(seed0_path, "pointer", expected_seed=0)
    seed1 = load_hev_run(seed1_path, "pointer", expected_seed=1)
    validate_compatibility(seed0, seed1)
    summaries = {"seed0": seed_summary(seed0, 0), "seed1": seed_summary(seed1, 1)}
    values = {name: [summary["headline"][name] for summary in summaries.values()]
              for name in summaries["seed0"]["headline"]}
    across_seed = {
        name: {"mean": float(np.mean(observed)), "range": [float(min(observed)), float(max(observed))]}
        for name, observed in values.items()
    }
    paired = {
        split: paired_bootstrap(seed1["rows"][split], seed0["rows"][split], 1.0, samples, seed)
        for split in ("decision", "transfer")
    }
    return {
        "schema_version": 1,
        "status": "success",
        "test_evaluated": False,
        "protocol": {
            "training_recipe": {**M2_RECIPE, "head": "pointer"},
            "replication_seed": 1,
            "evaluation_seed": 1,
            "permutations": 6,
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
            "mechanism_tolerance": TOLERANCE,
            "paired_direction": "seed1 minus seed0; descriptive only",
        },
        "provenance": {
            "base": seed0["result"]["base"],
            "base_revision": seed0["result"]["base_revision"],
            "training_suite_sha256": seed0["result"]["training_suite_sha256"],
            "transfer_suite_sha256": seed0["result"]["transfer_suite_sha256"],
            "critical_source_hashes": {
                name: seed0["result"]["training_config"]["provenance"]["source_hashes"][name]
                for name in CRITICAL_SOURCES
            },
            "evaluate_source_exception": {
                "hashes": EVALUATE_HASHES,
                "audit": "The only change adds evaluation_config fields to result.json after all predictions and metrics.",
            },
        },
        "seeds": summaries,
        "across_seed": across_seed,
        "paired_seed1_minus_seed0_raw": paired,
        "replication_passed": summaries["seed1"]["mechanism_passed"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0", required=True)
    parser.add_argument("--seed1", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing replication aggregate: {output}")
    output.mkdir(parents=True)
    try:
        result = aggregate(args.seed0, args.seed1, args.bootstrap_samples, args.seed)
        write_atomic(output / "result.json", result)
        print(json.dumps({"seeds": result["seeds"], "across_seed": result["across_seed"],
                          "replication_passed": result["replication_passed"]}, indent=2), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
