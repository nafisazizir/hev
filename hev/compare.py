"""Compare completed Hev M2 runs with each other and hash-verified kev point baselines."""
import argparse
import json
from pathlib import Path

from .evaluate import paired_bootstrap, write_atomic
from .suite import digest, write_json

KEV_HASHES = {
    "seed0": "cca4b992ea64f93bad193cd515bd422074f8101a6e26ffe14dd93b21d1c057dc",
    "seed1": "a3388221ef4c11496bdcf8faf551e7e8a182e3b31acc457be7482bd85cace8e9",
}
H1_DECISION_MIN = 0.765833
H1_TRANSFER_MIN = 0.569643
H1_TOLERANCE = 1e-4
H3_ECE_MAX = 0.0464021
H3_BRIER_MAX = 0.2794607
D6_BANKING77_MIN = 0.7125
M2_RECIPE = {
    "epochs": 2,
    "seed": 0,
    "lr": 2e-4,
    "lora": 16,
    "batch": 1,
    "accum": 8,
    "ord_w": 0.0,
    "p_none": 0.1,
    "p_none_distract": 0.12,
    "p_distract": 0.15,
    "device": "mps",
    "require_loss_decrease": True,
}


def load_json(path):
    return json.loads(Path(path).read_text())


def load_hev_run(path, expected_head, expected_seed=0):
    path = Path(path)
    result_path = path / "result.json"
    result = load_json(result_path)
    if result.get("status") != "success" or result.get("test_evaluated") is not False:
        raise ValueError(f"{path} is not a successful development-only result")
    if result.get("head") != expected_head:
        raise ValueError(f"expected {expected_head} result, got {result.get('head')}")
    args = result["training_config"]["args"]
    recipe = {**M2_RECIPE, "seed": expected_seed}
    if args.get("head") != expected_head or any(args.get(key) != value for key, value in recipe.items()):
        raise ValueError("run does not match the predeclared recipe")
    training = result["training_metrics"]
    if (training.get("status"), training.get("records_seen"), training.get("requested_records"), training.get("optimizer_steps")) != (
        "success", 6864, 6864, 858
    ):
        raise ValueError("run does not have complete M2 training coverage")
    rows = {}
    for name in ("decision", "transfer"):
        artifact = result["artifact_hashes"].get(name)
        if not artifact:
            raise ValueError(f"missing {name} row artifact")
        artifact_path = (path / artifact["path"]).resolve()
        if not artifact_path.is_relative_to(path.resolve()):
            raise ValueError(f"{name} row artifact escapes its run directory")
        if digest(artifact_path) != artifact["sha256"]:
            raise ValueError(f"{name} row artifact hash mismatch")
        rows[name] = load_json(artifact_path)
        if len(rows[name]) != artifact["rows"]:
            raise ValueError(f"{name} row artifact count mismatch")
    return {"path": path, "result_path": result_path, "result": result, "rows": rows}


def kev_summary(path, label):
    path = Path(path)
    actual_hash = digest(path)
    if actual_hash != KEV_HASHES[label]:
        raise ValueError(f"kev {label} result hash mismatch")
    result = load_json(path)
    if result.get("test_evaluated") is not False:
        raise ValueError(f"kev {label} result is not development-only")
    for report in (result, result["transfer"]):
        coverage = report["coverage"]
        complete = coverage["requested_records"] == coverage["evaluated_records"]
        complete = complete and coverage["requested_questions"] == coverage["evaluated_questions"]
        if not complete or coverage["rejected_records"] or coverage["truncated_records"]:
            raise ValueError(f"kev {label} result has incomplete coverage")
    return {
        "path": str(path),
        "sha256": actual_hash,
        "decision_suite_sha256": result["provenance"]["suite_sha256"],
        "transfer_suite_sha256": result["transfer"]["suite_sha256"],
        "decision": {
            "clean": result["clean"],
            "calibrated_clean": result["calibrated_clean"],
            "tasks": result["tasks"],
            "permutation": result["permutation"],
        },
        "transfer": {
            "clean": result["transfer"]["clean"],
            "tasks": result["transfer"]["tasks"],
            "permutation": result["transfer"]["permutation"],
            "paired_flip": result["transfer"]["paired_flip"],
        },
    }


def packed_ok(report):
    value = report["packed_vs_separate"]["max_abs_probability_difference"]
    return value is None or value <= H1_TOLERANCE


def h1_decision(pointer, kev_seed0):
    decision, transfer = pointer["decision"], pointer["transfer"]
    checks = {
        "decision_zero_flips": decision["permutation"]["argmax_flip_rate"] == 0,
        "transfer_zero_flips": transfer["permutation"]["argmax_flip_rate"] == 0,
        "decision_p90_spread": decision["permutation"]["p90_correct_probability_spread"] <= H1_TOLERANCE,
        "transfer_p90_spread": transfer["permutation"]["p90_correct_probability_spread"] <= H1_TOLERANCE,
        "decision_packed": packed_ok(decision),
        "transfer_packed": packed_ok(transfer),
        "decision_accuracy": decision["raw"]["overall"]["accuracy"] >= H1_DECISION_MIN,
        "transfer_accuracy": transfer["raw"]["overall"]["accuracy"] >= H1_TRANSFER_MIN,
    }
    return {
        "outcome": "supported" if all(checks.values()) else "contradicted",
        "checks": checks,
        "thresholds": {
            "mechanism_tolerance": H1_TOLERANCE,
            "decision_accuracy_min": H1_DECISION_MIN,
            "transfer_accuracy_min": H1_TRANSFER_MIN,
        },
        "kev_seed0_points": {
            "decision_accuracy": kev_seed0["decision"]["clean"]["acc"],
            "transfer_accuracy": kev_seed0["transfer"]["clean"]["acc"],
        },
    }


def h2_decision(pointer_accuracy, set_accuracy, interval, kev_accuracy):
    loss = kev_accuracy - pointer_accuracy
    recovery = set_accuracy - pointer_accuracy
    ratio = recovery / loss if loss > 0 else None
    if loss <= 0.01:
        outcome = "inconclusive"
        reason = "pointer loss is not material"
    elif recovery <= 0 or ratio < 0.5:
        outcome = "contradicted"
        reason = "SetHead recovers less than half of the pointer deficit"
    elif interval[0] > 0:
        outcome = "supported"
        reason = "majority recovery with paired interval above zero"
    else:
        outcome = "inconclusive"
        reason = "point recovery passes but paired interval crosses zero"
    return {
        "outcome": outcome,
        "reason": reason,
        "kev_minus_pointer": loss,
        "set_minus_pointer": recovery,
        "recovery_fraction": ratio,
        "paired_accuracy_ci95": interval,
        "material_loss_threshold": 0.01,
        "recovery_threshold": 0.5,
    }


def h3_decision(result):
    calibrated = result["decision"]["calibrated"]["overall"]
    checks = {
        "ece": calibrated["ece"] <= H3_ECE_MAX,
        "brier": calibrated["brier"] <= H3_BRIER_MAX,
    }
    return {
        "outcome": "supported" if all(checks.values()) else "contradicted",
        "checks": checks,
        "observed": {"ece": calibrated["ece"], "brier": calibrated["brier"]},
        "thresholds": {"ece_max": H3_ECE_MAX, "brier_max": H3_BRIER_MAX},
        "transfer": "descriptive only; checked-in kev v2 result has no calibrated transfer baseline",
    }


def ablation_decision(pointer):
    ablation = pointer.get("level_zero_ablation")
    if not ablation:
        raise ValueError("pointer result is missing the predeclared level-zero ablation")
    outcome = "supported" if ablation["useful"] else "inconclusive"
    return {
        "outcome": outcome,
        "used": ablation["used"],
        "useful": ablation["useful"],
        "max_abs_probability_change": ablation["max_abs_probability_change"],
        "nll_delta": ablation["paired_learned_minus_zeroed"]["nll"],
    }


def d6_decision(pointer, set_result):
    try:
        pointer_accuracy = pointer["decision"]["raw"]["by_task"]["banking77"]["accuracy"]
        set_accuracy = set_result["decision"]["raw"]["by_task"]["banking77"]["accuracy"]
    except KeyError as error:
        raise ValueError("D6 decision requires banking77 task metrics") from error
    triggered = pointer_accuracy < D6_BANKING77_MIN and set_accuracy < D6_BANKING77_MIN
    return {
        "triggered": triggered,
        "threshold": D6_BANKING77_MIN,
        "pointer_accuracy": pointer_accuracy,
        "set_accuracy": set_accuracy,
        "action": "recommend D6 top-M fallback" if triggered else "do not build D6 fallback from M2",
    }


def validate_compatibility(pointer, set_run, kev):
    left, right = pointer["result"], set_run["result"]
    fields = ("base", "base_revision", "training_suite_sha256", "transfer_suite_sha256", "evaluation_source_hashes")
    if any(left[field] != right[field] for field in fields):
        raise ValueError("pointer and set run provenance differs")
    if left["training_suite_sha256"] != kev["decision_suite_sha256"]:
        raise ValueError("Hev and kev decision suite hashes differ")
    if left["transfer_suite_sha256"] != kev["transfer_suite_sha256"]:
        raise ValueError("Hev and kev transfer suite hashes differ")


def compare(pointer_path, set_path, kev_seed0_path, kev_seed1_path, samples=10000, seed=20260919):
    pointer = load_hev_run(pointer_path, "pointer")
    set_run = load_hev_run(set_path, "set")
    kev0 = kev_summary(kev_seed0_path, "seed0")
    kev1 = kev_summary(kev_seed1_path, "seed1")
    validate_compatibility(pointer, set_run, kev0)
    validate_compatibility(pointer, set_run, kev1)
    paired = {
        split: paired_bootstrap(set_run["rows"][split], pointer["rows"][split], 1.0, samples, seed)
        for split in ("decision", "transfer")
    }
    pointer_result, set_result = pointer["result"], set_run["result"]
    hypotheses = {
        "h1": h1_decision(pointer_result, kev0),
        "h2": h2_decision(
            pointer_result["decision"]["raw"]["overall"]["accuracy"],
            set_result["decision"]["raw"]["overall"]["accuracy"],
            paired["decision"]["accuracy"]["ci95"],
            kev0["decision"]["clean"]["acc"],
        ),
        "h3": {"pointer": h3_decision(pointer_result), "set": h3_decision(set_result)},
        "level_embedding": ablation_decision(pointer_result),
        "d6": d6_decision(pointer_result, set_result),
    }
    return {
        "schema_version": 1,
        "status": "success",
        "test_evaluated": False,
        "pointer": {"path": str(pointer["path"]), "result_sha256": digest(pointer["result_path"]),
                    "decision": pointer_result["decision"], "transfer": pointer_result["transfer"]},
        "set": {"path": str(set_run["path"]), "result_sha256": digest(set_run["result_path"]),
                "decision": set_result["decision"], "transfer": set_result["transfer"]},
        "kev": {"seed0": kev0, "seed1": kev1,
                "uncertainty": "point baselines only; checked-in kev v2 artifacts contain no per-example rows"},
        "paired_set_minus_pointer": paired,
        "hypotheses": hypotheses,
    }


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointer", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--kev-seed0", required=True)
    ap.add_argument("--kev-seed1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260919)
    return ap


def main():
    args = parser().parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap samples must be positive")
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing comparison: {output}")
    output.mkdir(parents=True)
    try:
        result = compare(args.pointer, args.set, args.kev_seed0, args.kev_seed1, args.bootstrap_samples, args.seed)
        write_atomic(output / "result.json", result)
        print(json.dumps({"hypotheses": result["hypotheses"]}, indent=2), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
