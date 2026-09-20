"""Aggregate the predeclared M4b controlled comparison (DECISIONS.md D15): released kev-0.6b versus three Hev
PointerHead seeds retrained on kev's own decision-v4 partition under kev's published recipe.

Inputs are four `hev.evaluate` output directories on decision-v4 / transfer-v4 development (the kev checkpoint
scored through the vendored kev path, three Hev seeds scored on their own training suite), the checkpoint's Hub
`result.json` for the predictor sanity gate, and kev's seed-1/2 result files as hash-verified point-only
context. Everything is paired on identical rows. The populations, margins and verdict rule are D13's, reused
verbatim through `hev.released`; what changes is the Hev validation (same suite, kev-v4 recipe, seeds 0-2) and
the three-seed summary, which is a mean with the full range and never a selected seed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .evaluate import clean_rows, metric_report
from .released import (DECISION_SUITE_SHA256, KEV_HUB_COMMIT, MARGINS, MECHANISM_TOLERANCE, POPULATIONS,
                       PRIMARY_SOURCES, SECONDARY_SOURCES, SPLITS, TRANSFER_SUITE_SHA256, VERDICT_RULE,
                       interval, kev_seed_context, load_evaluation, load_json, lookup, model_summary, paired_deltas,
                       partition, population_sources, pp, sanity_gate, validate_alignment, validate_kev_run,
                       validate_suites, verdict)
from .suite import digest, write_json
from .train import KEV_V4_RECIPE

PROTOCOL = "D15"
SCHEMA_VERSION = 1
HEV_SEEDS = ("hev_seed0", "hev_seed1", "hev_seed2")
MODELS = ("kev",) + HEV_SEEDS
EXPECTED_OPTIMIZER_STEPS = 2724
EVALUATION_CONFIG = {"seed": 1, "permutations": 6, "bootstrap_samples": 10000, "level_zero_ablation": False}
TRAINING_RECORDS = 10896


def validate_hev_run(run, expected_seed):
    """A D15 Hev run trained on decision-v4 itself under `--recipe kev-v4`, with the predeclared seed."""
    result = run["result"]
    label = run["label"]
    if result.get("cross_suite") is not None:
        raise ValueError(f"{label}: cross_suite is set; D15 runs train on the evaluated suite and must not use --allow-cross-suite")
    if result.get("head") != "pointer":
        raise ValueError(f"{label}: head is {result.get('head')!r}, expected 'pointer'")
    config = lookup(result, "training_config")
    if lookup(config, "suite_sha256") != DECISION_SUITE_SHA256:
        raise ValueError(f"{label}: training suite hash {config.get('suite_sha256')} is not decision-v4")
    recipe = lookup(config, "recipe")
    if not isinstance(recipe, dict) or recipe.get("name") != "kev-v4":
        raise ValueError(f"{label}: training_config.recipe is not the kev-v4 recipe")
    if recipe.get("matched") != KEV_V4_RECIPE["matched"]:
        raise ValueError(f"{label}: recorded matched knobs differ from kev's published trial: {recipe.get('matched')}")
    seed = lookup(config, "args.seed")
    if seed != expected_seed:
        raise ValueError(f"{label}: training seed is {seed!r}, expected {expected_seed}")
    if lookup(config, "train_partition.records") != TRAINING_RECORDS:
        raise ValueError(f"{label}: training partition has {config['train_partition'].get('records')} records, expected {TRAINING_RECORDS}")
    metrics = lookup(result, "training_metrics")
    if metrics.get("status") != "success":
        raise ValueError(f"{label}: training status is {metrics.get('status')!r}, expected 'success'")
    if metrics.get("optimizer_steps") != EXPECTED_OPTIMIZER_STEPS:
        raise ValueError(f"{label}: {metrics.get('optimizer_steps')} optimizer steps, expected {EXPECTED_OPTIMIZER_STEPS}")
    evaluation = lookup(result, "evaluation_config")
    if evaluation != EVALUATION_CONFIG:
        raise ValueError(f"{label}: evaluation_config {evaluation} is not D13's {EVALUATION_CONFIG}")
    return {
        "kind": "hev",
        "head": result.get("head"),
        "training_seed": seed,
        "training_suite_sha256": config["suite_sha256"],
        "recipe": recipe.get("name"),
        "accepted_deviations": recipe.get("accepted_deviations"),
        "train_partition": config.get("train_partition"),
        "optimizer_steps": metrics.get("optimizer_steps"),
        "none_pair_records": metrics.get("none_pair_records"),
        "wall_seconds": metrics.get("wall_seconds"),
        "device": lookup(config, "provenance.device"),
        "dtype": lookup(config, "provenance.dtype"),
        "git_commit": lookup(config, "provenance.git.commit"),
    }


def spread(values):
    return {"mean": float(np.mean(values)), "min": float(min(values)), "max": float(max(values)),
            "range": float(max(values) - min(values)), "values": [float(value) for value in values]}


def three_seed_summary(paired):
    """Mean and full range of the per-seed point deltas; every per-seed interval listed; no pooled interval."""
    summary = {}
    for split in SPLITS:
        summary[split] = {}
        for kind in ("raw", "calibrated"):
            seeds = [paired[seed][split][kind] for seed in HEV_SEEDS]
            names = sorted(name for name in seeds[0] if isinstance(seeds[0][name], dict) and "estimate" in seeds[0][name]
                           and all(name in seed for seed in seeds))
            summary[split][kind] = {
                name: {
                    **{f"{key}_point_delta": value for key, value in spread([seed[name]["estimate"] for seed in seeds]).items()
                       if key != "values"},
                    "seed_point_deltas": [seed[name]["estimate"] for seed in seeds],
                    "intervals": {label: {"ci95": seed[name]["ci95"], "ci90": seed[name]["ci90"]}
                                  for label, seed in zip(HEV_SEEDS, seeds)},
                    "pooled_interval": None,
                }
                for name in names
            }
    summary["note"] = "mean and range of the three per-seed point deltas; every per-seed interval listed; no pooled interval; no seed selected"
    return summary


def seed_spread(models, kev_context):
    """Three Hev seeds against kev's three seeds, clean accuracy, point-only and descriptive.

    Hev's numbers come from this evaluator on the 'all' population, which is the same clean-question set kev's
    own `clean.acc` counts (1,264 decision, 656 transfer). kev seeds 1 and 2 were never re-evaluated here, so this
    block carries no interval and no verdict; the paired comparison above is against the re-evaluated seed 0 only.
    """
    table = {}
    for split in SPLITS:
        hev_values = [models[name]["summary"][split]["all"]["raw"]["overall"]["accuracy"] for name in HEV_SEEDS]
        kev_values = [kev_context["seeds"][label][f"{split}_clean_accuracy"] for label in ("hub", "seed1", "seed2")]
        table[split] = {
            "hev": spread(hev_values),
            "kev": spread(kev_values),
            "mean_difference_kev_minus_hev": float(np.mean(kev_values) - np.mean(hev_values)),
            "hev_population": "all clean development questions, this evaluator",
            "kev_population": "kev's own evaluator, clean development questions, point-only",
        }
    table["note"] = "descriptive seed spread only; kev seeds 1 and 2 have no rows here, so no interval and no verdict"
    return table


def order_study(models):
    table = {}
    for name in MODELS:
        result = models[name]["run"]["result"]
        entry = {}
        for split in SPLITS:
            permutation = dict(lookup(result, f"{split}.permutation"))
            flip_rate, n = permutation.get("argmax_flip_rate"), permutation.get("n") or 0
            permutation["flip_count"] = None if flip_rate is None else int(round(flip_rate * n))
            permutation["denominator"] = n
            entry[split] = {"permutation": permutation, "packed_vs_separate": lookup(result, f"{split}.packed_vs_separate")}
        if name in HEV_SEEDS:
            checks = {}
            for split in SPLITS:
                permutation = entry[split]["permutation"]
                checks[f"{split}_zero_flips"] = permutation["flip_count"] == 0
                spread_value = permutation.get("p90_correct_probability_spread")
                checks[f"{split}_p90_spread"] = spread_value is not None and spread_value <= MECHANISM_TOLERANCE
                packed = (entry[split]["packed_vs_separate"] or {}).get("max_abs_probability_difference")
                checks[f"{split}_packed_vs_separate"] = packed is None or packed <= MECHANISM_TOLERANCE
            entry["mechanism_check"] = {"checks": checks, "passed": all(checks.values()),
                                        "tolerance": MECHANISM_TOLERANCE,
                                        "note": "D15: zero flips, p90 spread and packed-versus-separate difference at most the tolerance; failure is a bug, not a result"}
        else:
            entry["mechanism_check"] = None
            entry["note"] = "reported as measured under the exhaustive six-order protocol"
        table[name] = entry
    return table


def training_disclosure(kev_run, hev_runs, hev_provenance):
    kev_args = lookup(kev_run["result"], "provenance.training_args", "provenance.config", "provenance")
    return {
        "shared": {
            "suite": "decision-v4 train",
            "records": TRAINING_RECORDS,
            "composition": "1,000 per source x 10 public sources + 896 policy pairs (448 legacy_policy + 448 compositional)",
            "minimal_pairs": True,
            "p_none_pair": 0.25,
            "recipe": KEV_V4_RECIPE["matched"],
            "optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        },
        "hev": {
            "precision": "fp32",
            "device": "mps",
            "batch": 1,
            "accum": 8,
            "encoding": "option-isolating mask, shared option start positions",
            "recorded_training_args": [run["result"]["training_config"]["args"] for run in hev_runs],
            "recorded_provenance": [{"device": p["device"], "dtype": p["dtype"], "git_commit": p["git_commit"],
                                     "wall_seconds": p["wall_seconds"], "none_pair_records": p["none_pair_records"]}
                                    for p in hev_provenance],
            "accepted_deviations": hev_provenance[0]["accepted_deviations"],
        },
        "kev": {
            "precision": "bf16",
            "device": "H100",
            "batch": 8,
            "accum": 1,
            "encoding": "plain packing, option_isolation off",
            "recorded_training_args": kev_args,
            "source": "docs/KEV.md and the Hub result.json provenance.config",
        },
        "warning": "Same training partition, recipe and augmentation; precision, hardware and micro-batching still "
                   "differ, so a remaining difference is not attributable to the encoding alone (D15).",
    }


def summary_markdown(result):
    lines = ["# M4b controlled comparison (D15)", "",
             f"Status: {result['status']}" + (f" ({result['failure_reason']})" if result.get("failure_reason") else ""),
             f"Sanity gate: {'passed' if result['sanity_gate']['passed'] else 'FAILED'} "
             f"(decision {pp(result['sanity_gate']['hev_measured']['decision'])} vs "
             f"{pp(result['sanity_gate']['kev_reported']['decision'])}, transfer "
             f"{pp(result['sanity_gate']['hev_measured']['transfer'])} vs "
             f"{pp(result['sanity_gate']['kev_reported']['transfer'])}, tolerance "
             f"{pp(result['sanity_gate']['tolerance'])} pp)", "",
             "## Headline (primary population, clean rows, pp)", "",
             "| model | decision acc | transfer acc | decision calibrated ECE | decision flips | transfer flips | mechanism |",
             "|---|---|---|---|---|---|---|"]
    for name in MODELS:
        model = result["models"][name]
        order = result["order_study"][name]
        decision, transfer = model["decision"]["primary"], model["transfer"]["primary"]
        mechanism = order["mechanism_check"]
        lines.append(
            f"| {name} | {pp(decision['raw']['overall']['accuracy'])} {interval(decision['bootstrap_raw']['accuracy']['ci95'])} "
            f"| {pp(transfer['raw']['overall']['accuracy'])} {interval(transfer['bootstrap_raw']['accuracy']['ci95'])} "
            f"| {pp(decision['calibrated']['overall']['ece'])} "
            f"| {order['decision']['permutation']['flip_count']}/{order['decision']['permutation']['denominator']} "
            f"| {order['transfer']['permutation']['flip_count']}/{order['transfer']['permutation']['denominator']} "
            f"| {'n/a' if mechanism is None else ('passed' if mechanism['passed'] else 'FAILED')} |")
    lines += ["", "## Paired kev minus Hev accuracy (primary, pp)", "",
              "| Hev seed | suite | delta | 95% | 90% | verdict |", "|---|---|---|---|---|---|"]
    for seed in HEV_SEEDS:
        for split in SPLITS:
            delta = result["paired_kev_minus_hev"][seed][split]["raw"]["accuracy"]
            verdicts = result.get("verdicts") or {}
            label = verdicts.get(seed, {}).get(split, {}).get("verdict", "not claimed") if verdicts else "not claimed"
            lines.append(f"| {seed} | {split} | {pp(delta['estimate'])} | {interval(delta['ci95'])} | "
                         f"{interval(delta['ci90'])} | {label} |")
    lines += ["", "## Three-seed summary (mean and range of point deltas, no pooled interval)", ""]
    for split in SPLITS:
        entry = result["three_seed_summary"][split]["raw"]["accuracy"]
        lines.append(f"- {split} accuracy: mean delta {pp(entry['mean_point_delta'])} pp, range "
                     f"{pp(entry['min_point_delta'])} to {pp(entry['max_point_delta'])}; per-seed 95% "
                     + " / ".join(interval(entry['intervals'][seed]['ci95']) for seed in HEV_SEEDS))
    lines += ["", "## Seed spread, all clean questions (Hev this evaluator; kev its own evaluator, point-only)", ""]
    for split in SPLITS:
        entry = result["seed_spread"][split]
        lines.append(f"- {split}: Hev {pp(entry['hev']['mean'])} ({pp(entry['hev']['min'])} to {pp(entry['hev']['max'])}), "
                     f"kev {pp(entry['kev']['mean'])} ({pp(entry['kev']['min'])} to {pp(entry['kev']['max'])})")
    lines += ["", f"Rule: {VERDICT_RULE}", ""]
    return "\n".join(lines)


def aggregate(kev_path, hev_paths, hub_result_path, kev_seed1_path, kev_seed2_path, samples=10000, seed=20260920,
              expected_hashes=None):
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if len(hev_paths) != len(HEV_SEEDS):
        raise ValueError(f"expected {len(HEV_SEEDS)} Hev evaluation directories")
    kev = load_evaluation(kev_path, "kev")
    hevs = [load_evaluation(path, label) for path, label in zip(hev_paths, HEV_SEEDS)]
    validate_suites([kev, *hevs])
    kev_provenance = validate_kev_run(kev)
    hev_provenance = [validate_hev_run(run, index) for index, run in enumerate(hevs)]
    validate_alignment(kev, hevs)
    kev_context = kev_seed_context(hub_result_path, kev_seed1_path, kev_seed2_path, expected_hashes)
    gate = sanity_gate(kev, load_json(hub_result_path))

    runs = {"kev": kev, **dict(zip(HEV_SEEDS, hevs))}
    models = {name: {"run": run, "summary": model_summary(run, samples, seed)} for name, run in runs.items()}
    kev_temperature = models["kev"]["summary"]["temperature"]
    paired = {}
    for name in HEV_SEEDS:
        hev_temperature = models[name]["summary"]["temperature"]
        paired[name] = {}
        for split in SPLITS:
            left = partition(kev["rows"][split], split, "primary")
            right = partition(runs[name]["rows"][split], split, "primary")
            paired[name][split] = {
                "raw": paired_deltas(left, right, 1.0, 1.0, samples, seed),
                "calibrated": paired_deltas(left, right, kev_temperature, hev_temperature, samples, seed),
            }
    order = order_study(models)
    mechanism_passed = all(order[name]["mechanism_check"]["passed"] for name in HEV_SEEDS)
    verdicts = None
    if gate["passed"]:
        verdicts = {
            name: {split: verdict(paired[name][split]["raw"]["accuracy"]["ci95"],
                                  paired[name][split]["raw"]["accuracy"]["ci90"], MARGINS[split])
                   for split in SPLITS}
            for name in HEV_SEEDS
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "success" if gate["passed"] else "failed",
        "test_evaluated": False,
        "inputs": {
            name: {"path": str(run["path"]), "result_sha256": run["result_sha256"],
                   "row_artifacts": run["result"]["artifact_hashes"]}
            for name, run in runs.items()
        } | {
            "kev_hub_result": {"path": str(hub_result_path), "sha256": kev_context["seeds"]["hub"]["sha256"]},
            "kev_seed1_result": {"path": str(kev_seed1_path), "sha256": kev_context["seeds"]["seed1"]["sha256"]},
            "kev_seed2_result": {"path": str(kev_seed2_path), "sha256": kev_context["seeds"]["seed2"]["sha256"]},
        },
        "suites": {"decision_sha256": DECISION_SUITE_SHA256, "transfer_sha256": TRANSFER_SUITE_SHA256,
                   "hev_training_suite_sha256": DECISION_SUITE_SHA256},
        "provenance": {"kev": kev_provenance, **dict(zip(HEV_SEEDS, hev_provenance))},
        "bootstrap": {"samples": samples, "seed": seed, "unit": "(source, group_id) clusters",
                      "stratification": "source", "intervals": "percentile 95% and 90% from the same draws"},
        "populations": {population: {split: sorted(population_sources(split, population)) for split in SPLITS}
                        for population in POPULATIONS},
        "population_note": "the secondary policy arms are now a trained population for Hev as well as kev (D15); "
                           "they remain descriptive and carry no verdict",
        "sanity_gate": gate,
        "mechanism_checks_passed": mechanism_passed,
        "models": {name: models[name]["summary"] for name in MODELS},
        "paired_kev_minus_hev": paired,
        "three_seed_summary": three_seed_summary(paired),
        "seed_spread": seed_spread(models, kev_context),
        "verdict_rule": VERDICT_RULE,
        "margins": MARGINS,
        "verdicts": verdicts,
        "order_study": order,
        "kev_seeds_point_only": kev_context,
        "training_data": training_disclosure(kev, hevs, hev_provenance),
        "caveats": [
            "Development splits only; the locked test split was not accessed.",
            "Same items, same training partition, same recipe and augmentation; precision, hardware and "
            "micro-batching still differ, so no difference is attributed to the encoding alone.",
            "Verdicts apply to the primary population only; secondary populations are descriptive and are now "
            "trained populations for both models.",
            "The three-seed summary is a mean with a range of point deltas; there is no pooled interval and no seed is selected.",
            "kev seeds 1 and 2 are point-only context from kev's own evaluator.",
        ],
    }
    if not gate["passed"]:
        result["failure_reason"] = "predictor sanity gate failed: re-evaluated kev accuracy differs from the Hub result.json by more than the tolerance"
        result["verdicts_note"] = "no comparison is claimed while the sanity gate fails (D13, reused by D15)"
    return result


def write_artifacts(output, result):
    output = Path(output)
    summary_path = output / "summary.md"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite result artifact: {summary_path}")
    summary_path.write_text(summary_markdown(result))
    result["artifact_hashes"] = {"summary": {"path": "summary.md", "sha256": digest(summary_path)}}
    from .evaluate import write_atomic
    write_atomic(output / "result.json", result)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kev", required=True)
    ap.add_argument("--hev-seed0", required=True)
    ap.add_argument("--hev-seed1", required=True)
    ap.add_argument("--hev-seed2", required=True)
    ap.add_argument("--kev-hub-result", required=True)
    ap.add_argument("--kev-seed1", required=True)
    ap.add_argument("--kev-seed2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260920)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing controlled comparison: {output}")
    output.mkdir(parents=True)
    try:
        result = aggregate(args.kev, [args.hev_seed0, args.hev_seed1, args.hev_seed2], args.kev_hub_result,
                           args.kev_seed1, args.kev_seed2, args.bootstrap_samples, args.seed)
        write_artifacts(output, result)
        print(json.dumps({"status": result["status"], "sanity_gate": result["sanity_gate"]["passed"],
                          "mechanism_checks_passed": result["mechanism_checks_passed"],
                          "verdicts": result["verdicts"]}, indent=2), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"error_type": type(error).__name__, "error": str(error)})
        raise
    if result["status"] != "success":
        print(result["failure_reason"], file=sys.stderr, flush=True)
        sys.exit(1)
    return result


if __name__ == "__main__":
    main()
