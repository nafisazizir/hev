"""Aggregate the predeclared M4 comparison (DECISIONS.md D13): released kev-0.6b versus Hev PointerHead seeds.

Inputs are three `hev.evaluate` output directories on decision-v4 / transfer-v4 development (one kev checkpoint
scored through the vendored kev path, two Hev seeds scored cross-suite), the checkpoint's own Hub `result.json`
for the predictor sanity gate, and kev's seed-1/2 result files as hash-verified point-only context. Everything
is paired on identical rows; the verdict rule is fixed in D13 and copied verbatim into the artifact.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .evaluate import (aligned_rows, bootstrap_draw_values, bootstrap_report, clean_rows, cluster_index,
                       contrastive_report, metric_report, none_diagnostics, report_values, write_atomic)
from .suite import digest, write_json

PROTOCOL = "D13"
SCHEMA_VERSION = 1
DECISION_SUITE_SHA256 = "1b33e566d114f9eafeff55b36c221fadb2a4ae358a1b9cc68006e82c7cfad8f1"
TRANSFER_SUITE_SHA256 = "31677c2256b406222e7d94ffdc0a02a70ce05746b9efe307876024c4e77291d1"
HEV_TRAINING_SUITE_SHA256 = "e0388e1d13284f5e0a2e6cb33a0fcd34f0099bfc31a6a852bd5cb619f505094d"
KEV_HUB_COMMIT = "83e05fabf7ef08e343bb4daf144e08777f99713d"
KEV_FILE_HASHES = {
    "hub": "44edfc8ec6cb7e4f9ba2aac8378c2b445cc7cab36a80d8415cb2c32548ccea24",
    "seed1": "85cbea696e3c1356f8e7e1d3694117effb634891e526eb3d71dc0adbde91467c",
    "seed2": "9c69fdd54470ffb22b33d86e8187077879c301930a660cb538a21551e33bd022",
}
SANITY_TOLERANCE = 0.01
MECHANISM_TOLERANCE = 1e-4
MARGINS = {"decision": 0.02, "transfer": 0.03}
PRIMARY_SOURCES = {
    "decision": ("banking77", "boolq", "agnews", "mnli", "sst5", "yelp", "trec", "dbpedia14", "amazon", "imdb"),
    "transfer": ("mmlu", "emotion", "tweet_offensive", "qnli", "paws", "sciq"),
}
SECONDARY_SOURCES = {
    "decision": ("legacy_policy", "compositional"),
    "transfer": ("legacy_holdout", "composition_holdout"),
}
POPULATIONS = ("primary", "secondary", "all")
SPLITS = ("decision", "transfer")
MODELS = ("kev", "hev_seed0", "hev_seed1")
VERDICT_RULE = (
    "Applied to the primary population only, per suite and per Hev seed, on the paired kev-minus-Hev clean "
    "accuracy delta. Margins: 2.0 points decision, 3.0 points transfer. 'equivalent' if the 90% percentile "
    "interval lies entirely inside [-margin, +margin] (two one-sided tests at 5%). 'kev_better' if the 95% "
    "interval lies entirely above zero; 'hev_better' if entirely below zero. If equivalence and a direction both "
    "hold, both flags are reported (direction plus within_margin=true). Anything else is 'inconclusive'. "
    "Secondary results carry no verdict."
)
KEV_SEED_LABEL = "point-only, kev's own evaluator, not re-evaluated here"


def load_json(path):
    return json.loads(Path(path).read_text())


def lookup(mapping, *paths):
    """Return the first value found among dotted key paths; fail naming every path tried."""
    for path in paths:
        value = mapping
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            return value
    raise ValueError(f"result.json is missing required key {paths[0]!r}" + (
        f" (also tried {', '.join(repr(p) for p in paths[1:])})" if len(paths) > 1 else ""))


def load_evaluation(path, label):
    path = Path(path)
    result_path = path / "result.json"
    if not result_path.exists():
        raise ValueError(f"{label}: {result_path} does not exist")
    result = load_json(result_path)
    if result.get("status") != "success":
        raise ValueError(f"{label}: result status is {result.get('status')!r}, expected 'success'")
    if result.get("test_evaluated") is not False:
        raise ValueError(f"{label}: test_evaluated must be false")
    artifacts = lookup(result, "artifact_hashes")
    rows = {}
    for name in ("calibration", "decision", "transfer"):
        artifact = artifacts.get(name)
        if not artifact:
            raise ValueError(f"{label}: artifact_hashes.{name} is missing")
        rows_path = (path / artifact["path"]).resolve()
        if not rows_path.is_relative_to(path.resolve()):
            raise ValueError(f"{label}: {name} row artifact escapes its run directory")
        if digest(rows_path) != artifact["sha256"]:
            raise ValueError(f"{label}: {name} row artifact hash mismatch")
        rows[name] = load_json(rows_path)
        if len(rows[name]) != artifact["rows"]:
            raise ValueError(f"{label}: {name} row artifact count mismatch")
    return {"label": label, "path": path, "result_path": result_path, "result_sha256": digest(result_path),
            "result": result, "rows": rows}


def suite_hashes(result):
    return {
        "decision": lookup(result, "evaluation_suite.decision_sha256", "evaluation_suite.sha256",
                           "evaluation_suite.suite_sha256", "training_suite_sha256"),
        "transfer": lookup(result, "evaluation_suite.transfer_sha256", "transfer_suite_sha256"),
    }


def validate_suites(runs):
    for run in runs:
        hashes = suite_hashes(run["result"])
        if hashes["decision"] != DECISION_SUITE_SHA256:
            raise ValueError(f"{run['label']}: decision suite hash {hashes['decision']} is not decision-v4")
        if hashes["transfer"] != TRANSFER_SUITE_SHA256:
            raise ValueError(f"{run['label']}: transfer suite hash {hashes['transfer']} is not transfer-v4")


def validate_kev_run(run):
    result = run["result"]
    kind = lookup(result, "provenance.kind", "provenance.checkpoint_kind")
    if kind != "kev":
        raise ValueError(f"{run['label']}: provenance.kind is {kind!r}, expected 'kev'")
    checkpoint = lookup(result, "provenance.checkpoint")
    reference = checkpoint.get("hub_commit") or checkpoint.get("resolved_path")
    if not reference:
        raise ValueError(f"{run['label']}: provenance.checkpoint records neither hub_commit nor resolved_path")
    if checkpoint.get("hub_commit") not in (None, KEV_HUB_COMMIT):
        raise ValueError(f"{run['label']}: hub_commit {checkpoint['hub_commit']} is not the D13 snapshot")
    isolation = lookup(result, "provenance.option_isolation")
    if isolation is not False:
        raise ValueError(f"{run['label']}: provenance.option_isolation is {isolation!r}, expected false")
    return {"kind": kind, "hub_commit": checkpoint.get("hub_commit"), "resolved_path": checkpoint.get("resolved_path"),
            "option_isolation": isolation, "file_hashes": checkpoint.get("file_hashes")}


def validate_hev_run(run):
    result = run["result"]
    allowed = lookup(result, "cross_suite.allowed_by_flag")
    if allowed is not True:
        raise ValueError(f"{run['label']}: cross_suite.allowed_by_flag is {allowed!r}, expected true")
    training = lookup(result, "cross_suite.training_suite_sha256", "training_config.suite_sha256")
    if training != HEV_TRAINING_SUITE_SHA256:
        raise ValueError(f"{run['label']}: training suite hash {training} is not decision-v2")
    if result.get("head") not in (None, "pointer"):
        raise ValueError(f"{run['label']}: head is {result.get('head')!r}, expected 'pointer'")
    return {"kind": "hev", "head": result.get("head"), "cross_suite": result.get("cross_suite"),
            "training_suite_sha256": training, "training_seed": result.get("training_config", {}).get("args", {}).get("seed")}


def validate_alignment(kev, hevs):
    for split in SPLITS:
        reference = clean_rows(kev["rows"][split])
        for run in hevs:
            try:
                aligned_rows(reference, clean_rows(run["rows"][split]))
            except ValueError as error:
                raise ValueError(f"{split} rows of {run['label']} do not align with {kev['label']}: {error}") from error


def sanity_gate(kev_run, hub_result, tolerance=SANITY_TOLERANCE):
    measured = {split: metric_report(clean_rows(kev_run["rows"][split]))["overall"] for split in SPLITS}
    reported = {
        "decision": lookup(hub_result, "clean"),
        "transfer": lookup(hub_result, "transfer_clean", "transfer.clean"),
    }
    checks = {}
    for split in SPLITS:
        difference = abs(measured[split]["accuracy"] - reported[split]["acc"])
        checks[split] = {
            "hev_measured": measured[split]["accuracy"],
            "hev_n": measured[split]["n"],
            "kev_reported": reported[split]["acc"],
            "kev_n": reported[split].get("n"),
            "abs_difference": float(difference),
            "passed": bool(difference <= tolerance),
        }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "hev_measured": {split: checks[split]["hev_measured"] for split in SPLITS},
        "kev_reported": {split: checks[split]["kev_reported"] for split in SPLITS},
        "tolerance": tolerance,
        "checks": checks,
        "n_definition": "Hev rows are one per question with variant == 'clean'; kev's reported n counts the same "
                        "clean questions (1264 decision, 656 transfer on v4 development) and is compared on acc only",
    }


def population_sources(split, population):
    if population == "primary":
        return set(PRIMARY_SOURCES[split])
    if population == "secondary":
        return set(SECONDARY_SOURCES[split])
    return set(PRIMARY_SOURCES[split]) | set(SECONDARY_SOURCES[split])


def partition(rows, split, population):
    known = population_sources(split, "all")
    unknown = sorted({row["source"] for row in rows} - known)
    if unknown:
        raise ValueError(f"{split} rows contain sources outside the declared populations: {unknown}")
    selected = population_sources(split, population)
    return [row for row in rows if row["source"] in selected]


def descriptive(name, function, rows):
    try:
        return function(rows)
    except ValueError as error:
        return {"unavailable": f"{name} not computable on this population: {error}"}


def population_report(rows, temperature, samples, seed):
    selected = clean_rows(rows)
    if not selected:
        return {"n_clean": 0, "n_rows": len(rows), "unavailable": "population has no clean rows"}
    return {
        "n_clean": len(selected),
        "n_rows": len(rows),
        "sources": sorted({row["source"] for row in selected}),
        "raw": metric_report(selected),
        "calibrated": metric_report(selected, temperature),
        "bootstrap_raw": bootstrap_report(selected, 1.0, samples, seed),
        "bootstrap_calibrated": bootstrap_report(selected, temperature, samples, seed),
        "none_diagnostics": descriptive("none diagnostics", none_diagnostics, rows),
        "contrastive": descriptive("contrastive report", contrastive_report, rows),
    }


def model_summary(run, samples, seed):
    result = run["result"]
    temperature = lookup(result, "temperature_fit.temperature")
    summary = {"path": str(run["path"]), "result_sha256": run["result_sha256"], "temperature": temperature,
               "temperature_fit": result["temperature_fit"]}
    for split in SPLITS:
        summary[split] = {
            population: population_report(partition(run["rows"][split], split, population), temperature, samples, seed)
            for population in POPULATIONS
        }
    return summary


def paired_deltas(left, right, left_temperature, right_temperature, samples, seed):
    """kev (left) minus Hev (right) on identical clean rows, with 95% and 90% intervals from one set of draws."""
    a, b = aligned_rows(clean_rows(left), clean_rows(right))
    units = cluster_index(list(a.values()))
    key_for = {id(row): key for key, row in a.items()}
    right_units = {source: [[b[key_for[id(row)]] for row in unit] for unit in source_units]
                   for source, source_units in units.items()}
    observed_left = report_values(list(a.values()), left_temperature)
    observed_right = report_values(list(b.values()), right_temperature)
    names = sorted(observed_left.keys() & observed_right.keys())
    left_draws, right_draws = bootstrap_draw_values([units, right_units], [left_temperature, right_temperature],
                                                    samples, seed)
    deltas = {}
    for name in names:
        draws = left_draws[name] - right_draws[name]
        deltas[name] = {
            "estimate": float(observed_left[name] - observed_right[name]),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "ci90": np.quantile(draws, [0.05, 0.95]).tolist(),
        }
    return deltas | {
        "n": len(a),
        "samples": samples,
        "seed": seed,
        "direction": "kev minus Hev",
        "temperatures": {"kev": left_temperature, "hev": right_temperature},
        "unit": "paired source-stratified (source, group_id) clusters",
    }


def verdict(ci95, ci90, margin):
    within_margin = bool(-margin <= ci90[0] and ci90[1] <= margin)
    if ci95[0] > 0:
        direction = "kev_better"
    elif ci95[1] < 0:
        direction = "hev_better"
    else:
        direction = None
    if direction and within_margin:
        label = f"{direction}_within_margin"
    elif direction:
        label = direction
    elif within_margin:
        label = "equivalent"
    else:
        label = "inconclusive"
    return {"verdict": label, "direction": direction, "within_margin": within_margin, "margin": margin,
            "ci95": list(ci95), "ci90": list(ci90)}


def two_seed_summary(paired):
    summary = {}
    for split in SPLITS:
        summary[split] = {}
        for kind in ("raw", "calibrated"):
            seeds = [paired[seed][split][kind] for seed in ("hev_seed0", "hev_seed1")]
            names = sorted(set(seeds[0]) & set(seeds[1]) & {
                name for name in seeds[0] if isinstance(seeds[0][name], dict) and "estimate" in seeds[0][name]})
            summary[split][kind] = {
                name: {
                    "mean_point_delta": float(np.mean([seed[name]["estimate"] for seed in seeds])),
                    "seed_point_deltas": [seed[name]["estimate"] for seed in seeds],
                    "intervals": {label: {"ci95": seed[name]["ci95"], "ci90": seed[name]["ci90"]}
                                  for label, seed in zip(("hev_seed0", "hev_seed1"), seeds)},
                    "pooled_interval": None,
                }
                for name in names
            }
    summary["note"] = "mean of the two per-seed point deltas; both per-seed intervals listed; no pooled interval"
    return summary


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
        if name.startswith("hev"):
            checks = {}
            for split in SPLITS:
                permutation = entry[split]["permutation"]
                checks[f"{split}_zero_flips"] = permutation["flip_count"] == 0
                spread = permutation.get("p90_correct_probability_spread")
                checks[f"{split}_p90_spread"] = spread is not None and spread <= MECHANISM_TOLERANCE
            entry["mechanism_check"] = {"checks": checks, "passed": all(checks.values()),
                                        "tolerance": MECHANISM_TOLERANCE}
        else:
            entry["mechanism_check"] = None
            entry["note"] = "reported as measured; first exhaustive six-order measurement of a released kev checkpoint"
        table[name] = entry
    return table


def kev_seed_context(hub_path, seed1_path, seed2_path, expected_hashes=None):
    expected = expected_hashes or KEV_FILE_HASHES
    entries = {}
    for label, path in (("hub", hub_path), ("seed1", seed1_path), ("seed2", seed2_path)):
        actual = digest(path)
        if actual != expected[label]:
            raise ValueError(f"kev {label} result hash mismatch: {actual} != {expected[label]}")
        result = load_json(path)
        if result.get("test_evaluated") is not False:
            raise ValueError(f"kev {label} result is not development-only")
        if lookup(result, "provenance.suite_sha256") != DECISION_SUITE_SHA256:
            raise ValueError(f"kev {label} result was not produced on decision-v4")
        if lookup(result, "transfer.suite_sha256") != TRANSFER_SUITE_SHA256:
            raise ValueError(f"kev {label} result was not produced on transfer-v4")
        entries[label] = {
            "path": str(path),
            "sha256": actual,
            "training_seed": lookup(result, "provenance.config.seed"),
            "decision_clean_accuracy": lookup(result, "clean.acc"),
            "decision_clean_n": lookup(result, "clean").get("n"),
            "transfer_clean_accuracy": lookup(result, "transfer_clean.acc", "transfer.clean.acc"),
            "transfer_clean_n": lookup(result, "transfer_clean", "transfer.clean").get("n"),
            "label": KEV_SEED_LABEL,
        }
    entries["hub"]["label"] = "the checkpoint re-evaluated here; its own evaluator's development numbers"
    spread = {}
    for metric in ("decision_clean_accuracy", "transfer_clean_accuracy"):
        values = [entries[label][metric] for label in ("hub", "seed1", "seed2")]
        spread[metric] = {"mean": float(np.mean(values)), "min": float(min(values)), "max": float(max(values)),
                          "range": float(max(values) - min(values))}
    return {"seeds": entries, "seed_spread": spread,
            "note": "kev seeds 1 and 2 are point-only context from kev's own evaluator; no rows, no intervals, "
                    "not re-evaluated here"}


def training_disclosure(kev_run, hev_runs):
    kev_args = lookup(kev_run["result"], "provenance.training_args", "provenance.config", "provenance")
    hev_args = [run["result"].get("training_config", {}).get("args") for run in hev_runs]
    hev_provenance = [run["result"].get("training_config", {}).get("provenance") for run in hev_runs]
    return {
        "hev": {
            "suite": "decision-v2 train",
            "records": 3432,
            "composition": "300 per source x 10 public sources + 432 contrastive",
            "minimal_pairs": False,
            "precision": "fp32",
            "device": "mps",
            "batch": 1,
            "accum": 8,
            "recorded_training_args": hev_args,
            "recorded_provenance": [{"device": p.get("device"), "dtype": p.get("dtype"), "gpu": p.get("gpu")}
                                    if isinstance(p, dict) else None for p in hev_provenance],
            "source": "docs/KEV.md (v2 recipe) and each Hev run's training_config",
        },
        "kev": {
            "suite": "decision-v4 train",
            "records": 10896,
            "composition": "1,000 per source x 10 public sources + 896 policy pairs",
            "minimal_pairs": True,
            "p_none_pair": 0.25,
            "precision": "bf16",
            "device": "H100",
            "batch": 8,
            "accum": 1,
            "recorded_training_args": kev_args,
            "source": "docs/KEV.md and the Hub result.json provenance.config; record counts hardcoded from kev PLAN.md",
        },
        "warning": "Different training data, precision and hardware; this is a same-items comparison, not an "
                   "architecture claim (D13).",
    }


def pp(value):
    return f"{100 * value:.2f}"


def interval(values):
    return f"({pp(values[0])}, {pp(values[1])})"


def summary_markdown(result):
    lines = ["# M4 released comparison (D13)", "",
             f"Status: {result['status']}" + (f" ({result['failure_reason']})" if result.get("failure_reason") else ""),
             f"Sanity gate: {'passed' if result['sanity_gate']['passed'] else 'FAILED'} "
             f"(decision {pp(result['sanity_gate']['hev_measured']['decision'])} vs "
             f"{pp(result['sanity_gate']['kev_reported']['decision'])}, transfer "
             f"{pp(result['sanity_gate']['hev_measured']['transfer'])} vs "
             f"{pp(result['sanity_gate']['kev_reported']['transfer'])}, tolerance "
             f"{pp(result['sanity_gate']['tolerance'])} pp)", "",
             "## Headline (primary population, clean rows, pp)", "",
             "| model | decision acc | transfer acc | decision calibrated ECE | decision flips | transfer flips |",
             "|---|---|---|---|---|---|"]
    for name in MODELS:
        model = result["models"][name]
        order = result["order_study"][name]
        decision, transfer = model["decision"]["primary"], model["transfer"]["primary"]
        lines.append(
            f"| {name} | {pp(decision['raw']['overall']['accuracy'])} {interval(decision['bootstrap_raw']['accuracy']['ci95'])} "
            f"| {pp(transfer['raw']['overall']['accuracy'])} {interval(transfer['bootstrap_raw']['accuracy']['ci95'])} "
            f"| {pp(decision['calibrated']['overall']['ece'])} "
            f"| {order['decision']['permutation']['flip_count']}/{order['decision']['permutation']['denominator']} "
            f"| {order['transfer']['permutation']['flip_count']}/{order['transfer']['permutation']['denominator']} |")
    lines += ["", "## Paired kev minus Hev accuracy (primary, pp)", "",
              "| Hev seed | suite | delta | 95% | 90% | verdict |", "|---|---|---|---|---|---|"]
    for seed in ("hev_seed0", "hev_seed1"):
        for split in SPLITS:
            delta = result["paired_kev_minus_hev"][seed][split]["raw"]["accuracy"]
            verdicts = result.get("verdicts") or {}
            label = verdicts.get(seed, {}).get(split, {}).get("verdict", "not claimed") if verdicts else "not claimed"
            lines.append(f"| {seed} | {split} | {pp(delta['estimate'])} | {interval(delta['ci95'])} | "
                         f"{interval(delta['ci90'])} | {label} |")
    lines += ["", "## Two-seed summary (mean of point deltas, no pooled interval)", ""]
    for split in SPLITS:
        entry = result["two_seed_summary"][split]["raw"]["accuracy"]
        lines.append(f"- {split} accuracy: mean delta {pp(entry['mean_point_delta'])} pp; per-seed 95% "
                     f"{interval(entry['intervals']['hev_seed0']['ci95'])} and {interval(entry['intervals']['hev_seed1']['ci95'])}")
    lines += ["", f"Rule: {VERDICT_RULE}", ""]
    return "\n".join(lines)


def aggregate(kev_path, hev_seed0_path, hev_seed1_path, hub_result_path, kev_seed1_path, kev_seed2_path,
              samples=10000, seed=20260920, expected_hashes=None):
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    kev = load_evaluation(kev_path, "kev")
    hev0 = load_evaluation(hev_seed0_path, "hev_seed0")
    hev1 = load_evaluation(hev_seed1_path, "hev_seed1")
    validate_suites([kev, hev0, hev1])
    kev_provenance = validate_kev_run(kev)
    hev_provenance = [validate_hev_run(run) for run in (hev0, hev1)]
    validate_alignment(kev, [hev0, hev1])
    kev_context = kev_seed_context(hub_result_path, kev_seed1_path, kev_seed2_path, expected_hashes)
    gate = sanity_gate(kev, load_json(hub_result_path))

    runs = {"kev": kev, "hev_seed0": hev0, "hev_seed1": hev1}
    models = {name: {"run": run, "summary": model_summary(run, samples, seed)} for name, run in runs.items()}
    kev_temperature = models["kev"]["summary"]["temperature"]
    paired = {}
    for name in ("hev_seed0", "hev_seed1"):
        hev_temperature = models[name]["summary"]["temperature"]
        paired[name] = {}
        for split in SPLITS:
            left = partition(kev["rows"][split], split, "primary")
            right = partition(runs[name]["rows"][split], split, "primary")
            paired[name][split] = {
                "raw": paired_deltas(left, right, 1.0, 1.0, samples, seed),
                "calibrated": paired_deltas(left, right, kev_temperature, hev_temperature, samples, seed),
            }
    verdicts = None
    if gate["passed"]:
        verdicts = {
            name: {split: verdict(paired[name][split]["raw"]["accuracy"]["ci95"],
                                  paired[name][split]["raw"]["accuracy"]["ci90"], MARGINS[split])
                   for split in SPLITS}
            for name in ("hev_seed0", "hev_seed1")
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
                   "hev_training_suite_sha256": HEV_TRAINING_SUITE_SHA256},
        "provenance": {"kev": kev_provenance, "hev_seed0": hev_provenance[0], "hev_seed1": hev_provenance[1]},
        "bootstrap": {"samples": samples, "seed": seed, "unit": "(source, group_id) clusters",
                      "stratification": "source", "intervals": "percentile 95% and 90% from the same draws"},
        "populations": {population: {split: sorted(population_sources(split, population)) for split in SPLITS}
                        for population in POPULATIONS},
        "sanity_gate": gate,
        "models": {name: models[name]["summary"] for name in MODELS},
        "paired_kev_minus_hev": paired,
        "two_seed_summary": two_seed_summary(paired),
        "verdict_rule": VERDICT_RULE,
        "margins": MARGINS,
        "verdicts": verdicts,
        "order_study": order_study(models),
        "kev_seeds_point_only": kev_context,
        "training_data": training_disclosure(kev, [hev0, hev1]),
        "caveats": [
            "Development splits only; the locked test split was not accessed.",
            "Same items, one evaluator, different training data and hardware; not an architecture claim.",
            "Verdicts apply to the primary population only; secondary populations are descriptive.",
            "The two-seed summary is a mean of point deltas; there is no pooled interval.",
            "kev seeds 1 and 2 are point-only context from kev's own evaluator.",
        ],
    }
    if not gate["passed"]:
        result["failure_reason"] = "predictor sanity gate failed: re-evaluated kev accuracy differs from the Hub result.json by more than the tolerance"
        result["verdicts_note"] = "no comparison is claimed while the sanity gate fails (D13)"
    return result


def write_artifacts(output, result):
    output = Path(output)
    summary_path = output / "summary.md"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite result artifact: {summary_path}")
    summary_path.write_text(summary_markdown(result))
    result["artifact_hashes"] = {"summary": {"path": "summary.md", "sha256": digest(summary_path)}}
    write_atomic(output / "result.json", result)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kev", required=True)
    ap.add_argument("--hev-seed0", required=True)
    ap.add_argument("--hev-seed1", required=True)
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
        raise FileExistsError(f"refusing to overwrite existing released comparison: {output}")
    output.mkdir(parents=True)
    try:
        result = aggregate(args.kev, args.hev_seed0, args.hev_seed1, args.kev_hub_result, args.kev_seed1,
                           args.kev_seed2, args.bootstrap_samples, args.seed)
        write_artifacts(output, result)
        print(json.dumps({"status": result["status"], "sanity_gate": result["sanity_gate"]["passed"],
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
