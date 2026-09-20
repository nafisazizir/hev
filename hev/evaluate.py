"""Evaluate Hev checkpoints on frozen calibration, development and transfer suites.

Default: a local Hev run is scored on the suite it was trained on and results land in the run directory.
Two explicit relaxations exist, each recorded in result.json (never silent):
  --allow-cross-suite  evaluate a run on a different suite and/or under changed source; requires --out.
  --checkpoint-kind kev  score an external kev checkpoint through hev.kev.KevPredictor; requires --out.
"""
import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .data import NONE_OPTIONS, TRAINABLE, materialize
from .model import DecisionModel, encode, load_tokenizer
from .suite import base_revision, digest, load_split, manifest, object_digest, source_hashes, write_json

EPSILON = 1e-9
BOOTSTRAP_SEED = 20260919


def ece(confidence, correct, bins=10):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (confidence >= low) & (confidence < high) if high < 1 else (confidence >= low) & (confidence <= high)
        if selected.any():
            value += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(value)


def validate_probabilities(values):
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 1 or not len(probabilities):
        raise ValueError("probabilities must be a nonempty vector")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not math.isclose(float(probabilities.sum()), 1.0, rel_tol=0, abs_tol=1e-5):
        raise ValueError("probabilities must sum to one")
    return probabilities


def scaled_probabilities(values, temperature):
    probabilities = validate_probabilities(values)
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    logits = np.log(np.maximum(probabilities, EPSILON)) / temperature
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def metrics(rows, temperature=1.0):
    if not rows:
        raise ValueError("cannot score an empty population")
    nll, accuracy, confidence, brier, score_mae, score_rps = [], [], [], [], [], []
    for row in rows:
        probabilities = scaled_probabilities(row["probabilities"], temperature)
        label = int(row["label"])
        if not 0 <= label < len(probabilities):
            raise ValueError("label is outside the probability vector")
        target = np.eye(len(probabilities))[label]
        nll.append(-math.log(max(float(probabilities[label]), EPSILON)))
        accuracy.append(int(probabilities.argmax() == label))
        confidence.append(float(probabilities.max()))
        brier.append(float(((probabilities - target) ** 2).sum()))
        if row["type"] == "score":
            score_mae.append(abs(float(probabilities @ np.arange(len(probabilities))) - label))
            score_rps.append(float(((probabilities.cumsum()[:-1] - target.cumsum()[:-1]) ** 2).mean()))
    result = {
        "n": len(rows),
        "accuracy": float(np.mean(accuracy)),
        "nll": float(np.mean(nll)),
        "brier": float(np.mean(brier)),
        "ece": ece(confidence, accuracy),
        "mean_confidence": float(np.mean(confidence)),
    }
    if score_mae:
        result["score_mae_levels"] = float(np.mean(score_mae))
        result["ranked_probability_score"] = float(np.mean(score_rps))
    return result


def row_group(row, key):
    if key == "task":
        return row.get("task", row["source"])
    return row[key]


def grouped_metrics(rows, key, temperature=1.0):
    groups = defaultdict(list)
    for row in rows:
        groups[row_group(row, key)].append(row)
    return {name: metrics(group, temperature) for name, group in sorted(groups.items())}


def macro_metrics(grouped):
    names = ("accuracy", "nll", "brier", "ece", "mean_confidence")
    return {"n_groups": len(grouped), **{
        name: float(np.mean([group[name] for group in grouped.values()])) for name in names
    }}


def macro_source_metrics(by_source):
    result = macro_metrics(by_source)
    result["n_sources"] = result.pop("n_groups")
    return result


def macro_task_metrics(by_task):
    result = macro_metrics(by_task)
    result["n_tasks"] = result.pop("n_groups")
    return result


def metric_report(rows, temperature=1.0):
    by_source = grouped_metrics(rows, "source", temperature)
    by_task = grouped_metrics(rows, "task", temperature)
    return {
        "overall": metrics(rows, temperature),
        "by_source": by_source,
        "by_task": by_task,
        "by_type": grouped_metrics(rows, "type", temperature),
        "macro_source": macro_source_metrics(by_source),
        "macro_task": macro_task_metrics(by_task),
    }


def clean_rows(rows):
    return [row for row in rows if row.get("variant", "clean") == "clean"]


def fit_temperature(rows):
    selected = clean_rows(rows)
    if not selected:
        raise ValueError("cannot fit temperature on an empty calibration split")
    candidates = np.exp(np.linspace(np.log(0.25), np.log(4.0), 81))
    losses = [metric_report(selected, float(candidate))["macro_task"]["nll"] for candidate in candidates]
    index = int(np.argmin(losses))
    raw_nll = metric_report(selected)["macro_task"]["nll"]
    return {
        "temperature": float(candidates[index]),
        "n": len(selected),
        "objective": "clean unweighted macro-task NLL",
        "raw_macro_task_nll": raw_nll,
        "fitted_macro_task_nll": float(losses[index]),
        "raw_macro_nll": raw_nll,
        "fitted_macro_nll": float(losses[index]),
        "candidate_count": len(candidates),
        "range": [0.25, 4.0],
    }


def question_keys(question):
    if question["type"] == "choice":
        return list(question["criteria"])
    if question["type"] == "noul":
        return ["false", "true"]
    return [str(index) for index in range(len(question["criteria"]))]


def semantic_label(question, keys):
    if question["type"] == "choice":
        return keys.index(question["label"])
    return int(question["label"])


def rows_for_record(request, probabilities):
    if len(probabilities) != len(request["questions"]):
        raise ValueError("prediction question count does not match request")
    meta = request["_meta"]
    rows = []
    for (question_id, question), values in zip(request["questions"].items(), probabilities):
        keys = question_keys(question)
        vector = validate_probabilities(values)
        if len(keys) != len(vector):
            raise ValueError("probability count does not match option count")
        label = semantic_label(question, keys)
        prediction = int(vector.argmax())
        variant = meta.get("variant", "clean")
        rows.append({
            "id": meta["id"],
            "group": meta.get("group_id", meta["id"]),
            "question": question_id,
            "source": meta["source"],
            "task": question.get("src", meta["source"]),
            "type": question["type"],
            "variant": variant,
            "pair_id": meta.get("pair_id"),
            "sibling": meta.get("sibling"),
            "parent": meta.get("parent_id") or (meta["id"] if variant == "clean" else meta.get("group_id", meta["id"])),
            "keys": keys,
            "label": label,
            "probabilities": vector.tolist(),
            "predicted_key": keys[prediction],
            "correct": bool(prediction == label),
            "confidence": float(vector.max()),
        })
    return rows


def default_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def output_paths(directory, include_transfer=True, include_ablation=False):
    directory = Path(directory)
    paths = {
        "result": directory / "result.json",
        "calibration": directory / "calibration_rows.json",
        "decision": directory / "decision_rows.json",
    }
    if include_transfer:
        paths["transfer"] = directory / "transfer_rows.json"
    if include_ablation:
        paths["level_zero"] = directory / "level_zero_rows.json"
    return paths


def prepare_outputs(directory, include_transfer=True, include_ablation=False):
    directory = Path(directory)
    paths = output_paths(directory, include_transfer, include_ablation)
    protected = list(paths.values()) + [directory / "eval.json", directory / "eval_rows.json", directory / "evaluation_failure.json"]
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite evaluation artifacts: {existing}")
    return paths


def resolve_output_dir(run, out):
    """Without --out, outputs go into the run directory as before. With --out, the directory must be new."""
    if out is None:
        return Path(run)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to write into an existing output directory: {out}")
    out.mkdir(parents=True)
    return out


class LocalPredictor:
    def __init__(self, run, device):
        run = Path(run)
        training_metrics = json.loads((run / "training_metrics.json").read_text())
        if training_metrics.get("status") != "success":
            raise ValueError("checkpoint training did not complete successfully")
        metadata = torch.load(run / "readout.pt", map_location="cpu", weights_only=True)
        tokenizer = load_tokenizer(metadata["base"], revision=metadata["base_revision"])
        model = DecisionModel(
            metadata["base"], tokenizer, device, lora=None, revision=metadata["base_revision"],
            head=metadata["readout"]["head_kind"],
        )
        from peft import PeftModel
        model.lm = PeftModel.from_pretrained(model.lm, run).to(device)
        model.load_readout_state_dict(metadata["readout"])
        model.eval()
        self.run = run
        self.device = device
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, request):
        packed = encode(self.tokenizer, materialize(request), strict=True)
        sync(self.device)
        started = time.perf_counter()
        probabilities = [values.numpy() for values in self.model.probs(packed)]
        sync(self.device)
        return probabilities, time.perf_counter() - started


def predict_rows(requests, predictor):
    rows, latencies = [], []
    for request in requests:
        probabilities, latency = predictor(request)
        rows.extend(rows_for_record(request, probabilities))
        latencies.append(latency)
    expected_questions = sum(len(request["questions"]) for request in requests)
    if len(rows) != expected_questions:
        raise ValueError("evaluation coverage is incomplete")
    return rows, latencies


def one_question(request, question_id, question):
    return {"state": request["state"], "questions": {question_id: question}, "_meta": request["_meta"]}


def permutation_study(requests, predictor, seed=1, permutations=6):
    if permutations < 2:
        raise ValueError("permutations must be at least two")
    rng = random.Random(seed)
    flips, spreads = [], []
    for request in requests:
        for question_id, question in request["questions"].items():
            if question["type"] != "choice" or len(question["criteria"]) < 3:
                continue
            predictions, correct_probabilities = [], []
            for _ in range(permutations):
                keys = list(question["criteria"])
                rng.shuffle(keys)
                permuted = {**question, "criteria": {key: question["criteria"][key] for key in keys}}
                values, _ = predictor(one_question(request, question_id, permuted))
                vector = validate_probabilities(values[0])
                predictions.append(keys[int(vector.argmax())])
                correct_probabilities.append(float(vector[keys.index(question["label"])]))
            flips.append(int(len(set(predictions)) > 1))
            spreads.append(max(correct_probabilities) - min(correct_probabilities))
    if not spreads:
        return {"n": 0, "orders": permutations, "argmax_flip_rate": None, "mean_correct_probability_spread": None,
                "p90_correct_probability_spread": None, "max_correct_probability_spread": None}
    return {
        "n": len(spreads),
        "orders": permutations,
        "argmax_flip_rate": float(np.mean(flips)),
        "mean_correct_probability_spread": float(np.mean(spreads)),
        "p90_correct_probability_spread": float(np.percentile(spreads, 90)),
        "max_correct_probability_spread": float(np.max(spreads)),
    }


def packed_vs_separate(requests, predictor):
    differences, records = [], 0
    for request in requests:
        if len(request["questions"]) < 2:
            continue
        packed, _ = predictor(request)
        records += 1
        for index, (question_id, question) in enumerate(request["questions"].items()):
            separate, _ = predictor(one_question(request, question_id, question))
            left = validate_probabilities(packed[index])
            right = validate_probabilities(separate[0])
            differences.append(float(np.max(np.abs(left - right))))
    if not differences:
        return {"n_records": 0, "n_questions": 0, "mean_abs_probability_difference": None,
                "p90_abs_probability_difference": None, "max_abs_probability_difference": None}
    return {
        "n_records": records,
        "n_questions": len(differences),
        "mean_abs_probability_difference": float(np.mean(differences)),
        "p90_abs_probability_difference": float(np.percentile(differences, 90)),
        "max_abs_probability_difference": float(np.max(differences)),
    }


def contrastive_report(rows):
    pairs = {}
    for row in clean_rows(rows):
        if not row.get("pair_id"):
            continue
        key = (row["pair_id"], row["question"])
        pair = pairs.setdefault(key, {})
        sibling = row.get("sibling")
        if sibling in pair:
            raise ValueError("duplicate contrastive sibling")
        pair[sibling] = row
    if not pairs:
        return None
    if any(set(pair) != {"a", "b"} for pair in pairs.values()):
        raise ValueError("incomplete contrastive pair")
    truth = lambda row: row["keys"][row["label"]]
    predicted = lambda row: row["predicted_key"]
    relevant = [pair for pair in pairs.values() if truth(pair["a"]) != truth(pair["b"])]
    invariant = [pair for pair in pairs.values() if truth(pair["a"]) == truth(pair["b"])]
    both_correct = lambda values: float(np.mean([
        all(predicted(row) == truth(row) for row in pair.values()) for pair in values
    ])) if values else None
    result = {
        "pairs": len(relevant),
        "flip_rate": float(np.mean([predicted(pair["a"]) != predicted(pair["b"]) for pair in relevant])) if relevant else None,
        "both_correct_rate": both_correct(relevant),
    }
    if invariant:
        result.update({
            "invariant_pairs": len(invariant),
            "invariance_rate": float(np.mean([predicted(pair["a"]) == predicted(pair["b"]) for pair in invariant])),
            "invariant_both_correct_rate": both_correct(invariant),
        })
    return result


def none_diagnostics(rows):
    result = {}
    none_keys = {key for key, _ in NONE_OPTIONS} | {"none_of_these"}
    for variant in ("none_present", "none_absent"):
        selected = [row for row in rows if row.get("variant") == variant]
        if not selected:
            result[variant] = {"n": 0, "mean_none_probability": None, "none_selection_rate": None, "accuracy": None}
            continue
        identified = []
        for row in selected:
            matches = ["none_of_these"] if "none_of_these" in row["keys"] else [key for key in row["keys"] if key in none_keys]
            if len(matches) != 1:
                raise ValueError(f"{variant} row must contain exactly one recognized none option")
            identified.append((row, matches[0]))
        none_probabilities = [row["probabilities"][row["keys"].index(key)] for row, key in identified]
        result[variant] = {
            "n": len(selected),
            "mean_none_probability": float(np.mean(none_probabilities)),
            "none_selection_rate": float(np.mean([row["predicted_key"] == key for row, key in identified])),
            "accuracy": float(np.mean([row["correct"] for row in selected])),
        }
    return result


def variant_reports(rows, temperature):
    variants = defaultdict(list)
    for row in rows:
        variants[row.get("variant", "clean")].append(row)
    return {name: {"raw": metric_report(values), "calibrated": metric_report(values, temperature)}
            for name, values in sorted(variants.items()) if name != "clean"}


def cluster_index(rows):
    sources = defaultdict(lambda: defaultdict(list))
    for row in rows:
        sources[row["source"]][row["group"]].append(row)
    return {source: list(groups.values()) for source, groups in sorted(sources.items())}


def resample_rows(index, rng):
    sampled = []
    for units in index.values():
        for position in rng.integers(0, len(units), size=len(units)):
            sampled.extend(units[int(position)])
    return sampled


def report_values(rows, temperature):
    core = metrics(rows, temperature)
    by_source = grouped_metrics(rows, "source", temperature)
    values = {
        "accuracy": core["accuracy"],
        "nll": core["nll"],
        "brier": core["brier"],
        "ece": core["ece"],
        "macro_source_accuracy": macro_source_metrics(by_source)["accuracy"],
    }
    if "score_mae_levels" in core:
        values["score_mae_levels"] = core["score_mae_levels"]
        values["ranked_probability_score"] = core["ranked_probability_score"]
    return values


def unit_statistics(rows, temperature):
    values = np.zeros(9 + 30, dtype=float)
    for row in rows:
        probabilities = scaled_probabilities(row["probabilities"], temperature)
        label = row["label"]
        target = np.eye(len(probabilities))[label]
        correct = int(probabilities.argmax() == label)
        confidence = float(probabilities.max())
        bin_index = min(int(confidence * 10), 9)
        values[0] += 1
        values[1] += correct
        values[2] += -math.log(max(float(probabilities[label]), EPSILON))
        values[3] += float(((probabilities - target) ** 2).sum())
        values[9 + bin_index] += 1
        values[19 + bin_index] += correct
        values[29 + bin_index] += confidence
        if row["type"] == "score":
            values[4] += 1
            values[5] += abs(float(probabilities @ np.arange(len(probabilities))) - label)
            values[6] += float(((probabilities.cumsum()[:-1] - target.cumsum()[:-1]) ** 2).mean())
    return values


def bootstrap_draw_values(indexes, temperatures, samples, seed):
    rng = np.random.default_rng(seed)
    totals = [np.zeros((samples, 39), dtype=float) for _ in indexes]
    macro_accuracy = [np.zeros(samples, dtype=float) for _ in indexes]
    sources = list(indexes[0])
    if any(list(index) != sources for index in indexes[1:]):
        raise ValueError("paired bootstrap source populations differ")
    for source in sources:
        count = len(indexes[0][source])
        if any(len(index[source]) != count for index in indexes[1:]):
            raise ValueError("paired bootstrap cluster populations differ")
        sampled = rng.multinomial(count, np.full(count, 1 / count), size=samples)
        for position, (index, temperature) in enumerate(zip(indexes, temperatures)):
            statistics = np.stack([unit_statistics(unit, temperature) for unit in index[source]])
            contribution = sampled @ statistics
            totals[position] += contribution
            macro_accuracy[position] += contribution[:, 1] / contribution[:, 0]
    outputs = []
    for total, macro in zip(totals, macro_accuracy):
        size = total[:, 0]
        output = {
            "accuracy": total[:, 1] / size,
            "nll": total[:, 2] / size,
            "brier": total[:, 3] / size,
            "ece": np.abs(total[:, 19:29] - total[:, 29:39]).sum(axis=1) / size,
            "macro_source_accuracy": macro / len(sources),
        }
        if total[:, 4].max() > 0:
            if (total[:, 4] == 0).any():
                raise ValueError("bootstrap draw contains no Score rows")
            output["score_mae_levels"] = total[:, 5] / total[:, 4]
            output["ranked_probability_score"] = total[:, 6] / total[:, 4]
        outputs.append(output)
    return outputs


def bootstrap_report(rows, temperature=1.0, samples=10000, seed=BOOTSTRAP_SEED):
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    selected = clean_rows(rows)
    if not selected:
        raise ValueError("cannot bootstrap an empty clean population")
    observed = report_values(selected, temperature)
    draws = bootstrap_draw_values([cluster_index(selected)], [temperature], samples, seed)[0]
    return {
        name: {"estimate": float(value), "ci95": np.quantile(draws[name], [0.025, 0.975]).tolist()}
        for name, value in observed.items()
    } | {"samples": samples, "seed": seed,
         "unit": "source-stratified (source, group_id) clusters; sibling questions stay together"}


def aligned_rows(left, right):
    index = lambda rows: {(row["id"], row["question"]): row for row in rows}
    a, b = index(left), index(right)
    if len(a) != len(left) or len(b) != len(right):
        raise ValueError("paired comparison contains duplicate examples")
    if not a or a.keys() != b.keys():
        raise ValueError("paired comparison requires identical complete examples")
    for key in a:
        x, y = a[key], b[key]
        if (x["group"], x["source"], x["keys"], x["label"]) != (y["group"], y["source"], y["keys"], y["label"]):
            raise ValueError("paired comparison metadata differs")
    return a, b


def paired_bootstrap(left, right, temperature=1.0, samples=10000, seed=BOOTSTRAP_SEED):
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    left = clean_rows(left)
    right = clean_rows(right)
    a, b = aligned_rows(left, right)
    units = cluster_index(list(a.values()))
    key_for = {id(row): key for key, row in a.items()}
    right_units = {
        source: [[b[key_for[id(row)]] for row in unit] for unit in source_units]
        for source, source_units in units.items()
    }
    observed_left = report_values(list(a.values()), temperature)
    observed_right = report_values(list(b.values()), temperature)
    names = sorted(observed_left.keys() & observed_right.keys())
    observed = {name: observed_left[name] - observed_right[name] for name in names}
    left_draws, right_draws = bootstrap_draw_values([units, right_units], [temperature, temperature], samples, seed)
    return {
        name: {"estimate": float(value),
               "ci95": np.quantile(left_draws[name] - right_draws[name], [0.025, 0.975]).tolist()}
        for name, value in observed.items()
    } | {"samples": samples, "seed": seed, "direction": "left minus right",
         "unit": "paired source-stratified (source, group_id) clusters"}


@contextmanager
def zero_level_embedding(model):
    saved = model.level.weight.detach().clone()
    with torch.no_grad():
        model.level.weight.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            model.level.weight.copy_(saved)


def score_requests(requests):
    return [
        request for request in requests
        if request["_meta"].get("variant", "clean") == "clean"
        and any(question["type"] == "score" for question in request["questions"].values())
    ]


def level_zero_report(requests, predictor, learned_rows, temperature, samples):
    if predictor.metadata["readout"]["head_kind"] != "pointer":
        raise ValueError("level-zero ablation is predeclared for PointerHead only")
    selected_requests = score_requests(requests)
    if not selected_requests:
        raise ValueError("level-zero ablation has no clean Score questions")
    selected_ids = {request["_meta"]["id"] for request in selected_requests}
    learned_all = [row for row in clean_rows(learned_rows) if row["id"] in selected_ids]
    with zero_level_embedding(predictor.model):
        zeroed_all, _ = predict_rows(selected_requests, predictor)
    learned_index, zeroed_index = aligned_rows(learned_all, zeroed_all)
    differences = {
        key: float(np.max(np.abs(np.asarray(learned_index[key]["probabilities"]) - np.asarray(zeroed_index[key]["probabilities"]))))
        for key in learned_index
    }
    learned = [row for row in learned_all if row["type"] == "score"]
    zeroed = [row for row in zeroed_all if row["type"] == "score"]
    score_keys = {(row["id"], row["question"]) for row in learned}
    max_change = max(differences[key] for key in score_keys)
    non_score = [difference for key, difference in differences.items() if key not in score_keys]
    paired = paired_bootstrap(learned, zeroed, temperature, samples)
    return zeroed, {
        "n": len(zeroed),
        "max_abs_probability_change": max_change,
        "max_non_score_probability_change": max(non_score, default=0.0),
        "used": max_change > 1e-4,
        "learned": metric_report(learned, temperature),
        "zeroed": metric_report(zeroed, temperature),
        "paired_learned_minus_zeroed": paired,
        "useful": metrics(learned, temperature)["nll"] < metrics(zeroed, temperature)["nll"]
                  and paired["nll"]["ci95"][1] < 0,
    }


def coverage(records, rows):
    expected_questions = sum(len(record["questions"]) for record in records)
    result = {
        "requested_records": len(records),
        "requested_questions": expected_questions,
        "evaluated_records": len({row["id"] for row in rows}),
        "evaluated_questions": len(rows),
        "rejected_records": 0,
        "truncated_records": 0,
    }
    if result["evaluated_records"] != result["requested_records"] or result["evaluated_questions"] != expected_questions:
        raise ValueError("evaluation coverage is incomplete")
    return result


def evaluation_report(records, rows, latencies, predictor, temperature, seed, permutations, samples):
    selected = clean_rows(rows)
    return {
        "coverage": coverage(records, rows),
        "clean_questions": len(selected),
        "raw": metric_report(selected),
        "calibrated": metric_report(selected, temperature),
        "variants": variant_reports(rows, temperature),
        "none_of_the_above": none_diagnostics(rows),
        "contrastive": contrastive_report(rows),
        "permutation": permutation_study(records, predictor, seed, permutations),
        "packed_vs_separate": packed_vs_separate(records, predictor),
        "bootstrap_raw": bootstrap_report(selected, 1.0, samples),
        "bootstrap_calibrated": bootstrap_report(selected, temperature, samples),
        "latency_seconds": {
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
        },
    }


def validate_transfer_suite(training_suite, transfer_suite, base, revision):
    training_suite, transfer_suite = Path(training_suite), Path(transfer_suite)
    training_manifest = manifest(training_suite)
    transfer_manifest = manifest(transfer_suite)
    if not transfer_manifest.get("eval_only"):
        raise ValueError("transfer suite must be declared eval-only")
    if transfer_manifest.get("base_revisions", {}).get(base) != revision:
        raise ValueError("transfer suite pins a different base revision")
    for split in ("train", "calibration"):
        records = load_split(transfer_suite, split)
        if records:
            raise ValueError(f"transfer {split} split must be empty")
    expected = transfer_manifest.get("excluded_training_states_from")
    if expected and Path(expected).name != training_suite.name:
        raise ValueError("transfer suite was not decontaminated against the training suite")
    records = load_split(transfer_suite, "development")
    sources = {record["_meta"]["source"] for record in records}
    tasks = {question.get("src") for record in records for question in record["questions"].values() if question.get("src")}
    trainable = set(TRAINABLE)
    declared = set(transfer_manifest.get("eval_only_sources", []))
    # Admission rule: the transfer manifest declares the source eval-only AND hev.data never trains it.
    # Membership in EVAL_ONLY is deliberately not required here, so a holdout family introduced by a newer
    # suite version is accepted once its manifest declares it; training refusal is EVAL_ONLY's job.
    admissible = declared - trainable
    forbidden = (sources & trainable) | (tasks & trainable)
    undeclared = sources - declared
    unknown = sources - admissible
    if forbidden or unknown or undeclared:
        raise ValueError(f"transfer suite contains non-eval-only sources: {sorted(forbidden | unknown | undeclared)}")
    if base_revision(training_suite, base) != revision or training_manifest.get("eval_only"):
        raise ValueError("training suite/base provenance is invalid")
    return records


def write_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite result artifact: {path}")
    write_json(temporary, value)
    os.replace(temporary, path)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Hev run directory, or for --checkpoint-kind kev a local dir / hf:// reference")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--transfer")
    ap.add_argument("--out", help="fresh output directory (must not exist); default writes into the run directory")
    ap.add_argument("--checkpoint-kind", choices=["hev", "kev"], default="hev")
    ap.add_argument("--allow-cross-suite", action="store_true",
                    help="permit a suite/source mismatch with the checkpoint; recorded in result.json; requires --out")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--permutations", type=int, default=6)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--level-zero-ablation", action="store_true")
    return ap


def validate_args(args):
    """Flag consistency, checked before any output directory or model is touched."""
    if args.permutations < 2 or args.bootstrap_samples < 1:
        raise ValueError("permutations must be at least two and bootstrap samples must be positive")
    if args.checkpoint_kind == "kev" and args.out is None:
        raise ValueError("--checkpoint-kind kev requires --out; an external checkpoint has no run directory to write into")
    if args.checkpoint_kind == "kev" and args.level_zero_ablation:
        raise ValueError("--level-zero-ablation needs a Hev PointerHead level embedding; kev checkpoints have none")
    if args.allow_cross_suite and args.out is None:
        raise ValueError("--allow-cross-suite requires --out; cross-suite results are never written into the run directory")


def load_predictor(kind, reference, device):
    if kind == "kev":
        from .kev import KevPredictor  # lazy: only external checkpoints need the vendored kev stack
        return KevPredictor(reference, device)
    return LocalPredictor(Path(reference), device)


def changed_files(recorded, current):
    return sorted(name for name in set(recorded) | set(current) if recorded.get(name) != current.get(name))


def hev_gates(run, predictor, current_sources, allow_cross_suite):
    """Attribution gates for a local Hev run.

    Always enforced: training_config self-hash, checkpoint/config hash agreement, checkpoint/config provenance
    agreement. Relaxed only by --allow-cross-suite: recorded training-time source hashes must equal the current
    hev.suite.source_hashes(). Returns (training_config, training_metrics, provenance)."""
    training_config = json.loads((run / "training_config.json").read_text())
    training_metrics = json.loads((run / "training_metrics.json").read_text())
    expected_config_hash = object_digest({key: value for key, value in training_config.items() if key != "config_sha256"})
    if training_config.get("config_sha256") != expected_config_hash:
        raise ValueError("training config hash mismatch")
    if predictor.metadata.get("training_config_sha256") != expected_config_hash:
        raise ValueError("checkpoint and training config hashes differ")
    recorded_sources = training_config.get("provenance", {}).get("source_hashes")
    if recorded_sources != predictor.metadata.get("provenance", {}).get("source_hashes"):
        raise ValueError("checkpoint and training provenance differ")
    changed = changed_files(recorded_sources or {}, current_sources)
    if changed and not allow_cross_suite:
        raise ValueError("source changed after training; result cannot be attributed to this checkpoint")
    provenance = {
        "kind": "hev",
        "gates": {
            "training_config": "enforced",
            "source_hashes": "relaxed by --allow-cross-suite" if allow_cross_suite else "enforced",
        },
        "training_source_hashes": recorded_sources,
        "evaluation_source_hashes": current_sources,
        "source_hashes_match": not changed,
        "changed_files": changed,
    }
    return training_config, training_metrics, provenance


def kev_provenance(predictor):
    metadata = predictor.metadata
    return {
        "kind": "kev",
        "gates": {
            "training_config": "not applicable: external checkpoint",
            "source_hashes": "not applicable: external checkpoint",
        },
        "checkpoint": metadata.get("checkpoint"),
        "kev_source": metadata.get("kev_source"),
        "training_args": metadata.get("training_args"),
    }


def evaluate(args):
    validate_args(args)
    kind = args.checkpoint_kind
    run = Path(args.run) if kind == "hev" else None
    out = resolve_output_dir(args.run, args.out)
    paths = prepare_outputs(out, bool(args.transfer), args.level_zero_ablation)
    failure = out / "evaluation_failure.json"
    try:
        suite = Path(args.suite)
        suite_sha256 = digest(suite / "manifest.json")
        transfer_sha256 = digest(Path(args.transfer) / "manifest.json") if args.transfer else None
        device = args.device or default_device()
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        predictor = load_predictor(kind, args.run, device)
        training_suite_sha256 = predictor.metadata.get("suite_sha256")
        if training_suite_sha256 != suite_sha256 and not args.allow_cross_suite:
            raise ValueError("checkpoint and training suite hashes differ")
        current_sources = source_hashes()
        if kind == "hev":
            training_config, training_metrics, provenance = hev_gates(run, predictor, current_sources, args.allow_cross_suite)
            head = predictor.metadata["readout"]["head_kind"]
        else:
            training_config = training_metrics = None
            provenance = kev_provenance(predictor)
            head = "kev"
        cross_suite = {
            "training_suite_sha256": training_suite_sha256,
            "evaluation_suite_sha256": suite_sha256,
            "transfer_suite_sha256": transfer_sha256,
            "suite_differs": training_suite_sha256 != suite_sha256,
            "allowed_by_flag": True,
        } if args.allow_cross_suite else None
        calibration = load_split(suite, "calibration")
        development = load_split(suite, "development")
        calibration_rows, _ = predict_rows(calibration, predictor)
        temperature_fit = fit_temperature(calibration_rows)
        temperature = temperature_fit["temperature"]
        decision_rows, decision_latencies = predict_rows(development, predictor)
        transfer_records = validate_transfer_suite(suite, args.transfer, predictor.metadata.get("base"), predictor.metadata.get("base_revision")) if args.transfer else None
        transfer_rows, transfer_latencies = predict_rows(transfer_records, predictor) if transfer_records is not None else (None, None)
        decision = evaluation_report(development, decision_rows, decision_latencies, predictor, temperature,
                                     args.seed, args.permutations, args.bootstrap_samples)
        transfer = evaluation_report(transfer_records, transfer_rows, transfer_latencies, predictor, temperature,
                                     args.seed, args.permutations, args.bootstrap_samples) if transfer_records is not None else None
        zero_rows = zero_report = None
        if args.level_zero_ablation:
            zero_rows, zero_report = level_zero_report(development, predictor, decision_rows, temperature, args.bootstrap_samples)
        if source_hashes() != current_sources or digest(suite / "manifest.json") != suite_sha256:
            raise ValueError("source or training suite changed during evaluation")
        if args.transfer and digest(Path(args.transfer) / "manifest.json") != transfer_sha256:
            raise ValueError("transfer suite changed during evaluation")
        write_json(paths["calibration"], calibration_rows)
        write_json(paths["decision"], decision_rows)
        if transfer_rows is not None:
            write_json(paths["transfer"], transfer_rows)
        if zero_rows is not None:
            write_json(paths["level_zero"], zero_rows)
        artifact_hashes = {
            name: {"path": str(path.relative_to(out)), "sha256": digest(path),
                   "rows": len(json.loads(path.read_text()))}
            for name, path in paths.items() if name != "result"
        }
        result = {
            "schema_version": 2,
            "status": "success",
            "run": str(run) if run is not None else args.run,
            "checkpoint_kind": kind,
            "output_dir": str(out),
            "test_evaluated": False,
            "head": head,
            "base": predictor.metadata.get("base"),
            "base_revision": predictor.metadata.get("base_revision"),
            "training_suite_sha256": training_suite_sha256,
            "transfer_suite_sha256": transfer_sha256,
            "evaluation_suite": {"path": str(suite), "manifest_sha256": suite_sha256},
            "transfer_suite": {"path": str(args.transfer), "manifest_sha256": transfer_sha256} if args.transfer else None,
            "cross_suite": cross_suite,
            "provenance": provenance,
            "training_config": training_config,
            "training_metrics": training_metrics,
            "evaluation_source_hashes": current_sources,
            "evaluation_config": {
                "seed": args.seed,
                "permutations": args.permutations,
                "bootstrap_samples": args.bootstrap_samples,
                "level_zero_ablation": args.level_zero_ablation,
            },
            "evaluation_flags": {
                "out": args.out,
                "checkpoint_kind": kind,
                "allow_cross_suite": args.allow_cross_suite,
            },
            "temperature_fit": temperature_fit,
            "decision": decision,
            "transfer": transfer,
            "level_zero_ablation": zero_report,
            "artifact_hashes": artifact_hashes,
        }
        write_atomic(paths["result"], result)
        print(json.dumps({
            "head": result["head"],
            "decision_clean": decision["raw"]["overall"],
            "transfer_clean": transfer and transfer["raw"]["overall"],
            "temperature": temperature,
        }, indent=2), flush=True)
        return result
    except Exception as error:
        if not failure.exists():
            write_json(failure, {"error_type": type(error).__name__, "error": str(error)})
        raise


def main():
    ap = parser()
    args = ap.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        ap.error(str(error))
    evaluate(args)


if __name__ == "__main__":
    main()
