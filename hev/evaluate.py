"""Evaluate trained Hev checkpoints on calibration and development splits."""
import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import materialize
from .model import DecisionModel, encode, load_tokenizer
from .suite import digest, load_split, manifest, write_json

EPSILON = 1e-9


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
    nll = []
    accuracy = []
    confidence = []
    brier = []
    score_mae = []
    score_rps = []
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


def grouped_metrics(rows, key, temperature=1.0):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {name: metrics(group, temperature) for name, group in sorted(groups.items())}


def macro_source_metrics(by_source):
    names = ("accuracy", "nll", "brier", "ece", "mean_confidence")
    return {"n_sources": len(by_source), **{
        name: float(np.mean([group[name] for group in by_source.values()])) for name in names
    }}


def metric_report(rows, temperature=1.0):
    by_source = grouped_metrics(rows, "source", temperature)
    return {
        "overall": metrics(rows, temperature),
        "by_source": by_source,
        "by_type": grouped_metrics(rows, "type", temperature),
        "macro_source": macro_source_metrics(by_source),
    }


def fit_temperature(rows):
    if not rows:
        raise ValueError("cannot fit temperature on an empty calibration split")
    candidates = np.exp(np.linspace(np.log(0.25), np.log(4.0), 81))
    losses = [metric_report(rows, float(candidate))["macro_source"]["nll"] for candidate in candidates]
    index = int(np.argmin(losses))
    return {
        "temperature": float(candidates[index]),
        "n": len(rows),
        "raw_macro_nll": metric_report(rows)["macro_source"]["nll"],
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
    rows = []
    for (question_id, question), values in zip(request["questions"].items(), probabilities):
        keys = question_keys(question)
        vector = validate_probabilities(values)
        if len(keys) != len(vector):
            raise ValueError("probability count does not match option count")
        label = semantic_label(question, keys)
        rows.append({
            "id": request["_meta"]["id"],
            "question": question_id,
            "source": request["_meta"]["source"],
            "task": question.get("src", request["_meta"]["source"]),
            "type": question["type"],
            "keys": keys,
            "label": label,
            "probabilities": vector.tolist(),
            "correct": bool(vector.argmax() == label),
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


def prepare_outputs(run):
    paths = [Path(run) / "eval.json", Path(run) / "eval_rows.json"]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite evaluation artifacts: {existing}")
    return paths


class LocalPredictor:
    def __init__(self, run, device):
        run = Path(run)
        training_metrics = json.loads((run / "training_metrics.json").read_text())
        if training_metrics.get("status") != "success":
            raise ValueError("checkpoint training did not complete successfully")
        metadata = torch.load(run / "readout.pt", map_location="cpu", weights_only=True)
        tokenizer = load_tokenizer(metadata["base"], revision=metadata["base_revision"])
        model = DecisionModel(
            metadata["base"],
            tokenizer,
            device,
            lora=None,
            revision=metadata["base_revision"],
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
    rows = []
    latencies = []
    for request in requests:
        probabilities, latency = predictor(request)
        rows.extend(rows_for_record(request, probabilities))
        latencies.append(latency)
    expected_questions = sum(len(request["questions"]) for request in requests)
    if len(rows) != expected_questions:
        raise ValueError("evaluation coverage is incomplete")
    return rows, latencies


def one_question(request, question_id, question):
    return {
        "state": request["state"],
        "questions": {question_id: question},
        "_meta": request["_meta"],
    }


def permutation_study(requests, predictor, seed=1, permutations=6):
    if permutations < 2:
        raise ValueError("permutations must be at least two")
    rng = random.Random(seed)
    flips = []
    spreads = []
    for request in requests:
        for question_id, question in request["questions"].items():
            if question["type"] != "choice" or len(question["criteria"]) < 3:
                continue
            predictions = []
            correct_probabilities = []
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
    differences = []
    records = 0
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
                "max_abs_probability_difference": None}
    return {
        "n_records": records,
        "n_questions": len(differences),
        "mean_abs_probability_difference": float(np.mean(differences)),
        "max_abs_probability_difference": float(np.max(differences)),
    }


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--permutations", type=int, default=6)
    return ap


def evaluate(args):
    run = Path(args.run)
    summary_path, rows_path = prepare_outputs(run)
    suite = Path(args.suite)
    suite_sha256 = digest(suite / "manifest.json")
    device = args.device or default_device()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    predictor = LocalPredictor(run, device)
    if predictor.metadata["suite_sha256"] != suite_sha256:
        raise ValueError("checkpoint and evaluation suite hashes differ")
    calibration = load_split(suite, "calibration")
    development = load_split(suite, "development")
    started = time.perf_counter()
    calibration_rows, _ = predict_rows(calibration, predictor)
    temperature_fit = fit_temperature(calibration_rows)
    development_rows, latencies = predict_rows(development, predictor)
    raw = metric_report(development_rows)
    calibrated = metric_report(development_rows, temperature_fit["temperature"])
    permutation = permutation_study(development, predictor, args.seed, args.permutations)
    packed = packed_vs_separate(development, predictor)
    sync(device)
    elapsed = time.perf_counter() - started
    expected_questions = sum(len(request["questions"]) for request in development)
    suite_manifest = manifest(suite)
    summary = {
        "run": str(run),
        "suite": str(suite),
        "suite_sha256": suite_sha256,
        "split": "development",
        "test_evaluated": False,
        "base": predictor.metadata["base"],
        "base_revision": predictor.metadata["base_revision"],
        "head": predictor.metadata["readout"]["head_kind"],
        "temperature_fit": temperature_fit,
        "raw": raw,
        "calibrated": calibrated,
        "permutation": permutation,
        "packed_vs_separate": packed,
        "coverage": {
            "requested_records": len(development),
            "requested_questions": expected_questions,
            "evaluated_records": len(development),
            "evaluated_questions": len(development_rows),
            "rejected_records": 0,
            "truncated_records": 0,
        },
        "heldout_sources": suite_manifest.get("holdout_sources", []),
        "wall_seconds": elapsed,
        "latency_seconds": {
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
        },
    }
    write_json(rows_path, development_rows)
    write_json(summary_path, summary)
    print(json.dumps({
        "raw": raw["overall"],
        "temperature": temperature_fit["temperature"],
        "permutation": permutation,
        "packed_vs_separate": packed,
    }, indent=2), flush=True)


def main():
    args = parser().parse_args()
    if args.permutations < 2:
        raise ValueError("permutations must be at least two")
    evaluate(args)


if __name__ == "__main__":
    main()
