"""Evaluate hosted Jev directly on checksum-verified calibration or development splits."""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .evaluate import clean_rows, contrastive_report, metric_report, none_diagnostics, rows_for_record
from .suite import digest, load_split, object_digest, source_hashes, write_json

PRICE_PER_MILLION = 0.042


def normalize_returned(values):
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 1 or not len(probabilities):
        raise ValueError("probabilities must be a nonempty vector")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    total = float(probabilities.sum())
    tolerance = max(1e-5, len(probabilities) * 0.005 + 1e-8)
    if total <= 0 or abs(total - 1) > tolerance:
        raise ValueError(f"invalid returned probability sum: {total}")
    return probabilities / total, total, int((probabilities == 0).sum())


def api_request(record):
    return {
        "state": record["state"],
        "questions": {
            question_id: {key: value for key, value in question.items() if key in ("type", "instructions", "criteria")}
            for question_id, question in record["questions"].items()
        },
    }


def permutation_report(rows):
    originals = {(row["id"], row["question"]): row for row in clean_rows(rows)}
    differences, flips = [], []
    for row in rows:
        if row["variant"] != "permuted" or row["type"] != "choice":
            continue
        original = originals[(row["parent"], row["question"])]
        aligned = np.asarray([row["probabilities"][row["keys"].index(key)] for key in original["keys"]])
        baseline = np.asarray(original["probabilities"])
        differences.append(float(np.max(np.abs(aligned - baseline))))
        flips.append(int(aligned.argmax() != baseline.argmax()))
    return {
        "n": len(differences),
        "orders": 2,
        "argmax_flip_rate": float(np.mean(flips)) if flips else None,
        "mean_max_abs_probability_difference": float(np.mean(differences)) if differences else None,
        "p90_max_abs_probability_difference": float(np.percentile(differences, 90)) if differences else None,
        "max_abs_probability_difference": float(np.max(differences)) if differences else None,
    }


class DirectJevPredictor:
    def __init__(self, key, budget, max_calls):
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        self.client = TypeSafeClient(
            api_key=key,
            model="jev-latest",
            retry=RetryPolicy(max_retries=3),
            timeout=60,
        )
        self.budget = budget
        self.max_calls = max_calls
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_versions = set()
        self.started_at = datetime.now(timezone.utc).isoformat()

    def close(self):
        self.client.close()

    def __call__(self, record):
        if self.calls >= self.max_calls or (self.input_tokens + 65536) * PRICE_PER_MILLION / 1e6 > self.budget:
            raise RuntimeError("Jev evaluation reached the request/token cost cap")
        request = api_request(record)
        started = time.perf_counter()
        response = self.client.system_one(state=request["state"], questions=request["questions"])
        latency_ms = 1000 * (time.perf_counter() - started)
        self.calls += 1
        if response.usage.input_tokens is None:
            raise RuntimeError("Jev returned no input token usage; cannot account for cost")
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens or 0
        self.model_versions.add(response.model)
        if set(response.answers) != set(record["questions"]):
            raise ValueError("Jev answer IDs differ from request IDs")
        probabilities, raw_sums, zero_counts = [], [], []
        for question_id, question in record["questions"].items():
            answer = response.answers[question_id]
            if question["type"] == "noul":
                values = [1 - answer.noul, answer.noul]
            else:
                keys = list(question["criteria"]) if question["type"] == "choice" else list(range(len(question["criteria"])))
                values = [answer.probabilities[key] for key in keys]
            normalized, total, zeros = normalize_returned(values)
            probabilities.append(normalized)
            raw_sums.append(total)
            zero_counts.append(zeros)
        return probabilities, {
            "latency_ms": latency_ms,
            "model": response.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens or 0,
            "raw_probability_sums": raw_sums,
            "zero_counts": zero_counts,
        }

    def accounting(self):
        return {
            "provider": "TypeSafe direct API",
            "model_alias": "jev-latest",
            "model_versions": sorted(self.model_versions),
            "started_at": self.started_at,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "listed_input_usd_per_million": PRICE_PER_MILLION,
            "estimated_usd": self.input_tokens * PRICE_PER_MILLION / 1e6,
            "budget_usd": self.budget,
            "max_calls": self.max_calls,
            "sdk": f"typesafe-sdk=={version('typesafe-sdk')}",
            "retry_policy": {"max_retries": 3},
        }


def evaluate(suite, split, output, budget, max_calls):
    suite, output = Path(suite), Path(output)
    if split not in ("calibration", "development"):
        raise ValueError("Jev evaluation permits calibration or development only")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing Jev evaluation: {output}")
    if not 0 < budget <= 1 or not 1 <= max_calls <= 2000:
        raise ValueError("budget must be in (0, 1] and max_calls in [1, 2000]")
    records = load_split(suite, split)
    output.mkdir(parents=True)
    predictor = DirectJevPredictor(os.environ.get("TYPESAFE_API_KEY"), budget, max_calls)
    rows, latencies = [], []
    coverage = {
        "requested_records": len(records),
        "requested_questions": sum(len(record["questions"]) for record in records),
        "evaluated_records": 0,
        "evaluated_questions": 0,
        "rejected_records": 0,
        "truncated_records": 0,
    }
    try:
        with (output / "predictions.jsonl").open("w") as stream:
            for record in records:
                try:
                    probabilities, metadata = predictor(record)
                    new_rows = rows_for_record(record, probabilities)
                    for row, total, zeros in zip(new_rows, metadata["raw_probability_sums"], metadata["zero_counts"]):
                        row["raw_probability_sum"] = total
                        row["zero_count"] = zeros
                except Exception as error:
                    coverage["rejected_records"] += 1
                    write_json(output / "failure.json", {
                        "error_type": type(error).__name__,
                        "record_id": record["_meta"]["id"],
                        "coverage": coverage,
                    })
                    raise
                stream.write(json.dumps({
                    "request_sha256": object_digest(api_request(record)),
                    "id": record["_meta"]["id"],
                    "provider": metadata,
                    "rows": new_rows,
                }, allow_nan=False) + "\n")
                stream.flush()
                rows.extend(new_rows)
                latencies.append(metadata["latency_ms"])
                coverage["evaluated_records"] += 1
                coverage["evaluated_questions"] += len(new_rows)
                if coverage["evaluated_records"] % 50 == 0:
                    print(f"evaluated {coverage['evaluated_records']}/{len(records)}", flush=True)
        write_json(output / "rows.json", rows)
        selected = clean_rows(rows)
        result = {
            "schema_version": 1,
            "status": "success",
            "test_evaluated": False,
            "split": split,
            "suite": str(suite),
            "suite_manifest_sha256": digest(suite / "manifest.json"),
            "raw": metric_report(selected),
            "variants": {
                name: metric_report([row for row in rows if row["variant"] == name])
                for name in sorted({row["variant"] for row in rows} - {"clean"})
            },
            "contrastive": contrastive_report(rows),
            "none_diagnostics": none_diagnostics(rows),
            "permutation": permutation_report(rows),
            "metric_policy": {
                "nll_floor": 1e-9,
                "renormalize_returned_probabilities": True,
                "sum_tolerance": "max(1e-5, option_count * 0.005 + 1e-8)",
                "raw_sums_outside_1e_5": sum(abs(row["raw_probability_sum"] - 1) > 1e-5 for row in rows),
                "returned_zeros": sum(row["zero_count"] for row in rows),
            },
            "coverage": coverage,
            "latency_ms": {
                "median": float(np.median(latencies)),
                "p95": float(np.percentile(latencies, 95)),
            },
            "provider": predictor.accounting(),
            "provenance": {"source_hashes": source_hashes()},
        }
        result["artifact_hashes"] = {
            "rows": {"path": "rows.json", "sha256": digest(output / "rows.json"), "rows": len(rows)},
            "predictions": {"path": "predictions.jsonl", "sha256": digest(output / "predictions.jsonl"), "records": len(records)},
        }
        write_json(output / "result.json", result)
        return result
    finally:
        predictor.close()
        write_json(output / "usage.json", predictor.accounting())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--split", choices=("calibration", "development"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--max-calls", type=int, required=True)
    args = parser.parse_args()
    result = evaluate(args.suite, args.split, args.out, args.budget, args.max_calls)
    print(json.dumps({
        "raw": result["raw"]["overall"],
        "coverage": result["coverage"],
        "provider": result["provider"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
