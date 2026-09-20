"""Offline tests for the comprehensive M3 three-way comparison aggregate."""
import json

import pytest

from hev.evaluate import report_values
from hev.suite import digest
from hev.three_way import (load_jev_run, metric_aggregate, paired_bootstrap_temperatures,
                           report_aggregate)


def metrics(accuracy, n=10):
    return {
        "n": n,
        "accuracy": accuracy,
        "nll": 1 - accuracy,
        "brier": 0.5 - accuracy / 2,
        "ece": 0.1,
        "mean_confidence": 0.8,
    }


def report(accuracy):
    values = metrics(accuracy)
    return {
        "overall": values,
        "by_task": {"task": values},
        "by_source": {"source": values},
        "by_type": {"choice": values},
    }


def test_metric_and_report_aggregates_preserve_both_seeds():
    aggregate = metric_aggregate([metrics(0.8), metrics(0.9)])
    assert aggregate["n"] == 10
    assert aggregate["accuracy"] == {"mean": pytest.approx(0.85), "range": [0.8, 0.9]}
    combined = report_aggregate([report(0.8), report(0.9)])
    assert combined["overall"]["accuracy"]["mean"] == pytest.approx(0.85)
    assert combined["by_task"]["task"]["accuracy"]["range"] == [0.8, 0.9]
    with pytest.raises(ValueError, match="populations"):
        metric_aggregate([metrics(0.8, n=10), metrics(0.9, n=11)])


def row(probabilities):
    return {
        "id": "example",
        "group": "group",
        "question": "question",
        "source": "source",
        "task": "task",
        "type": "choice",
        "variant": "clean",
        "keys": ["a", "b"],
        "label": 0,
        "probabilities": probabilities,
    }


def test_paired_bootstrap_uses_model_specific_temperatures():
    left, right = [row([0.8, 0.2])], [row([0.6, 0.4])]
    result = paired_bootstrap_temperatures(left, right, 2.0, 0.5, samples=20, seed=7)
    expected = report_values(left, 2.0)["nll"] - report_values(right, 0.5)["nll"]
    assert result["nll"]["estimate"] == pytest.approx(expected)
    assert result["temperatures"] == {"left": 2.0, "right": 0.5}
    assert result["direction"] == "left minus right"
    assert result["samples"] == 20


def test_load_jev_run_verifies_status_coverage_version_and_rows_hash(tmp_path):
    rows = [row([0.8, 0.2])]
    (tmp_path / "rows.json").write_text(json.dumps(rows))
    result = {
        "status": "success",
        "test_evaluated": False,
        "split": "development",
        "coverage": {
            "requested_records": 1,
            "evaluated_records": 1,
            "requested_questions": 1,
            "evaluated_questions": 1,
            "rejected_records": 0,
            "truncated_records": 0,
        },
        "artifact_hashes": {
            "rows": {"path": "rows.json", "sha256": digest(tmp_path / "rows.json"), "rows": 1}
        },
        "provider": {"model_versions": ["jev-1.2.3"]},
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    loaded = load_jev_run(tmp_path, "development")
    assert loaded["version"] == "jev-1.2.3" and loaded["rows"] == rows
    result["provider"]["model_versions"] = ["jev-latest"]
    (tmp_path / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="versioned"):
        load_jev_run(tmp_path, "development")
