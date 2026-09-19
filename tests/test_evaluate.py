"""Offline regression tests for Hev evaluation metrics and mechanism checks."""
import math

import pytest

from hev.evaluate import (
    ece,
    fit_temperature,
    macro_source_metrics,
    metric_report,
    metrics,
    packed_vs_separate,
    permutation_study,
    prepare_outputs,
    rows_for_record,
    validate_probabilities,
)


def row(probabilities, label, source="a", qtype="choice"):
    return {
        "source": source,
        "type": qtype,
        "probabilities": probabilities,
        "label": label,
    }


def request(two_questions=False):
    questions = {
        "choice": {
            "type": "choice",
            "instructions": "pick",
            "criteria": {"a": None, "b": None, "c": None},
            "label": "a",
            "src": "fixture",
        }
    }
    if two_questions:
        questions["noul"] = {
            "type": "noul",
            "instructions": "yes?",
            "label": True,
            "src": "fixture",
        }
    return {
        "state": "state",
        "questions": questions,
        "_meta": {"id": "fixture/1", "source": "fixture"},
    }


class SemanticPredictor:
    def __call__(self, item):
        output = []
        for question in item["questions"].values():
            if question["type"] == "choice":
                weights = {"a": 0.7, "b": 0.2, "c": 0.1}
                output.append([weights[key] for key in question["criteria"]])
            else:
                output.append([0.25, 0.75])
        return output, 0.0


class PositionPredictor:
    def __call__(self, item):
        return [[0.7, 0.2, 0.1]], 0.0


def test_ece_equal_width_bins_include_one():
    assert ece([0.05, 0.15, 1.0], [0, 1, 1]) == pytest.approx(0.3)


def test_core_metrics_have_known_values():
    result = metrics([row([0.8, 0.2], 0), row([0.25, 0.75], 0)])
    assert result["accuracy"] == 0.5
    assert result["nll"] == pytest.approx((-math.log(0.8) - math.log(0.25)) / 2)
    assert result["brier"] == pytest.approx((0.08 + 1.125) / 2)
    assert result["mean_confidence"] == pytest.approx(0.775)


def test_score_metrics_have_known_values():
    result = metrics([row([0.25, 0.5, 0.25], 1, qtype="score")])
    assert result["score_mae_levels"] == 0
    assert result["ranked_probability_score"] == pytest.approx(0.0625)


def test_grouping_and_macro_sources_are_unweighted():
    rows = [row([0.9, 0.1], 0, "large"), row([0.9, 0.1], 0, "large"), row([0.9, 0.1], 1, "small")]
    report = metric_report(rows)
    assert report["overall"]["accuracy"] == pytest.approx(2 / 3)
    assert report["macro_source"]["accuracy"] == 0.5
    assert set(report["by_source"]) == {"large", "small"}
    assert set(report["by_type"]) == {"choice"}
    assert macro_source_metrics(report["by_source"])["n_sources"] == 2


def test_temperature_fit_uses_macro_calibration_nll():
    rows = [row([0.6, 0.4], 0, "a"), row([0.6, 0.4], 0, "b")]
    fitted = fit_temperature(rows)
    assert fitted["temperature"] == pytest.approx(0.25)
    assert fitted["fitted_macro_nll"] < fitted["raw_macro_nll"]


def test_permutation_study_aligns_semantic_keys():
    stable = permutation_study([request()], SemanticPredictor(), seed=1, permutations=6)
    assert stable["n"] == 1
    assert stable["argmax_flip_rate"] == 0
    assert stable["max_correct_probability_spread"] == pytest.approx(0)
    sensitive = permutation_study([request()], PositionPredictor(), seed=1, permutations=6)
    assert sensitive["argmax_flip_rate"] == 1


def test_packed_vs_separate_uses_every_question():
    result = packed_vs_separate([request(two_questions=True)], SemanticPredictor())
    assert result == {
        "n_records": 1,
        "n_questions": 2,
        "mean_abs_probability_difference": 0.0,
        "max_abs_probability_difference": 0.0,
    }


def test_rows_preserve_semantic_keys_and_metadata():
    rows = rows_for_record(request(two_questions=True), [[0.7, 0.2, 0.1], [0.25, 0.75]])
    assert rows[0]["keys"] == ["a", "b", "c"]
    assert rows[0]["label"] == 0 and rows[0]["correct"]
    assert rows[1]["keys"] == ["false", "true"] and rows[1]["label"] == 1
    assert all(item["source"] == "fixture" for item in rows)


@pytest.mark.parametrize("values", [[], [0.5, float("nan")], [-0.1, 1.1], [0.2, 0.2]])
def test_invalid_probability_vectors_are_rejected(values):
    with pytest.raises(ValueError):
        validate_probabilities(values)


def test_empty_metrics_and_existing_outputs_are_refused(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        metrics([])
    run = tmp_path / "run"
    run.mkdir()
    (run / "eval.json").write_text("keep")
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_outputs(run)
    assert (run / "eval.json").read_text() == "keep"
