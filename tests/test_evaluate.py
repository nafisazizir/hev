"""Offline regression tests for Hev evaluation metrics and mechanism checks."""
import math
from pathlib import Path

import pytest
import torch

from hev.evaluate import (
    bootstrap_report,
    clean_rows,
    contrastive_report,
    ece,
    fit_temperature,
    level_zero_report,
    macro_source_metrics,
    metric_report,
    metrics,
    none_diagnostics,
    packed_vs_separate,
    paired_bootstrap,
    permutation_study,
    prepare_outputs,
    rows_for_record,
    validate_probabilities,
    validate_transfer_suite,
    zero_level_embedding,
)


def row(probabilities, label, source="a", qtype="choice", **extra):
    return {
        "source": source,
        "task": extra.pop("task", source),
        "type": qtype,
        "variant": extra.pop("variant", "clean"),
        "probabilities": probabilities,
        "label": label,
        **extra,
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
        "p90_abs_probability_difference": 0.0,
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


def full_row(identifier, group, probabilities, label, source="a", question="q", keys=None, **extra):
    keys = keys or [str(index) for index in range(len(probabilities))]
    prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
    return row(
        probabilities,
        label,
        source,
        id=identifier,
        group=group,
        question=question,
        keys=keys,
        predicted_key=keys[prediction],
        correct=prediction == label,
        confidence=max(probabilities),
        **extra,
    )


def test_rows_preserve_group_variant_pair_and_parent_metadata():
    item = request()
    item["_meta"].update({"group_id": "group", "variant": "none_present", "pair_id": "pair", "sibling": "a", "parent_id": "parent"})
    result = rows_for_record(item, [[0.7, 0.2, 0.1]])[0]
    assert {key: result[key] for key in ("group", "variant", "pair_id", "sibling", "parent")} == {
        "group": "group", "variant": "none_present", "pair_id": "pair", "sibling": "a", "parent": "parent"
    }
    assert result["predicted_key"] == "a"


def test_clean_rows_and_temperature_exclude_diagnostic_variants():
    base = [
        full_row("1", "g1", [0.6, 0.4], 0, task="one"),
        full_row("2", "g2", [0.6, 0.4], 0, source="b", task="two"),
    ]
    diagnostic = full_row("3", "g3", [0.001, 0.999], 0, variant="permuted")
    assert clean_rows(base + [diagnostic]) == base
    assert fit_temperature(base + [diagnostic]) == fit_temperature(base)


def test_contrastive_report_uses_semantic_keys_and_rejects_incomplete_pairs():
    rows = [
        full_row("a", "pair", [0.9, 0.1], 0, keys=["no", "yes"], pair_id="p", sibling="a"),
        full_row("b", "pair", [0.1, 0.9], 1, keys=["no", "yes"], pair_id="p", sibling="b"),
    ]
    assert contrastive_report(rows) == {"pairs": 1, "flip_rate": 1.0, "both_correct_rate": 1.0}
    with pytest.raises(ValueError, match="incomplete"):
        contrastive_report(rows[:1])


def test_none_diagnostics_separate_mass_selection_and_accuracy():
    rows = [
        full_row("p", "p", [0.8, 0.2], 0, keys=["answer", "none_of_these"], variant="none_present"),
        full_row("a", "a", [0.1, 0.9], 1, keys=["answer", "none_of_these"], variant="none_absent"),
    ]
    report = none_diagnostics(rows)
    assert report["none_present"] == {"n": 1, "mean_none_probability": 0.2, "none_selection_rate": 0.0, "accuracy": 1.0}
    assert report["none_absent"] == {"n": 1, "mean_none_probability": 0.9, "none_selection_rate": 1.0, "accuracy": 1.0}


def test_clustered_bootstrap_is_deterministic_and_identical_pairs_have_zero_delta():
    rows = [
        full_row("1", "shared", [0.8, 0.2], 0, question="a"),
        full_row("1", "shared", [0.3, 0.7], 1, question="b"),
        full_row("2", "other", [0.6, 0.4], 0, source="b"),
    ]
    assert bootstrap_report(rows, samples=30, seed=7) == bootstrap_report(rows, samples=30, seed=7)
    paired = paired_bootstrap(rows, rows, samples=30, seed=7)
    for name in ("accuracy", "nll", "brier", "ece", "macro_source_accuracy"):
        assert paired[name]["estimate"] == pytest.approx(0)
        assert paired[name]["ci95"] == pytest.approx([0, 0])
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(rows, rows[:-1], samples=2)


def test_zero_level_embedding_restores_after_exception():
    model = torch.nn.Module()
    model.level = torch.nn.Embedding(4, 3)
    before = model.level.weight.detach().clone()
    with pytest.raises(RuntimeError):
        with zero_level_embedding(model):
            assert torch.count_nonzero(model.level.weight) == 0
            raise RuntimeError("stop")
    assert torch.equal(model.level.weight, before)


def test_level_zero_report_keeps_packing_identical_and_filters_score_rows():
    class Predictor:
        def __init__(self):
            self.metadata = {"readout": {"head_kind": "pointer"}}
            self.model = torch.nn.Module()
            self.model.level = torch.nn.Embedding(3, 2)
            with torch.no_grad():
                self.model.level.weight.fill_(1)

        def __call__(self, item):
            active = bool(torch.count_nonzero(self.model.level.weight))
            values = [([0.8, 0.2] if active else [0.5, 0.5]) if question["type"] == "score" else [0.25, 0.75]
                      for question in item["questions"].values()]
            return values, 0.0

    item = {
        "state": "state",
        "questions": {
            "score": {"type": "score", "instructions": "rate", "criteria": ["low", "high"], "label": 0, "src": "score"},
            "noul": {"type": "noul", "instructions": "yes?", "label": True, "src": "noul"},
        },
        "_meta": {"id": "mixed", "group_id": "mixed", "source": "fixture", "variant": "clean"},
    }
    predictor = Predictor()
    probabilities, _ = predictor(item)
    learned = rows_for_record(item, probabilities)
    zeroed, report = level_zero_report([item], predictor, learned, 1.0, 10)
    assert len(zeroed) == 1 and zeroed[0]["type"] == "score"
    assert report["max_abs_probability_change"] == pytest.approx(0.3)
    assert report["max_non_score_probability_change"] == 0
    assert report["used"] and report["useful"]
    assert torch.count_nonzero(predictor.model.level.weight) > 0


def test_transfer_suite_requires_eval_only_and_matching_revision():
    root = Path(__file__).resolve().parents[1]
    base = "Qwen/Qwen3-0.6B-Base"
    revision = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    records = validate_transfer_suite(root / "evals/decision-v2", root / "evals/transfer-v2", base, revision)
    assert len(records) == 668
    with pytest.raises(ValueError, match="different base revision"):
        validate_transfer_suite(root / "evals/decision-v2", root / "evals/transfer-v2", base, "wrong")
