"""Offline tests for the predeclared M2 comparison decisions."""
import pytest

from hev.compare import M2_RECIPE, ablation_decision, d6_decision, h1_decision, h2_decision, h3_decision, load_hev_run
from hev.suite import digest, write_json


def suite_report(accuracy=0.8, spread=1e-6, flips=0.0, packed=1e-6, ece=0.03, brier=0.27, banking=0.72):
    return {
        "raw": {"overall": {"accuracy": accuracy}, "by_task": {"banking77": {"accuracy": banking}}},
        "calibrated": {"overall": {"ece": ece, "brier": brier}},
        "permutation": {"argmax_flip_rate": flips, "p90_correct_probability_spread": spread},
        "packed_vs_separate": {"max_abs_probability_difference": packed},
    }


def model_result(decision_accuracy=0.8, transfer_accuracy=0.6, **decision_overrides):
    return {
        "decision": suite_report(accuracy=decision_accuracy, **decision_overrides),
        "transfer": suite_report(accuracy=transfer_accuracy, packed=None),
        "level_zero_ablation": {
            "used": True,
            "useful": True,
            "max_abs_probability_change": 0.2,
            "paired_learned_minus_zeroed": {"nll": {"estimate": -0.1, "ci95": [-0.2, -0.01]}},
        },
    }


def kev_result():
    return {"decision": {"clean": {"acc": 0.815833}}, "transfer": {"clean": {"acc": 0.619643}}}


def test_h1_requires_every_mechanism_and_accuracy_boundary():
    assert h1_decision(model_result(0.765833, 0.569643), kev_result())["outcome"] == "supported"
    assert h1_decision(model_result(0.765832, 0.569643), kev_result())["outcome"] == "contradicted"
    changed = model_result(0.8, 0.6)
    changed["transfer"]["permutation"]["argmax_flip_rate"] = 0.01
    assert h1_decision(changed, kev_result())["outcome"] == "contradicted"


def test_h2_supported_contradicted_and_inconclusive_states():
    assert h2_decision(0.81, 0.82, [0.001, 0.02], 0.815833)["outcome"] == "inconclusive"
    assert h2_decision(0.75, 0.79, [0.01, 0.07], 0.815833)["outcome"] == "supported"
    assert h2_decision(0.75, 0.76, [-0.01, 0.03], 0.815833)["outcome"] == "contradicted"
    assert h2_decision(0.75, 0.79, [-0.01, 0.07], 0.815833)["outcome"] == "inconclusive"


def test_h3_uses_predeclared_noninferiority_boundaries():
    assert h3_decision(model_result(ece=0.0464021, brier=0.2794607))["outcome"] == "supported"
    assert h3_decision(model_result(ece=0.0464022, brier=0.2794607))["outcome"] == "contradicted"


def test_level_ablation_outcomes_are_not_reinterpreted():
    assert ablation_decision(model_result())["outcome"] == "supported"
    result = model_result()
    result["level_zero_ablation"].update(useful=False)
    assert ablation_decision(result)["outcome"] == "inconclusive"
    result["level_zero_ablation"].update(used=False)
    assert ablation_decision(result)["outcome"] == "inconclusive"
    with pytest.raises(ValueError, match="missing"):
        ablation_decision({})


def test_d6_requires_both_heads_below_the_boundary():
    assert d6_decision(model_result(banking=0.71), model_result(banking=0.70))["triggered"]
    assert not d6_decision(model_result(banking=0.7125), model_result(banking=0.70))["triggered"]


def test_load_hev_run_verifies_recipe_row_hashes_and_counts(tmp_path):
    run = tmp_path / "pointer"
    run.mkdir()
    for name in ("decision", "transfer"):
        write_json(run / f"{name}_rows.json", [{"name": name}])
    artifacts = {
        name: {"path": f"{name}_rows.json", "sha256": digest(run / f"{name}_rows.json"), "rows": 1}
        for name in ("decision", "transfer")
    }
    write_json(run / "result.json", {
        "status": "success",
        "test_evaluated": False,
        "head": "pointer",
        "training_config": {"args": {**M2_RECIPE, "head": "pointer"}},
        "training_metrics": {"status": "success", "records_seen": 6864, "requested_records": 6864, "optimizer_steps": 858},
        "artifact_hashes": artifacts,
    })
    assert load_hev_run(run, "pointer")["rows"]["decision"] == [{"name": "decision"}]
    (run / "decision_rows.json").write_text("[]\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_hev_run(run, "pointer")
