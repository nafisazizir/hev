"""Offline regression tests for Hev training objectives and safeguards."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from hev.data import EVAL_ONLY, source_policy
from hev.model import DecisionModel
from hev.suite import object_digest, source_hashes
from hev.train import (KEV_V4_RECIPE, accumulation_records, make_scheduler, objective_probe, prepare_output,
                       question_loss, reference_recipe_report, runtime_provenance, validate_args,
                       validate_training_records)

ROOT = Path(__file__).resolve().parents[1]


def args(**overrides):
    values = {
        "epochs": 1,
        "lora": 16,
        "batch": 1,
        "accum": 8,
        "lr": 2e-4,
        "ord_w": 0.0,
        "p_none": 0.1,
        "p_none_distract": 0.12,
        "p_distract": 0.15,
        "p_none_pair": 0.0,
        "objective_records": 0,
        "base": "Qwen/Qwen3-0.6B-Base",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_choice_loss_matches_cross_entropy():
    logits = torch.tensor([0.2, -0.1, 0.7])
    question = {"label": 2, "qtype": "choice"}
    expected = F.cross_entropy(logits[None], torch.tensor([2]))
    assert torch.equal(question_loss(logits, question, "cpu", ord_w=4.0), expected)


def test_score_ranked_probability_loss_is_proper():
    logits = torch.tensor([0.2, 0.8]).log().requires_grad_()
    expected = sum(
        probability * question_loss(logits, {"label": label, "qtype": "score"}, "cpu", ord_w=0.5)
        for label, probability in enumerate([0.2, 0.8])
    )
    expected.backward()
    assert logits.grad.abs().max().item() < 1e-6


def test_uneven_accumulation_groups_weight_records_equally():
    values = torch.arange(10, dtype=torch.float32)
    gradients = []
    for batch, accum in ((8, 1), (3, 3), (2, 4)):
        weight = torch.tensor(1.0, requires_grad=True)
        for microbatch, start in enumerate(range(0, len(values), batch)):
            chunk = values[start:start + batch]
            ((weight * chunk).sum() / accumulation_records(len(values), batch, accum, microbatch)).backward()
        gradients.append(weight.grad.item())
    assert gradients[0] == gradients[2]
    assert accumulation_records(10, 3, 3, 2) == 9
    assert accumulation_records(10, 3, 3, 3) == 1


def test_ten_step_smoke_scheduler_is_well_defined():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    scheduler = make_scheduler(optimizer, 2e-4, 10)
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert scheduler.last_epoch == 10


def test_training_source_policy():
    validate_training_records([{"_meta": {"source": "boolq"}}])
    with pytest.raises(ValueError, match="empty"):
        validate_training_records([])
    for source in EVAL_ONLY:
        with pytest.raises(ValueError, match="eval-only"):
            validate_training_records([{"_meta": {"source": source}}])
    disguised = {"_meta": {"source": "boolq"}, "questions": {"q": {"src": EVAL_ONLY[0]}}}
    with pytest.raises(ValueError, match="eval-only"):
        validate_training_records([disguised])
    with pytest.raises(ValueError, match="does not declare trainable"):
        validate_training_records([{"_meta": {"source": "legacy_policy"}}])
    # a question's src is a task name inside its record's source, not a source of its own
    validate_training_records([{"_meta": {"source": "agnews"}, "questions": {"q": {"src": "agnews_yn"}}}])


def test_suite_manifest_governs_which_sources_may_train():
    v4 = json.loads((ROOT / "evals" / "decision-v4" / "manifest.json").read_text())
    trainable, eval_only = source_policy(v4)
    assert {"legacy_policy", "compositional"} <= set(trainable)
    assert "contrastive" in eval_only and "contrastive" not in trainable
    policy_records = [{"_meta": {"source": "legacy_policy"}, "questions": {"q": {"src": "contrastive_return_window"}}},
                      {"_meta": {"source": "compositional"}, "questions": {"q": {"src": "composition_nested_or"}}}]
    validate_training_records(policy_records, trainable, eval_only)
    with pytest.raises(ValueError, match="eval-only"):
        validate_training_records([{"_meta": {"source": "mmlu"}}], trainable, eval_only)

    # a transfer suite declares an empty trainable list; that means nothing may train, not "fall back to defaults"
    transfer = json.loads((ROOT / "evals" / "transfer-v4" / "manifest.json").read_text())
    assert source_policy(transfer)[0] == ()
    # a suite that declares no policy at all keeps the inherited defaults
    smoke = json.loads((ROOT / "evals" / "smoke-v1" / "manifest.json").read_text())
    assert "banking77" in source_policy(smoke)[0]
    with pytest.raises(ValueError, match="both trainable and eval-only"):
        source_policy({"trainable_sources": ["boolq"], "eval_only_sources": ["boolq"]})


def test_minimal_pair_siblings_do_not_inflate_a_records_share_of_the_gradient():
    """A record that emits none-pair siblings must not get three times the gradient weight of one that does not."""
    weights = []
    for variants_per_record in (1, 3):
        weight = torch.tensor(1.0, requires_grad=True)
        for microbatch in range(8):                      # one accumulation group: 8 records, batch 1, accum 8
            loss_sum = sum(weight * 2.0 for _ in range(variants_per_record))
            group = accumulation_records(8, 1, 8, microbatch) * variants_per_record
            (loss_sum / group).backward()
        weights.append(weight.grad.item())
    assert weights[0] == pytest.approx(weights[1])


def test_kev_v4_recipe_guard_matches_knobs_and_names_the_deviations():
    suite_hash = KEV_V4_RECIPE["suite_sha256"]
    matched = args(epochs=2, lr=2e-4, lora=16, batch=1, accum=8, ord_w=0.0, p_none_pair=0.25)
    report = reference_recipe_report(matched, suite_hash, "mps")
    assert report["matched"]["effective_batch"] == 8 and report["matched"]["p_none_pair"] == 0.25
    deviations = {item["knob"]: item for item in report["accepted_deviations"]}
    assert set(deviations) == {"precision", "device", "microbatching", "encoding"}
    assert deviations["precision"]["kev"] == "bf16 autocast" and deviations["precision"]["hev"] == "fp32"
    assert deviations["device"]["hev"] == "mps"
    with pytest.raises(ValueError, match="match kev's published trial"):
        reference_recipe_report(args(epochs=2, lr=2e-4, p_none_pair=0.0), suite_hash, "mps")
    with pytest.raises(ValueError, match="requires the decision-v4 suite"):
        reference_recipe_report(matched, "0" * 64, "mps")


def test_objective_probe_is_deterministic_and_independent_of_order():
    records = [{"_meta": {"id": f"r{i}"}} for i in range(50)]
    probe = objective_probe(records, 8)
    assert len(probe) == 8
    assert probe == objective_probe(list(reversed(records)), 8)
    assert objective_probe(records, 0) == records and objective_probe(records, 100) == records


@pytest.mark.parametrize("bad", [
    {"p_none_pair": 1.5},
    {"p_none_pair": -0.1},
    {"objective_records": -1},
    {"epochs": 0},
    {"lora": 0},
    {"batch": 0},
    {"accum": 0},
    {"lr": 0},
    {"ord_w": -1},
    {"p_none": -0.1},
    {"p_none": 0.8, "p_none_distract": 0.2, "p_distract": 0.1},
])
def test_argument_validation_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        validate_args(args(**bad))


def test_existing_output_is_untouched(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep")
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_output(output)
    assert marker.read_text() == "keep"


def test_readout_checkpoint_includes_level_embedding(tiny_backbone):
    model = DecisionModel(backbone=tiny_backbone, head="set")
    with torch.no_grad():
        model.level.weight[3].fill_(2.0)
    state = model.readout_state_dict()
    assert state["head_kind"] == "set"
    assert torch.equal(state["level"]["weight"][3], torch.full_like(state["level"]["weight"][3], 2.0))
    other = DecisionModel(backbone=tiny_backbone, head="pointer")
    with pytest.raises(ValueError, match="checkpoint head"):
        other.load_readout_state_dict(state)


def test_provenance_hashes_are_deterministic_and_exclude_runs():
    assert object_digest({"b": 2, "a": 1}) == object_digest({"a": 1, "b": 2})
    first, second = source_hashes(), source_hashes()
    assert first == second and "hev/train.py" in first and "uv.lock" in first
    assert not any(path.startswith("runs/") for path in first)


def test_runtime_provenance_records_code_environment_and_device():
    provenance = runtime_provenance("cpu")
    assert provenance["device"] == "cpu" and provenance["dtype"] == "fp32"
    assert provenance["git"]["commit"] and isinstance(provenance["source_hashes"], dict)
    assert all(provenance["packages"].values())
