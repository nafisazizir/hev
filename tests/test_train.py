"""Offline regression tests for Hev training objectives and safeguards."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from hev.data import EVAL_ONLY
from hev.model import DecisionModel
from hev.suite import object_digest, source_hashes
from hev.train import accumulation_records, make_scheduler, prepare_output, question_loss, runtime_provenance, validate_args, validate_training_records


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


@pytest.mark.parametrize("bad", [
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
