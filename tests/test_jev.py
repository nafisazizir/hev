"""Offline tests for direct Jev request shaping, probability conversion, and permutation reporting."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hev.jev import DirectJevPredictor, api_request, normalize_returned, permutation_report


def record():
    return {
        "state": {"text": "example"},
        "questions": {
            "yn": {"type": "noul", "instructions": "yes?", "label": True, "src": "yn"},
            "pick": {"type": "choice", "instructions": "pick", "criteria": {"a": None, "b": None}, "label": "b", "src": "pick"},
            "rate": {"type": "score", "instructions": "rate", "criteria": ["low", "high"], "label": 1, "src": "rate"},
        },
        "_meta": {"id": "r", "group_id": "g", "source": "s", "variant": "clean"},
    }


def test_api_request_removes_labels_source_and_metadata():
    request = api_request(record())
    assert request["state"] == {"text": "example"}
    assert set(request) == {"state", "questions"}
    assert request["questions"]["pick"] == {
        "type": "choice", "instructions": "pick", "criteria": {"a": None, "b": None}
    }


def test_normalize_returned_accepts_documented_rounding_and_rejects_bad_distributions():
    probabilities, total, zeros = normalize_returned([0.32, 0.15, 0.52])
    assert probabilities.sum() == pytest.approx(1)
    assert total == pytest.approx(0.99) and zeros == 0
    with pytest.raises(ValueError, match="sum"):
        normalize_returned([0.1, 0.1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        normalize_returned([1.1, -0.1])


def test_direct_predictor_converts_all_question_types(monkeypatch):
    response = SimpleNamespace(
        model="jev-1.2.3",
        usage=SimpleNamespace(input_tokens=123, output_tokens=0),
        answers={
            "yn": SimpleNamespace(noul=0.75),
            "pick": SimpleNamespace(probabilities={"a": 0.2, "b": 0.8}),
            "rate": SimpleNamespace(probabilities={0: 0.1, 1: 0.9}),
        },
    )

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def system_one(self, **kwargs):
            return response

        def close(self):
            pass

    fake = SimpleNamespace(TypeSafeClient=Client, RetryPolicy=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "typesafe_sdk", fake)
    predictor = DirectJevPredictor("secret", budget=0.1, max_calls=10)
    probabilities, metadata = predictor(record())
    assert np.allclose(probabilities[0], [0.25, 0.75])
    assert np.allclose(probabilities[1], [0.2, 0.8])
    assert np.allclose(probabilities[2], [0.1, 0.9])
    assert metadata["model"] == "jev-1.2.3"
    assert metadata["raw_probability_sums"] == pytest.approx([1, 1, 1])
    assert metadata["zero_counts"] == [0, 0, 0]
    assert predictor.calls == 1 and predictor.input_tokens == 123
    assert predictor.model_versions == {"jev-1.2.3"}


def test_direct_predictor_rejects_missing_answer(monkeypatch):
    response = SimpleNamespace(
        model="jev-1.2.3",
        usage=SimpleNamespace(input_tokens=1, output_tokens=0),
        answers={"yn": SimpleNamespace(noul=0.5)},
    )

    class Client:
        def __init__(self, **kwargs):
            pass

        def system_one(self, **kwargs):
            return response

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "typesafe_sdk", SimpleNamespace(TypeSafeClient=Client, RetryPolicy=lambda **kwargs: kwargs))
    predictor = DirectJevPredictor("secret", budget=0.1, max_calls=10)
    with pytest.raises(ValueError, match="answer IDs"):
        predictor(record())


def test_permutation_report_aligns_semantic_keys():
    original = {
        "id": "clean", "group": "g", "question": "q", "source": "s", "task": "t", "type": "choice",
        "variant": "clean", "parent": "clean", "keys": ["a", "b", "c"], "label": 0,
        "probabilities": [0.6, 0.3, 0.1],
    }
    permuted = {
        **original,
        "id": "permuted",
        "variant": "permuted",
        "parent": "clean",
        "keys": ["c", "a", "b"],
        "probabilities": [0.1, 0.55, 0.35],
    }
    report = permutation_report([original, permuted])
    assert report["n"] == 1 and report["orders"] == 2
    assert report["argmax_flip_rate"] == 0
    assert report["max_abs_probability_difference"] == pytest.approx(0.05)
