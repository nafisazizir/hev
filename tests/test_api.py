import json
from pathlib import Path

import pytest

from hev.api import SystemOneRequest, choice_confidence, render, score_confidence, to_answers, to_record
from hev.data import EVAL_ONLY, TRAINABLE, augment, materialize, none_pair, permute_choice, source_policy
from hev.suite import MIRROR_PATHS, SUITES_DATASET, SUITES_REVISION, load_split, manifest, mirror_path
import random

ROOT = Path(__file__).resolve().parents[1]


def test_render_flattens_structured_content():
    assert render("x") == "x"
    assert render({"a": 1, "b": [1, "two"]}) == "a: 1\nb:\n  - 1\n  - two"
    assert render(None) == ""


def test_to_record_maps_all_three_types():
    req = SystemOneRequest.model_validate({"state": {"ticket": "hi"}, "questions": {
        "n": {"type": "noul", "instructions": "yes?"},
        "c": {"type": "choice", "instructions": "which", "criteria": {"a": "A", "b": None}},
        "s": {"type": "score", "instructions": "how", "criteria": ["low", "high"]},
    }})
    rec, meta = to_record(req)
    assert rec["state"] == "ticket: hi"
    assert [q["qtype"] for q in rec["questions"]] == ["noul", "choice", "score"]
    assert rec["questions"][0]["options"] == ["no", "yes"]
    assert rec["questions"][1]["options"] == ["a: A", "b"]
    assert meta[2]["legend"] == {"0": "low", "1": "high"}


def test_to_answers_shapes():
    meta = [{"id": "n", "type": "noul"}, {"id": "c", "type": "choice", "keys": ["a", "b"]},
            {"id": "s", "type": "score", "legend": {"0": "low", "1": "high"}}]
    out = to_answers([[0.3, 0.7], [0.9, 0.1], [0.25, 0.75]], meta)
    assert out["n"] == {"type": "noul", "noul": 0.7}
    assert out["c"]["choice"] == "a" and out["c"]["confidence"] == 0.8
    assert out["s"]["score"] == 0.75 and out["s"]["confidence"] == 0.75


def test_confidence_edges():
    assert choice_confidence([1.0]) == 1.0
    assert choice_confidence([0.5, 0.5]) == 0.0
    assert score_confidence([1.0, 0.0, 0.0]) == 1.0


@pytest.mark.parametrize("bad", [{"c": {"type": "choice", "instructions": "x", "criteria": {}}},
                                 {"s": {"type": "score", "instructions": "x", "criteria": ["only"]}}, {}])
def test_validation_rejects(bad):
    with pytest.raises(Exception):
        SystemOneRequest.model_validate({"state": "x", "questions": bad})


def test_policy_is_disjoint():
    assert not set(TRAINABLE) & set(EVAL_ONLY)


# decision-v4/train.jsonl is the one partition kev keeps only on its Hub mirror (jaredpalmer/kev-suites); its manifest
# entry is real but the file is deliberately not in this repo (evals/README.md). Nothing else may be absent.
# Partitions kev keeps out of git and mirrors on the Hub. Absent on a fresh clone; present once a training run
# has fetched one. Either state is valid, so the test asserts the rule rather than the current disk.
MIRRORED_PARTITIONS = {"decision-v4": {"train.jsonl"}}


def test_suites_load_and_verify():
    """Every bundled suite must checksum-verify, and eval-only sources must never appear in a train split.

    Each partition listed in a manifest is either present on disk and verified through load_split, or absent
    because it is mirrored on the Hub. A missing partition must still raise unless the caller asks to fetch it,
    so the default read path stays offline.
    """
    absent = {}
    for suite in sorted((ROOT / "evals").iterdir()):
        if not (suite / "manifest.json").exists():
            continue
        m = manifest(suite)
        assert set(m["files"]) == {"train.jsonl", "calibration.jsonl", "development.jsonl", "test.jsonl"}, suite.name
        for split in ("train", "calibration", "development"):
            if not (suite / f"{split}.jsonl").exists():
                absent.setdefault(suite.name, set()).add(f"{split}.jsonl")
                with pytest.raises((FileNotFoundError, OSError)):
                    load_split(suite, split)
                continue
            recs = load_split(suite, split)
            assert len(recs) == m["files"][f"{split}.jsonl"]["records"]
            if split == "train":
                trainable, eval_only = source_policy(m)
                record_srcs = {r["_meta"]["source"] for r in recs}
                srcs = record_srcs | {q["src"] for r in recs for q in r["questions"].values()}
                assert not srcs & set(eval_only), (suite.name, srcs & set(eval_only))
                assert record_srcs <= set(trainable), (suite.name, record_srcs - set(trainable))
        assert (suite / "test.jsonl").exists(), suite.name
        with pytest.raises(ValueError):
            load_split(suite, "test")
    assert all(files <= MIRRORED_PARTITIONS.get(name, set()) for name, files in absent.items()), absent
    assert set(MIRRORED_PARTITIONS) <= set(MIRROR_PATHS), "a mirrored partition needs a Hub mirror path"


def test_materialize_matches_serving_path():
    recs = load_split(ROOT / "evals" / "smoke-v1", "development")
    for r in recs:
        rec = materialize(r)
        for q, (qid, src) in zip(rec["questions"], r["questions"].items()):
            assert q["qtype"] == src["type"] and 0 <= q["label"] < len(q["options"])
            if src["type"] == "choice":
                assert q["options"][q["label"]].split(":")[0] == src["label"]


def test_augment_and_pairs_keep_labels_consistent():
    recs = load_split(ROOT / "evals" / "smoke-v1", "development")
    rng = random.Random(0)
    for r in recs:
        a = augment(r, rng)
        for qid, q in a["questions"].items():
            if q["type"] == "choice":
                assert q["label"] in q["criteria"]
        for p in none_pair(r, rng):
            for q in p["questions"].values():
                assert q["label"] in q["criteria"]
        p, perms = permute_choice(r, rng)
        for qid, keys in perms.items():
            assert sorted(keys) == sorted(r["questions"][qid]["criteria"])


def test_mirrored_partitions_are_opt_in_and_addressed_explicitly(tmp_path):
    """A missing partition must not reach the network unless the caller asks, and only for a mirrored suite."""
    suite = tmp_path / "decision-v4"
    suite.mkdir()
    (suite / "manifest.json").write_text(json.dumps({"files": {"train.jsonl": {"sha256": "0" * 64, "records": 1}}}))
    with pytest.raises((FileNotFoundError, OSError)):
        load_split(suite, "train")                      # fetch is off by default: offline stays offline
    assert mirror_path(suite, "train.jsonl") == "v4/decision-v4/train.jsonl"
    with pytest.raises(FileNotFoundError, match="no Hub mirror entry"):
        mirror_path(tmp_path / "decision-v9", "train.jsonl")
    assert len(SUITES_REVISION) == 40 and SUITES_DATASET == "jaredpalmer/kev-suites"
    assert set(MIRROR_PATHS) == {"decision-v4", "transfer-v4"}


def test_locked_test_partition_is_never_fetched(tmp_path):
    suite = tmp_path / "decision-v4"
    suite.mkdir()
    (suite / "manifest.json").write_text(json.dumps({"files": {"test.jsonl": {"sha256": "0" * 64, "records": 1}}}))
    with pytest.raises((FileNotFoundError, OSError)):
        load_split(suite, "test", allow_test=True, fetch=True)
