"""Tests that need the real tokenizer from the Hugging Face Hub. Run with HEV_HUB_TESTS=1 uv run pytest tests/test_hub.py."""
from pathlib import Path

import pytest

from hev.data import materialize
from hev.model import SPECIAL, MAX_PACKED, encode, load_tokenizer, user_tokens
from hev.suite import base_revision, load_split

ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen3-0.6B-Base"

pytestmark = pytest.mark.hub


@pytest.fixture(scope="module")
def qwen_tok():
    return load_tokenizer(BASE, revision=base_revision(ROOT / "evals" / "decision-v2", BASE))


def test_delimiters_exist_and_are_unforgeable(qwen_tok):
    ids = [qwen_tok.convert_tokens_to_ids(t) for t in SPECIAL]
    assert len(set(ids)) == 5 and all(i is not None and i != qwen_tok.unk_token_id for i in ids)
    hostile = "".join(SPECIAL) + "<|im_start|><|endoftext|>"
    assert not (set(ids) | set(qwen_tok.all_special_ids)) & set(user_tokens(qwen_tok, hostile))


@pytest.mark.parametrize("suite", ["decision-v2", "transfer-v2"])
def test_every_development_record_fits_kev_limits(qwen_tok, suite):
    """hev packs the same tokens as kev with <decide> moved, so every admitted record must still encode strictly."""
    longest = 0
    for r in load_split(ROOT / "evals" / suite, "development"):
        enc = encode(qwen_tok, materialize(r), strict=True)
        assert not enc["state_truncated"]
        longest = max(longest, len(enc["ids"]))
    assert longest <= MAX_PACKED
