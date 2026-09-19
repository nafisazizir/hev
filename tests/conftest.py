"""Offline fixtures: a character-level fake tokenizer and a tiny randomly initialised Qwen2 backbone.

Nothing here touches the network, so `uv run pytest` works on a fresh clone. Tests that need the real
Qwen tokenizer are marked `hub` and skipped unless HEV_HUB_TESTS=1.
"""
import os

import pytest
import torch

from hev.model import SPECIAL


class FakeTokenizer:
    """Maps each character to an id; the five delimiter strings map to reserved ids 1..5."""
    vocab_size = 512
    pad_token_id = 0

    def __init__(self):
        self.special = {t: i + 1 for i, t in enumerate(SPECIAL)}

    class _Out:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, add_special_tokens=False):
        return self._Out([10 + (ord(c) % 500) for c in text])

    def convert_tokens_to_ids(self, t):
        return self.special[t]

    @property
    def all_special_ids(self):
        return list(self.special.values())


@pytest.fixture(scope="session")
def tok():
    return FakeTokenizer()


@pytest.fixture(scope="session")
def tiny_backbone():
    from transformers import Qwen2Config, Qwen2Model
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      attn_implementation="eager")
    return Qwen2Model(cfg).float()


def pytest_configure(config):
    config.addinivalue_line("markers", "hub: needs the real tokenizer/model from the Hugging Face Hub")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("HEV_HUB_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set HEV_HUB_TESTS=1 to run Hub tests")
    for item in items:
        if "hub" in item.keywords:
            item.add_marker(skip)
