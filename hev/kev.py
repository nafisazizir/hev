"""KevPredictor: score a released kev checkpoint (e.g. hf://jaredpalmer/kev-0.6b) through Hev's evaluator.

Loads exactly as kev/evaluate.py:load does (fp32, LoRA adapter via PEFT, PointerHead from head.pt) but using the
vendored hev.kev_model so records are packed the way the checkpoint was trained. Exposes the same duck-typed
contract as hev.evaluate.LocalPredictor: __call__(request) -> (list of numpy probability vectors, seconds), plus
.run/.device/.tokenizer/.model/.metadata. The model deliberately has no `.level` attribute, so Hev's level-zero
ablation cannot be applied to it.
"""
import json
import time
from pathlib import Path

import torch

from . import kev_model
from .checkpoint import resolve_run
from .data import materialize
from .evaluate import sync
from .kev_model import KEV_COMMIT, KEV_MODEL_SHA256, SPECIAL, DecisionModel
from .model import load_tokenizer
from .suite import digest

MAX_PACKED = 2048
HASHED_FILES = ("adapter_config.json", "adapter_model.safetensors", "head.pt", "provenance.json",
                "result.json", "training_config.json")


def load_adapter(lm, run, device):
    """Attach the checkpoint's LoRA adapter, as kev does. Module-level so tests can substitute it."""
    from peft import PeftModel
    return PeftModel.from_pretrained(lm, str(run)).to(device)


def special_ids(tok):
    """The five delimiter ids; refuses tokenizers that lack them or map two delimiters to one id."""
    ids = [tok.convert_tokens_to_ids(t) for t in SPECIAL]
    unk = getattr(tok, "unk_token_id", None)
    if any(i is None or i < 0 or i == unk for i in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"tokenizer does not carry kev's five delimiters: {ids}")
    return ids


def configure_backends(device):
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)


def read_head(run):
    meta = torch.load(run / "head.pt", map_location="cpu", weights_only=True)
    for key in ("head", "base", "suite_sha256"):
        if key not in meta:
            raise ValueError(f"head.pt lacks {key!r}")
    if meta.get("special_embeddings", False):
        raise ValueError("kev checkpoints with trainable delimiter embeddings (special_embeddings=True) are not supported")
    head_dim = int(meta.get("head_dim", 256))
    shapes = {name: tuple(meta["head"][name].shape) for name in ("q.weight", "q.bias", "k.weight", "k.bias")}
    if shapes["q.weight"][0] != head_dim or shapes["k.weight"][0] != head_dim or shapes["q.weight"] != shapes["k.weight"]:
        raise ValueError(f"head.pt head_dim {head_dim} disagrees with PointerHead weights {shapes}")
    adapter = json.loads((run / "adapter_config.json").read_text())
    if adapter.get("trainable_token_indices"):
        raise ValueError("adapter carries trainable token embeddings; unsupported")
    if adapter.get("base_model_name_or_path") not in (None, meta["base"]):
        raise ValueError("adapter base model differs from head.pt base")
    return meta, head_dim


class KevPredictor:
    def __init__(self, reference, device):
        run = Path(resolve_run(reference))
        configure_backends(device)
        meta, head_dim = read_head(run)
        base, revision = meta["base"], meta.get("base_revision")
        tokenizer = load_tokenizer(base, revision=revision)
        ids = special_ids(tokenizer)
        if (run / "tokenizer.json").exists() and special_ids(load_tokenizer(str(run))) != ids:
            raise ValueError("delimiter ids of the checkpoint's bundled tokenizer differ from the base tokenizer's")
        option_isolation = bool(meta.get("option_isolation", False))
        model = DecisionModel(base, tokenizer, device, lora=None, revision=revision, head_dim=head_dim,
                              option_isolation=option_isolation, dtype=torch.float32)
        hidden = model.lm.config.hidden_size
        if meta["head"]["q.weight"].shape[1] != hidden:
            raise ValueError(f"PointerHead expects hidden size {meta['head']['q.weight'].shape[1]}, backbone has {hidden}")
        if max(ids) >= model.lm.config.vocab_size:
            raise ValueError("delimiter id outside the backbone vocabulary")
        model.lm = load_adapter(model.lm, run, device)
        model.head.load_state_dict(meta["head"])
        model.eval()
        self.run, self.device, self.tokenizer, self.model = run, device, tokenizer, model
        self.metadata = {
            "base": base,
            "base_revision": revision,
            "suite_sha256": meta["suite_sha256"],
            "readout": {"head_kind": "kev-pointer"},
            "head_dim": head_dim,
            "option_isolation": option_isolation,
            "special_embeddings": bool(meta.get("special_embeddings", False)),
            "lora": meta.get("lora"),
            "training_args": meta.get("args"),
            "kev_source": {"commit": KEV_COMMIT, "model_py_sha256": KEV_MODEL_SHA256},
            "checkpoint": {
                "reference": str(reference),
                "resolved_path": str(run),
                "hub_commit": run.name if run.parent.name == "snapshots" else None,
                "file_hashes": {name: digest(run / name) for name in HASHED_FILES if (run / name).exists()},
            },
        }

    def __call__(self, request):
        packed = self.model.encode(self.tokenizer, materialize(request), strict=True)
        if len(packed["ids"]) > MAX_PACKED:
            raise ValueError(f"packed request exceeds frozen {MAX_PACKED}-token limit: {len(packed['ids'])}")
        sync(self.device)
        started = time.perf_counter()
        probabilities = [values.numpy() for values in self.model.probs(packed)]
        sync(self.device)
        return probabilities, time.perf_counter() - started
