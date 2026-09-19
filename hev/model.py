"""hev decision model: causal LM backbone + option-isolating block mask + permutation-equivariant readout.

The design is described in docs/DESIGN.md. In one sentence: kev isolates *questions* from each other
under a block-causal mask; hev additionally isolates every *option* from its sibling options, so the
hidden state of an option is, by construction, independent of where in the list that option appears.
Order can then only enter through the readout head, and the heads here are permutation-equivariant.

Packing (one sequence, no decoding):

    <state> s1 s2 ... sS                                        segment 0
    <q> i1 i2 ... <decide>                                      question k, sub 0
    <opt> o1 o2 ... </opt>   (option 1 of question k)           question k, sub 1
    <opt> o1 o2 ... </opt>   (option 2 of question k)           question k, sub 2
    ...
    <q> ... <decide>  <opt> ... </opt> ...                      question k+1

Attention rule for query i and key j (packed order, j <= i):
    seg[j] == 0                                           any token may read the state
    or (seg[j] == seg[i] and sub[j] == 0)                 an option may read its own question's instruction
    or (seg[j] == seg[i] and sub[j] == sub[i])            a token may read its own branch

Position ids restart after the state for every question, and every option of a question starts at the
same position (right after that question's <decide>). Hence h(option j) is a function of
(state, instruction, option-j text) only, never of sibling options or list position.
"""
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

# Rarely used Qwen special tokens reused as delimiters: <state>, <q>, <opt>, </opt>, <decide>.
# No embedding rows are added; LoRA adapts their meaning. Same choice as kev so that packed token
# counts, and hence the frozen suites' context limits, are identical between the two models.
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048
QTYPES = ("noul", "choice", "score")

_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def load_tokenizer(name, revision=None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, revision=revision)


def user_tokens(tok, text):
    """Tokenize caller-supplied text so it can never produce delimiter/control tokens (boundaries are unforgeable).
    Fast tokenizers ignore split_special_tokens, so `<|name|>` is rewritten to `<¦name¦>` before tokenizing."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False):
    """Pack one internal record. Returns a dict with

        ids         token ids
        seg         0 = state, k = question k (1-based)
        sub         0 = instruction/decide part, j = option j (1-based); 0 for state tokens
        pos         position ids (restart after the state; all options of a question share a start)
        decide_idx  [Q] index of each question's <decide> token
        opt_idx     [Q][K] index of each option's </opt> token
        qtypes      [Q] "noul" | "choice" | "score"
        labels      [Q] int
    """
    state_tokens = user_tokens(tok, rec["state"])
    if strict and len(state_tokens) + 1 > max_state:
        raise ValueError(f"state exceeds {max_state} tokens: {len(state_tokens) + 1}")
    st_id, q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL)
    S = [st_id] + state_tokens[: max_state - 1]
    ids, seg, sub, pos = list(S), [0] * len(S), [0] * len(S), list(range(len(S)))
    decide_idx, opt_idx, qtypes, labels = [], [], [], []
    for k, q in enumerate(rec["questions"], start=1):
        if q.get("qtype", "choice") not in QTYPES:
            raise ValueError(f"unknown question type: {q.get('qtype')}")
        head = [q_id] + user_tokens(tok, q["instr"]) + [d_id]
        branch_len = len(head)
        base = len(ids)
        ids += head; seg += [k] * len(head); sub += [0] * len(head)
        pos += list(range(len(S), len(S) + len(head)))
        decide_idx.append(base + len(head) - 1)
        opt_start = len(S) + len(head)  # every option of this question starts here
        oi = []
        for j, o in enumerate(q["options"], start=1):
            opt = [o_id] + user_tokens(tok, o) + [c_id]
            branch_len += len(opt)
            ids += opt; seg += [k] * len(opt); sub += [j] * len(opt)
            pos += list(range(opt_start, opt_start + len(opt)))
            oi.append(len(ids) - 1)
        if branch_len > max_branch - len(S):
            raise ValueError(f"branch too long: {branch_len}")
        opt_idx.append(oi); qtypes.append(q.get("qtype", "choice")); labels.append(q.get("label", 0))
    if strict and len(ids) > MAX_PACKED:
        raise ValueError(f"packed request exceeds {MAX_PACKED} tokens: {len(ids)}")
    return {"ids": ids, "seg": seg, "sub": sub, "pos": pos, "decide_idx": decide_idx, "opt_idx": opt_idx,
            "qtypes": qtypes, "labels": labels, "state_truncated": len(state_tokens) + 1 > max_state}


def branch_mask_batch(segs, subs, device, dtype=torch.float32):
    """Batched additive mask [B, 1, L, L], right-padded to the longest sequence.

    allow(i, j) iff j <= i and key j is real and (seg[j] == 0 or (seg[j] == seg[i] and (sub[j] == 0 or sub[j] == sub[i]))).
    Padded query rows keep the diagonal so no row is fully masked; finfo.min rather than -inf keeps softmax finite."""
    L = max(len(s) for s in segs)
    B = len(segs)
    s = torch.full((B, L), -1, device=device)
    u = torch.full((B, L), -1, device=device)
    for b, (seg, sub) in enumerate(zip(segs, subs)):
        s[b, : len(seg)] = torch.tensor(seg, device=device)
        u[b, : len(sub)] = torch.tensor(sub, device=device)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))[None]
    key_state = (s == 0)[:, None, :]
    same_q = s[:, None, :] == s[:, :, None]
    key_instr = (u == 0)[:, None, :]
    same_opt = u[:, None, :] == u[:, :, None]
    valid_key = (s != -1)[:, None, :]
    allow = causal & valid_key & (key_state | (same_q & (key_instr | same_opt)))
    allow = allow | torch.eye(L, dtype=torch.bool, device=device)[None]
    return torch.zeros(B, L, L, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]


def branch_mask(seg, sub, device, dtype=torch.float32):
    return branch_mask_batch([seg], [sub], device, dtype)


class PointerHead(nn.Module):
    """Independent scoring: logit_j = <W_q h_decide, W_k h_opt_j> / sqrt(dp). Exactly permutation-invariant."""

    def __init__(self, d, dp=256):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)

    def forward(self, h_decide, h_opts):  # [d], [K, d] -> [K]
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale


class SetHead(nn.Module):
    """Listwise scoring: a small transformer with no positional encoding over {decide, opt_1..opt_K}, then a pointer
    score. Options can interact (the effect archerhume measured in Jev) while remaining permutation-equivariant."""

    def __init__(self, d, dp=256, layers=2, heads=4):
        super().__init__()
        self.inp = nn.Linear(d, dp)
        layer = nn.TransformerEncoderLayer(dp, heads, dim_feedforward=4 * dp, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.q, self.k = nn.Linear(dp, dp), nn.Linear(dp, dp)
        self.scale = 1 / math.sqrt(dp)

    def forward(self, h_decide, h_opts):  # [d], [K, d] -> [K]
        x = self.inp(torch.cat([h_decide[None], h_opts], 0))[None]  # [1, 1+K, dp]
        x = self.enc(x)[0]
        return (self.k(x[1:]) @ self.q(x[0])) * self.scale


HEADS = {"pointer": PointerHead, "set": SetHead}


class DecisionModel(nn.Module):
    """Backbone (no vocab head; we never generate) + LoRA + readout head + ordinal level embedding for Score."""

    def __init__(self, name=None, tok=None, device="cpu", lora=None, revision=None, attn=None, head="pointer",
                 backbone=None, max_levels=64):
        super().__init__()
        if backbone is None:
            from transformers import AutoModelForCausalLM
            attn = attn or ("sdpa" if str(device).startswith("cuda") else "eager")
            backbone = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.float32, attn_implementation=attn).model
        self.lm = backbone
        self.pad_id = (tok.pad_token_id if tok is not None and tok.pad_token_id is not None else 0)
        if lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(task_type="FEATURE_EXTRACTION", r=lora, lora_alpha=2 * lora, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            self.lm = get_peft_model(self.lm, cfg)
        d = self.lm.config.hidden_size
        self.head_kind = head
        self.head = HEADS[head](d)
        # Score questions are ordered by definition, so order is information there rather than a nuisance.
        # A learned level embedding is added to option vectors of Score questions only (docs/DESIGN.md).
        self.level = nn.Embedding(max_levels, d)
        nn.init.zeros_(self.level.weight)
        self.device = device
        self.to(device)

    def hidden_batch(self, encs):
        """[B, L_max, d] hidden states for a right-padded batch of encoded records."""
        L = max(len(e["ids"]) for e in encs)
        ids = torch.full((len(encs), L), self.pad_id, device=self.device)
        pos = torch.zeros((len(encs), L), dtype=torch.long, device=self.device)
        for b, e in enumerate(encs):
            ids[b, : len(e["ids"])] = torch.tensor(e["ids"], device=self.device)
            pos[b, : len(e["pos"])] = torch.tensor(e["pos"], device=self.device)
        mask = branch_mask_batch([e["seg"] for e in encs], [e["sub"] for e in encs], self.device)
        return self.lm(input_ids=ids, position_ids=pos, attention_mask=mask).last_hidden_state

    def hidden(self, enc):
        return self.hidden_batch([enc])[0, : len(enc["ids"])]

    def _readout(self, h, enc):
        out = []
        for d, oi, qt in zip(enc["decide_idx"], enc["opt_idx"], enc["qtypes"]):
            h_opts = h[torch.tensor(oi, device=self.device)]
            if qt == "score":
                h_opts = h_opts + self.level(torch.arange(len(oi), device=self.device))
            out.append(self.head(h[d], h_opts))
        return out

    def forward(self, enc):
        """List of logits tensors, one per question."""
        return self._readout(self.hidden(enc), enc)

    def forward_batch(self, encs):
        """List (per record) of lists (per question) of logits, from one padded forward pass."""
        hs = self.hidden_batch(encs)
        return [self._readout(hs[b], e) for b, e in enumerate(encs)]

    @torch.no_grad()
    def probs(self, enc):
        return [F.softmax(z, -1).cpu() for z in self.forward(enc)]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def readout_state_dict(self):
        return {"head": self.head.state_dict(), "level": self.level.state_dict(), "head_kind": self.head_kind}

    def load_readout_state_dict(self, sd):
        if sd.get("head_kind", "pointer") != self.head_kind:
            raise ValueError(f"checkpoint head is {sd.get('head_kind')}, model head is {self.head_kind}")
        self.head.load_state_dict(sd["head"]); self.level.load_state_dict(sd["level"])
