"""Labelled requests -> internal records, plus training-time augmentation.

A labelled request is the on-disk format of the frozen suites in evals/:
    {"state": JSONContent,
     "questions": {id: {"type", "instructions", "criteria", "label", "src"}},
     "_meta": {...provenance...}}
    label: choice -> option key, noul -> bool, score -> level index.

materialize() runs it through the *serving* path (api.to_record) so training text is byte-identical to
what the API feeds the model. augment() and none_pair() are ported from kev/data.py (Apache-2.0) because
the shortcut they defend against ("this wording => pick it") was found empirically there; see docs/KEV.md.

Dataset download/conversion (kev/data.py build()) is intentionally not ported yet: the frozen suites in
evals/ already contain every record we train and evaluate on. Port it only if a new suite is needed.
"""
import hashlib
import random

from .api import SystemOneRequest, to_record

# Source policy inherited from kev. A source is trainable or eval-only, never both. MMLU is a knowledge
# probe and stays eval-only permanently. Training must refuse eval-only sources. legacy_holdout and
# composition_holdout are the transfer-v4 held-out policy families; they are appended (order is stable).
TRAINABLE = ("banking77", "boolq", "agnews", "mnli", "sst5", "yelp", "trec", "dbpedia14", "amazon", "imdb")
EVAL_ONLY = ("mmlu", "emotion", "tweet_offensive", "qnli", "paws", "sciq", "legacy_holdout", "composition_holdout")

# "None of the above" must appear both as the correct answer and as a wrong alternative, with varied
# wording, or the model learns the wording rather than the evidence.
NONE_OPTIONS = [("other", "None of the above"), ("other", "A reason that fits none of the above"), ("none", "None of these"),
                ("other", "Something else"), ("not_listed", "Not listed here"), ("none_of_the_above", None),
                ("other", "A category that fits none of the above"), ("other", "None of the listed options apply"),
                ("unknown", "Cannot be determined from the options given"), ("other", "Other"), ("none", None),
                ("other", "An answer not covered by the other options"), ("no_match", "No option matches")]
DISTRACTORS = {"weather": "Bad weather caused it", "purple": "The colour purple", "pancakes": "A recipe for pancakes",
               "taxes": "Unrelated: quarterly tax filing"}


def source_seed(seed, source):
    return int.from_bytes(hashlib.sha256(f"{seed}:{source}".encode()).digest()[:8], "big")


def materialize(req):
    """Labelled request -> internal record via the serving path, attaching int labels, src and qtype."""
    clean = {"state": req["state"], "questions": {qid: {k: v for k, v in q.items() if k not in ("label", "src")}
                                                   for qid, q in req["questions"].items()}}
    rec, meta = to_record(SystemOneRequest.model_validate(clean))
    for q, m, (qid, src_q) in zip(rec["questions"], meta, req["questions"].items()):
        y = src_q["label"]
        q["label"] = int(y) if m["type"] == "noul" else m["keys"].index(y) if m["type"] == "choice" else int(y)
        q["src"] = src_q.get("src")
    return rec


def augment(req, rng, p_none=0.1, p_none_distract=0.12, p_distract=0.15):
    """Choice only: permute option order (always); sometimes add a 'none of the above' option, either as the correct
    answer (true option removed) or as a wrong alternative (true option kept); sometimes add an irrelevant distractor.

    Note for Hev: permutation is kept even though the backbone is order-invariant by construction. It still matters
    for the `set` head and as a no-op sanity check for the `pointer` head."""
    if min(p_none, p_none_distract, p_distract) < 0 or p_none + p_none_distract + p_distract > 1:
        raise ValueError("augmentation probabilities must be nonnegative and sum to at most one")
    out = {"state": req["state"], "questions": {}}
    for qid, q in req["questions"].items():
        if q["type"] != "choice":
            out["questions"][qid] = q
            continue
        crit, y = dict(q["criteria"]), q["label"]
        r = rng.random()
        none_options = [(k, v) for k, v in NONE_OPTIONS if k not in crit]
        distractors = [k for k in DISTRACTORS if k not in crit]
        if len(crit) > 2 and r < p_none and none_options:
            nk, nd = rng.choice(none_options); crit.pop(y); crit[nk] = nd; y = nk
        elif p_none <= r < p_none + p_none_distract and len(crit) < 255 and none_options:
            nk, nd = rng.choice(none_options); crit[nk] = nd
        elif p_none + p_none_distract <= r < p_none + p_none_distract + p_distract and len(crit) < 255 and distractors:
            k = rng.choice(distractors); crit[k] = DISTRACTORS[k]
        keys = list(crit); rng.shuffle(keys)
        out["questions"][qid] = {**q, "criteria": {k: crit[k] for k in keys}, "label": y}
    return out


def none_pair(req, rng):
    """Minimal pair for the none-of-the-above shortcut: same state and question twice, once with the true option
    present (none is wrong) and once with it removed (the same none option is right). Returns [] if no eligible Choice."""
    eligible = [(qid, q) for qid, q in req["questions"].items() if q["type"] == "choice" and len(q["criteria"]) >= 3]
    if not eligible:
        return []
    qid, q = rng.choice(eligible)
    nk, nd = rng.choice([o for o in NONE_OPTIONS if o[0] not in q["criteria"]] or [("none_of_these", None)])
    keys = list(q["criteria"]) + [nk]; rng.shuffle(keys)
    present = {**q, "criteria": {k: (nd if k == nk else q["criteria"][k]) for k in keys}}
    absent = {**present, "criteria": {k: v for k, v in present["criteria"].items() if k != q["label"]}, "label": nk}
    return [{"state": req["state"], "questions": {qid: present}}, {"state": req["state"], "questions": {qid: absent}}]


def permute_choice(req, rng):
    """Return a copy with every Choice question's options reshuffled, and per-question perms (new -> old key order)."""
    out, perms = {"state": req["state"], "questions": {}}, {}
    for qid, q in req["questions"].items():
        if q["type"] == "choice" and len(q["criteria"]) >= 2:
            keys = list(q["criteria"]); rng.shuffle(keys)
            out["questions"][qid] = {**q, "criteria": {k: q["criteria"][k] for k in keys}}
            perms[qid] = keys
        else:
            out["questions"][qid] = q
    return out, perms
