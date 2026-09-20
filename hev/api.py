"""TypeSafe-compatible request/response shapes for POST /v1/systemone.

The contract is TypeSafe's System One API (https://docs.typesafe.ai/api), mirrored so the official
`typesafe-sdk` and kev's playground work against hev with only a base_url change. The three question
types are mapped onto one primitive, "score K options against a decision vector":

    noul   -> 2 options [no, yes];              answer = p(yes)
    choice -> options "name" or "name: desc";   answer = argmax, probabilities by name, confidence
    score  -> options = ordered level texts;    answer = expected level, legend, probabilities by index

Shapes follow kev/api.py (Apache-2.0, Jared Palmer) so that kev's frozen evaluation suites are
byte-compatible with hev. See docs/KEV.md.
"""
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

JSONContent = Union[str, dict, list, int, float, bool, None]
MAX_OPTIONS = 255
MODEL_NAME = "hev-latest"


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent
    criteria: dict[str, JSONContent] | None = None


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def _check(self):
        if not 1 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    type: Literal["score"]
    instructions: JSONContent
    criteria: list[JSONContent] = Field(min_length=2, max_length=MAX_OPTIONS)


Question = Union[Noul, Choice, Score]


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = MODEL_NAME
    questions: dict[str, Question] = Field(min_length=1)


def render(v: JSONContent, indent: int = 0) -> str:
    """Flatten str | object | array into the text the model sees. Field names are kept as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(
        f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
        for k, x in v.items()
    )


def option_text(name: str, desc: JSONContent) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def to_record(req: SystemOneRequest):
    """Request -> internal record for hev.model.encode(), plus per-question metadata to map probabilities back.

    Internal record: {"state": str, "questions": [{"instr": str, "options": [str], "qtype": str, "label": int}]}
    """
    qs, meta = [], []
    for qid, q in req.questions.items():
        instr = render(q.instructions)
        if q.type == "noul":
            c = q.criteria or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
            meta.append({"id": qid, "type": "noul"})
        elif q.type == "choice":
            opts = [option_text(k, v) for k, v in q.criteria.items()]
            meta.append({"id": qid, "type": "choice", "keys": list(q.criteria.keys())})
        else:
            opts = [render(x) for x in q.criteria]
            meta.append({"id": qid, "type": "score", "legend": {str(i): render(x) for i, x in enumerate(q.criteria)}})
        qs.append({"instr": instr, "options": opts, "qtype": q.type, "label": 0})
    return {"state": render(req.state), "questions": qs}, meta


def choice_confidence(p: list[float]) -> float:
    K = len(p)
    return 1.0 if K == 1 else (max(p) - 1 / K) / (1 - 1 / K)


def score_confidence(p: list[float]) -> float:
    """Stand-in for TypeSafe's 'distance from the modal level' statistic (exact formula unpublished):
    1 - E|level - mode| / (L - 1)."""
    L = len(p)
    mode = max(range(L), key=lambda i: p[i])
    return 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / (L - 1)


def r2(x: float) -> float:
    return round(float(x), 2)


def to_answers(probs: list[list[float]], meta: list[dict]) -> dict[str, Any]:
    out = {}
    for p, m in zip(probs, meta):
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": r2(p[1])}
        elif m["type"] == "choice":
            best = max(range(len(p)), key=lambda i: p[i])
            out[m["id"]] = {"type": "choice", "choice": m["keys"][best], "confidence": r2(choice_confidence(p)),
                            "probabilities": {k: r2(v) for k, v in zip(m["keys"], p)}}
        else:
            score = sum(i * pi for i, pi in enumerate(p))
            out[m["id"]] = {"type": "score", "score": r2(score), "legend": m["legend"],
                            "probabilities": {str(i): r2(v) for i, v in enumerate(p)}, "confidence": r2(score_confidence(p))}
    return out


def output_tokens(tok, answers: dict) -> int:
    """Billing-style count of serialized answer tokens; the model still performs no decoding."""
    import json
    return len(tok(json.dumps(answers), add_special_tokens=False).input_ids)
