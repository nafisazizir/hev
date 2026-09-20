"""FastAPI sidecar for the playground: loads one evaluated checkpoint, exposes prefill-only decisions.

Run: uv run --extra serve python -m hev.serve --run runs/m2-pointer-s0 --port 8008
     --run also accepts hf://OWNER/hev-0.6b@seed-0
"""
import argparse
import json
import math
import threading
import time

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .api import MODEL_NAME, SystemOneRequest, output_tokens, to_answers, to_record
from .checkpoint import resolve_run
from .evaluate import LocalPredictor, default_device, scaled_probabilities, sync
from .model import encode

INFER_MAX_STATE, INFER_MAX_BRANCH = 8192, 8192

app = FastAPI(title="hev")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATE = {"run": None, "predictor": None, "temperature": None, "lock": threading.Lock()}


def _probs(record):
    """Encode one record and return calibrated per-question probabilities plus request metadata."""
    predictor = STATE["predictor"]
    if predictor is None:
        raise HTTPException(503, "model is not loaded")
    try:
        encoded = encode(predictor.tokenizer, record, max_state=INFER_MAX_STATE, max_branch=INFER_MAX_BRANCH)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if len(encoded["ids"]) > predictor.model.lm.config.max_position_embeddings:
        raise HTTPException(422, "request exceeds the model's context window")
    for qtype, option_indices in zip(encoded["qtypes"], encoded["opt_idx"]):
        if qtype == "score" and len(option_indices) > predictor.model.level.num_embeddings:
            raise HTTPException(422, "score has more levels than this checkpoint supports")
    with STATE["lock"]:
        sync(predictor.device)
        t = time.time()
        raw = predictor.model.probs(encoded)
        sync(predictor.device)
        seconds = time.time() - t
    probs = [scaled_probabilities(p.numpy(), STATE["temperature"]).tolist() for p in raw]
    return probs, {"tokens": len(encoded["ids"]), "latency_ms": round(seconds * 1000, 1)}


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    """TypeSafe-compatible endpoint: typed questions in, typed answers out, one prefill pass."""
    record, meta = to_record(req)
    probs, m = _probs(record)
    answers = to_answers(probs, meta)
    return {"model": req.model,
            "answers": answers,
            "usage": {"input_tokens": m["tokens"],
                      "output_tokens": output_tokens(STATE["predictor"].tokenizer, answers)},
            "latency_ms": m["latency_ms"]}


@app.get("/v1/models")
def models():
    predictor = STATE["predictor"]
    if predictor is None:
        return {"models": [{"id": MODEL_NAME, "aliases": ["jev-latest"], "run": STATE["run"],
                            "base": None, "head": None, "temperature": STATE["temperature"]}]}
    metadata = predictor.metadata
    return {"models": [{"id": MODEL_NAME, "aliases": ["jev-latest"], "run": STATE["run"],
                        "base": metadata["base"], "head": metadata["readout"]["head_kind"],
                        "temperature": STATE["temperature"]}]}


def load_run(run, device):
    """Load a checkpoint and its evaluated calibration temperature into STATE."""
    source = str(run)
    resolved = resolve_run(run)
    predictor = LocalPredictor(resolved, device)
    result = json.loads((resolved / "result.json").read_text())
    if result.get("status") != "success":
        raise ValueError("run evaluation did not complete successfully")
    if result.get("test_evaluated") is not False:
        raise ValueError("refusing to serve a checkpoint evaluated on the locked test split")
    if result.get("head") != predictor.metadata["readout"]["head_kind"]:
        raise ValueError("result head does not match the checkpoint readout head")
    identity = {
        "base": predictor.metadata["base"],
        "base_revision": predictor.metadata["base_revision"],
        "training_suite_sha256": predictor.metadata["suite_sha256"],
    }
    if any(result.get(key) != value for key, value in identity.items()):
        raise ValueError("result provenance does not match the checkpoint")
    config_sha256 = result.get("training_config", {}).get("config_sha256")
    if config_sha256 != predictor.metadata.get("training_config_sha256"):
        raise ValueError("result training config does not match the checkpoint")
    temperature = result.get("temperature_fit", {}).get("temperature")
    if temperature is None or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("run has no finite positive fitted temperature")
    STATE.update({"run": source, "predictor": predictor, "temperature": temperature})


def main():
    parser = argparse.ArgumentParser(description="serve one evaluated hev checkpoint")
    parser.add_argument("--run", default="runs/m2-pointer-s0",
                        help="local run directory or hf://OWNER/hev-0.6b@seed-0")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    args = parser.parse_args()
    device = args.device or default_device()
    load_run(args.run, device)
    print(f"serving {STATE['run']} on {device} (head={STATE['predictor'].metadata['readout']['head_kind']}, "
          f"temperature={STATE['temperature']:.3f}) at http://127.0.0.1:{args.port}")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
