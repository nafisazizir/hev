"""Train Hev LoRA adapters and readouts on a checksummed frozen suite."""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import EVAL_ONLY, TRAINABLE, augment, materialize, none_pair, source_policy, source_seed
from .model import DecisionModel, encode, load_tokenizer
from .suite import (ROOT, SUITES_DATASET, SUITES_REVISION, base_revision, digest, load_split, manifest, object_digest,
                    source_hashes, write_json)

# The recipe that produced the published `jaredpalmer/kev-0.6b` preview (kev trial `v4-06b-hardened/00-trial-0`,
# seeds 0-2, kev commit 6cfa03ab, from that trial's provenance.json). `--recipe kev-v4` refuses to start unless
# every knob under "matched" agrees, and records the three deviations Hev cannot remove: Hev has no bf16 path,
# runs on MPS rather than an H100, and cannot hold an eight-record forward, so it reaches the same effective
# batch by accumulation. The fourth difference is the point of the experiment, not an accident: Hev's encoding
# isolates option spans and shares their start positions, kev's published checkpoint does not.
KEV_V4_RECIPE = {
    "source": "kev runs/v4-06b-hardened/00-trial-0/provenance.json (kev commit 6cfa03ab, HEAD 20fa626)",
    "published_checkpoint": "jaredpalmer/kev-0.6b",
    "suite_sha256": "1b33e566d114f9eafeff55b36c221fadb2a4ae358a1b9cc68006e82c7cfad8f1",
    "matched": {
        "base": "Qwen/Qwen3-0.6B-Base",
        "epochs": 2,
        "lr": 2e-4,
        "lora": 16,
        "effective_batch": 8,
        "ord_w": 0.0,
        "p_none": 0.1,
        "p_none_distract": 0.12,
        "p_distract": 0.15,
        "p_none_pair": 0.25,
    },
    "kev_only_knobs": {"perm_kl": 0.0, "anchor_w": 0.0, "head_dim": 256, "special_embeddings": False},
}


def question_loss(logits, question, device, ord_w=0.0):
    label = int(question["label"])
    loss = F.cross_entropy(logits[None], torch.tensor([label], device=device))
    if question["qtype"] == "score" and ord_w > 0:
        probabilities = F.softmax(logits, -1)
        observed_cdf = (torch.arange(len(probabilities) - 1, device=device) >= label).to(probabilities.dtype)
        loss = loss + ord_w * (probabilities.cumsum(-1)[:-1] - observed_cdf).square().mean()
    return loss


def reference_recipe_report(args, suite_sha256, device):
    """Check the run against kev's published v4 recipe and list the deviations that remain. Raises on a mismatch."""
    matched = {
        "base": args.base,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora": args.lora,
        "effective_batch": args.batch * args.accum,
        "ord_w": args.ord_w,
        "p_none": args.p_none,
        "p_none_distract": args.p_none_distract,
        "p_distract": args.p_distract,
        "p_none_pair": args.p_none_pair,
    }
    differing = {key: {"kev": value, "hev": matched[key]} for key, value in KEV_V4_RECIPE["matched"].items()
                 if matched[key] != value}
    if differing:
        raise ValueError(f"--recipe kev-v4 requires these knobs to match kev's published trial: {differing}")
    if suite_sha256 != KEV_V4_RECIPE["suite_sha256"]:
        raise ValueError("--recipe kev-v4 requires the decision-v4 suite kev trained on")
    return {
        "name": "kev-v4",
        "reference": KEV_V4_RECIPE,
        "matched": matched,
        "accepted_deviations": [
            {"knob": "precision", "kev": "bf16 autocast", "hev": "fp32",
             "why": "Hev has no mixed-precision path; the float 4D mask is only known-good on eager fp32 on MPS"},
            {"knob": "device", "kev": "NVIDIA H100 80GB (cuda)", "hev": device,
             "why": "this is the hardware available; kernels, reductions and RNG all differ"},
            {"knob": "microbatching", "kev": "batch 8, accum 1", "hev": f"batch {args.batch}, accum {args.accum}",
             "why": "same effective batch of 8 records; MPS memory does not hold an eight-record forward"},
            {"knob": "encoding", "kev": "plain packing, option_isolation off", "hev": "option-isolating mask, shared option start positions",
             "why": "the object of study, not a deviation to remove"},
        ],
        "not_implemented_in_hev": KEV_V4_RECIPE["kev_only_knobs"],
    }


def accumulation_records(total, batch, accum, microbatch):
    start = (microbatch // accum) * accum * batch
    return min(accum * batch, total - start)


def validate_args(args):
    if min(args.epochs, args.lora, args.batch, args.accum) < 1:
        raise ValueError("epochs, lora, batch and accum must be positive")
    if args.lr <= 0 or args.ord_w < 0:
        raise ValueError("lr must be positive and ord-w must be nonnegative")
    if not 0 <= args.p_none_pair <= 1:
        raise ValueError("p-none-pair must be a probability")
    if args.objective_records < 0:
        raise ValueError("objective-records must be nonnegative")
    probabilities = (args.p_none, args.p_none_distract, args.p_distract)
    if min(probabilities) < 0 or sum(probabilities) > 1:
        raise ValueError("augmentation probabilities must be nonnegative and sum to at most one")


def validate_training_records(records, trainable=TRAINABLE, eval_only=EVAL_ONLY):
    """Refuse an eval-only source at either level, and refuse a record source the suite does not declare trainable.

    Record sources are the policy unit: `_meta.source` is one of the suite's declared sources, while a question's
    `src` is a finer task name inside it (`agnews_yn` inside agnews, `contrastive_return_window` inside v4's
    `legacy_policy`). So the trainable check is on record sources and the eval-only refusal is on exact names at
    both levels, which is what keeps a disguised mmlu question out of a training batch.
    """
    if not records:
        raise ValueError("empty training set")
    record_sources = {record["_meta"]["source"] for record in records}
    sources = record_sources | {
        question.get("src")
        for record in records
        for question in record.get("questions", {}).values()
        if question.get("src")
    }
    forbidden = sources & set(eval_only)
    if forbidden:
        raise ValueError(f"training partition contains eval-only sources: {sorted(forbidden)}")
    undeclared = record_sources - set(trainable)
    if undeclared:
        raise ValueError(f"training partition contains sources the suite does not declare trainable: {sorted(undeclared)}")


def prepare_output(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {path}")
    path.mkdir(parents=True)
    return path


def default_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def package_versions():
    values = {}
    for package in ("torch", "transformers", "peft", "numpy"):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = None
    return values


def git_state(root=ROOT):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip())
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def runtime_provenance(device):
    return {
        "source_hashes": source_hashes(),
        "git": git_state(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": package_versions(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "dtype": "fp32",
    }


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def objective_probe(records, limit):
    """Records for the fixed before/after objective: all of them, or a deterministic subset shared by every seed.

    The objective is a loss-decrease gate, not a research metric, so on a large partition it is measured on a fixed
    probe instead of every record. The order is by record id digest, so the probe does not depend on the seed, the
    shuffle or the file order, and `training_metrics.json` records how many records went into it.
    """
    if not limit or limit >= len(records):
        return list(records)
    return sorted(records, key=lambda record: hashlib.sha256(str(record["_meta"]["id"]).encode()).hexdigest())[:limit]


def fixed_objective(model, tokenizer, requests, ord_w):
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for request in requests:
            record = materialize(request)
            logits = model(encode(tokenizer, record, strict=True))
            losses.append(torch.stack([
                question_loss(z.float(), question, model.device, ord_w)
                for z, question in zip(logits, record["questions"])
            ]).mean().item())
    model.train(was_training)
    return float(sum(losses) / len(losses))


def peak_device_bytes(device):
    if device == "mps":
        return int(torch.mps.current_allocated_memory())
    if device == "cuda":
        return int(torch.cuda.max_memory_allocated())
    return 0


def make_scheduler(optimizer, lr, total_steps):
    if total_steps <= 2:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    warmup_steps = max(2, math.ceil(total_steps * 0.1))
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=warmup_steps / total_steps
    )


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--head", choices=["pointer", "set"], required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ord-w", type=float, default=0.0)
    ap.add_argument("--p-none", type=float, default=0.1)
    ap.add_argument("--p-none-distract", type=float, default=0.12)
    ap.add_argument("--p-distract", type=float, default=0.15)
    ap.add_argument("--p-none-pair", type=float, default=0.0,
                    help="fraction of records that additionally emit a none-present/none-absent minimal pair")
    ap.add_argument("--objective-records", type=int, default=0,
                    help="cap the fixed before/after objective at this many records (0 = the whole partition)")
    ap.add_argument("--recipe", choices=["none", "kev-v4"], default="none",
                    help="refuse to start unless the run matches a published reference recipe, and record the deviations")
    ap.add_argument("--require-loss-decrease", action="store_true")
    return ap


def train(args, output):
    suite = Path(args.suite)
    partition = suite / "train.jsonl"
    mirrored = not partition.exists()          # load_split fetches it from the pinned mirror and verifies the bytes
    records = load_split(suite, "train", fetch=True)
    suite_manifest = manifest(suite)
    trainable, eval_only = source_policy(suite_manifest)
    validate_training_records(records, trainable, eval_only)
    revision = base_revision(suite, args.base)
    if not revision:
        raise ValueError("base model is not pinned by the suite manifest")
    suite_sha256 = digest(suite / "manifest.json")
    device = args.device or default_device()
    resolved = {**vars(args), "device": device}
    provenance = runtime_provenance(device)
    config = {
        "status": "configured",
        "args": resolved,
        "suite_sha256": suite_sha256,
        "base_revision": revision,
        "holdout_sources": suite_manifest.get("holdout_sources", []),
        "source_policy": {"trainable": list(trainable), "eval_only": list(eval_only),
                          "declared_by": "suite manifest" if suite_manifest.get("trainable_sources") is not None
                          else "hev.data defaults"},
        "train_partition": {
            "path": str(partition),
            "sha256": digest(partition),
            "records": len(records),
            "fetched_from_mirror": mirrored,
            "mirror": {"dataset": SUITES_DATASET, "revision": SUITES_REVISION} if mirrored else None,
        },
        "recipe": reference_recipe_report(args, suite_sha256, device) if args.recipe == "kev-v4" else None,
        "objective": "record-mean cross-entropy plus optional normalized ranked probability score",
        "provenance": provenance,
    }
    config["config_sha256"] = object_digest({key: value for key, value in config.items() if key != "config_sha256"})
    write_json(output / "training_config.json", config)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    tokenizer = load_tokenizer(args.base, revision=revision)
    model = DecisionModel(args.base, tokenizer, device, lora=args.lora, revision=revision, head=args.head)
    model.lm.config.use_cache = False
    parameters = model.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    microbatches = math.ceil(len(records) / args.batch)
    total_steps = args.epochs * math.ceil(microbatches / args.accum)
    scheduler = make_scheduler(optimizer, args.lr, max(total_steps, 1))

    probe = objective_probe(records, args.objective_records)
    before = fixed_objective(model, tokenizer, probe, args.ord_w)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sync(device)
    started = time.perf_counter()
    records_seen = questions_seen = forward_tokens = optimizer_steps = 0
    forward_records = none_pair_records = 0
    peak_memory = peak_device_bytes(device)
    step_metrics = []
    group_loss = 0.0
    group_forward_records = 0

    for epoch in range(args.epochs):
        order = list(records)
        rng.shuffle(order)
        for microbatch in range(microbatches):
            chunk = order[microbatch * args.batch:(microbatch + 1) * args.batch]
            encoded = []
            materialized = []
            for request in chunk:
                item_rng = random.Random(source_seed(args.seed, f"{epoch}:{request['_meta']['id']}"))
                variants = [augment(
                    request,
                    item_rng,
                    p_none=args.p_none,
                    p_none_distract=args.p_none_distract,
                    p_distract=args.p_distract,
                )]
                if args.p_none_pair > 0 and item_rng.random() < args.p_none_pair:
                    pair = none_pair(request, item_rng)      # kev/train.py:170-171; [] when no Choice is eligible
                    variants += pair
                    none_pair_records += len(pair)
                for variant in variants:
                    record = materialize(variant)
                    packed = encode(tokenizer, record, strict=True)
                    materialized.append(record)
                    encoded.append(packed)
                    forward_tokens += len(packed["ids"])
            logits_batch = model.forward_batch(encoded)
            record_losses = []
            for logits, record in zip(logits_batch, materialized):
                record_losses.append(torch.stack([
                    question_loss(z.float(), question, device, args.ord_w)
                    for z, question in zip(logits, record["questions"])
                ]).mean())
                questions_seen += len(record["questions"])
            loss_sum = torch.stack(record_losses).sum()
            if not torch.isfinite(loss_sum):
                raise ValueError("non-finite training loss")
            # weight by source records in the accumulation group so none-pair siblings do not inflate a record's
            # share of the gradient (kev/train.py, "group_records")
            group_size = accumulation_records(len(order), args.batch, args.accum, microbatch) * (len(materialized) / len(chunk))
            (loss_sum / group_size).backward()
            records_seen += len(chunk)
            forward_records += len(materialized)
            group_forward_records += len(materialized)
            group_loss += loss_sum.detach().item()
            peak_memory = max(peak_memory, peak_device_bytes(device))
            final_microbatch = microbatch + 1 == microbatches
            if (microbatch + 1) % args.accum == 0 or final_microbatch:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                step_metrics.append({
                    "step": optimizer_steps,
                    "epoch": epoch,
                    "objective": group_loss / group_forward_records,
                    "lr": scheduler.get_last_lr()[0],
                })
                print(
                    f"ep{epoch} step {optimizer_steps}/{total_steps} objective "
                    f"{step_metrics[-1]['objective']:.4f}",
                    flush=True,
                )
                group_loss = 0.0
                group_forward_records = 0
                if device == "mps":
                    torch.mps.empty_cache()

    sync(device)
    training_seconds = time.perf_counter() - started
    after = fixed_objective(model, tokenizer, probe, args.ord_w)
    metrics = {
        "status": "success",
        "wall_seconds": training_seconds,
        "fixed_train_objective_before": before,
        "fixed_train_objective_after": after,
        "fixed_train_objective_delta": after - before,
        "fixed_train_objective_records": len(probe),
        "loss_decrease_required": args.require_loss_decrease,
        "optimizer_steps": optimizer_steps,
        "step_metrics": step_metrics,
        "records_seen": records_seen,
        "forward_records": forward_records,
        "none_pair_records": none_pair_records,
        "questions_seen": questions_seen,
        "requested_records": args.epochs * len(records),
        "forward_tokens": forward_tokens,
        "peak_device_bytes": peak_memory,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "device": device,
    }
    if args.require_loss_decrease and not after < before:
        metrics["status"] = "failed_loss_gate"
        write_json(output / "training_metrics.json", metrics)
        raise ValueError(f"fixed training objective did not decrease: {before} -> {after}")
    if source_hashes() != provenance["source_hashes"] or digest(suite / "manifest.json") != suite_sha256:
        metrics["status"] = "failed_provenance_gate"
        write_json(output / "training_metrics.json", metrics)
        raise ValueError("source or suite changed during training")

    model.lm.save_pretrained(output)
    tokenizer.save_pretrained(output)
    torch.save({
        "readout": model.readout_state_dict(),
        "base": args.base,
        "base_revision": revision,
        "lora": args.lora,
        "suite_sha256": suite_sha256,
        "args": resolved,
        "training_config_sha256": config["config_sha256"],
        "provenance": provenance,
    }, output / "readout.pt")
    write_json(output / "training_metrics.json", metrics)
    print(f"saved {output}", flush=True)


def main():
    ap = parser()
    args = ap.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        ap.error(str(error))
    try:
        output = prepare_output(args.out)
    except FileExistsError as error:
        ap.error(str(error))
    try:
        train(args, output)
    except Exception as error:
        write_json(output / "failure.json", {"error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
