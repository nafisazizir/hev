"""Train Hev LoRA adapters and readouts on a checksummed frozen suite."""
import argparse
import json
import math
import os
import random
import resource
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import EVAL_ONLY, augment, materialize, source_seed
from .model import DecisionModel, encode, load_tokenizer
from .suite import base_revision, digest, load_split, manifest, write_json


def question_loss(logits, question, device, ord_w=0.0):
    label = int(question["label"])
    loss = F.cross_entropy(logits[None], torch.tensor([label], device=device))
    if question["qtype"] == "score" and ord_w > 0:
        probabilities = F.softmax(logits, -1)
        observed_cdf = (torch.arange(len(probabilities) - 1, device=device) >= label).to(probabilities.dtype)
        loss = loss + ord_w * (probabilities.cumsum(-1)[:-1] - observed_cdf).square().mean()
    return loss


def accumulation_records(total, batch, accum, microbatch):
    start = (microbatch // accum) * accum * batch
    return min(accum * batch, total - start)


def validate_args(args):
    if min(args.epochs, args.lora, args.batch, args.accum) < 1:
        raise ValueError("epochs, lora, batch and accum must be positive")
    if args.lr <= 0 or args.ord_w < 0:
        raise ValueError("lr must be positive and ord-w must be nonnegative")
    probabilities = (args.p_none, args.p_none_distract, args.p_distract)
    if min(probabilities) < 0 or sum(probabilities) > 1:
        raise ValueError("augmentation probabilities must be nonnegative and sum to at most one")


def validate_training_records(records):
    if not records:
        raise ValueError("empty training set")
    sources = {record["_meta"]["source"] for record in records}
    sources.update(
        question.get("src")
        for record in records
        for question in record.get("questions", {}).values()
        if question.get("src")
    )
    forbidden = sources & set(EVAL_ONLY)
    if forbidden:
        raise ValueError(f"training partition contains eval-only sources: {sorted(forbidden)}")


def prepare_output(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {path}")
    path.mkdir(parents=True)
    return path


def default_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


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
    ap.add_argument("--require-loss-decrease", action="store_true")
    return ap


def train(args, output):
    suite = Path(args.suite)
    records = load_split(suite, "train")
    validate_training_records(records)
    suite_manifest = manifest(suite)
    revision = base_revision(suite, args.base)
    if not revision:
        raise ValueError("base model is not pinned by the suite manifest")
    suite_sha256 = digest(suite / "manifest.json")
    device = args.device or default_device()
    resolved = {**vars(args), "device": device}
    config = {
        "status": "configured",
        "args": resolved,
        "suite_sha256": suite_sha256,
        "base_revision": revision,
        "holdout_sources": suite_manifest.get("holdout_sources", []),
        "objective": "record-mean cross-entropy plus optional normalized ranked probability score",
    }
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

    before = fixed_objective(model, tokenizer, records, args.ord_w)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sync(device)
    started = time.perf_counter()
    records_seen = questions_seen = forward_tokens = optimizer_steps = 0
    peak_memory = peak_device_bytes(device)
    step_metrics = []
    group_loss = 0.0
    group_records_seen = 0

    for epoch in range(args.epochs):
        order = list(records)
        rng.shuffle(order)
        for microbatch in range(microbatches):
            chunk = order[microbatch * args.batch:(microbatch + 1) * args.batch]
            encoded = []
            materialized = []
            for request in chunk:
                item_rng = random.Random(source_seed(args.seed, f"{epoch}:{request['_meta']['id']}"))
                request = augment(
                    request,
                    item_rng,
                    p_none=args.p_none,
                    p_none_distract=args.p_none_distract,
                    p_distract=args.p_distract,
                )
                record = materialize(request)
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
            group_size = accumulation_records(len(order), args.batch, args.accum, microbatch)
            (loss_sum / group_size).backward()
            records_seen += len(chunk)
            group_records_seen += len(chunk)
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
                    "objective": group_loss / group_records_seen,
                    "lr": scheduler.get_last_lr()[0],
                })
                print(
                    f"ep{epoch} step {optimizer_steps}/{total_steps} objective "
                    f"{step_metrics[-1]['objective']:.4f}",
                    flush=True,
                )
                group_loss = 0.0
                group_records_seen = 0
                if device == "mps":
                    torch.mps.empty_cache()

    sync(device)
    training_seconds = time.perf_counter() - started
    after = fixed_objective(model, tokenizer, records, args.ord_w)
    metrics = {
        "status": "success",
        "wall_seconds": training_seconds,
        "fixed_train_objective_before": before,
        "fixed_train_objective_after": after,
        "fixed_train_objective_delta": after - before,
        "loss_decrease_required": args.require_loss_decrease,
        "optimizer_steps": optimizer_steps,
        "step_metrics": step_metrics,
        "records_seen": records_seen,
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

    model.lm.save_pretrained(output)
    tokenizer.save_pretrained(output)
    torch.save({
        "readout": model.readout_state_dict(),
        "base": args.base,
        "base_revision": revision,
        "lora": args.lora,
        "suite_sha256": suite_sha256,
        "args": resolved,
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
