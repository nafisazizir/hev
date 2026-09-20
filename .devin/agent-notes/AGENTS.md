# AGENTS.md

Orientation for any agent or contributor starting a fresh session on this repo. Read this, then PLAN.md, then whichever doc the current milestone points at.

## What this is

Hev is Nafis's research project: a Jev-style decision model (typed questions, probabilities out, single prefill, no decoding) whose one deliberate departure from kev is **option-level isolation** in the attention mask plus a permutation-equivariant readout. README.md has the pitch, docs/DESIGN.md has the mechanism, docs/DECISIONS.md has the why.

The sibling repo `/Users/nafis/Documents/personal/kev` is Jared Palmer's kev (clone at commit cc954f2). It is the reference implementation and the source of the eval suites. docs/KEV.md maps its files and says what to port and what not to.

## Working rules

- **Fresh code, borrowed evals.** Model, mask, heads, training loop are written here. Eval suites, API schema and augmentation are copied from kev with attribution. Do not fork kev's train/evaluate wholesale; port pieces when a milestone needs them and note it in docs/KEV.md.
- **Evals are immutable.** Never edit anything under `evals/`. New suites go in a new directory with a manifest. `hev.suite.load_split` checksums every load.
- **Test split is locked.** Never pass `allow_test=True` during model selection or hyperparameter search. It is for a final, pre-declared comparison only.
- **Eval-only sources never train.** `hev.data.EVAL_ONLY` lists them. Training code must refuse them.
- **Runs are never overwritten.** A run directory that exists is an error. Failures are kept.
- **Every claim links to an artifact.** Numbers in PLAN.md or docs point at a file under `runs/` or `evals/`. No unlinked numbers.
- **Kev's numbers are the baseline.** Kev's PLAN.md and MODEL_CARD.md hold the kev and Jev figures on the same suites. Quote them with their source path.
- **Tests must run offline.** `uv run pytest` uses a fake tokenizer and a tiny random backbone. Hub-dependent tests are marked `hub` and gated by `HEV_HUB_TESTS=1`.

## Commands

```bash
export PATH="/opt/homebrew/bin:$PATH"   # uv lives here on this machine
uv sync --group dev
uv run pytest
uv run pytest -k invariance -v
```

M2 commands (completed 2026-09-20; retained for reproducibility). The listed output directories are immutable and already exist, so do not rerun these commands with the same paths:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HEV_HUB_TESTS=1 uv run pytest -p no:cacheprovider tests/test_hub.py -v

uv run python -m hev.train --suite evals/decision-v2 --out runs/m2-pointer-s0 --head pointer --device mps --seed 0 --epochs 2 --lr 2e-4 --lora 16 --batch 1 --accum 8 --ord-w 0 --p-none 0.1 --p-none-distract 0.12 --p-distract 0.15 --require-loss-decrease
uv run python -m hev.train --suite evals/decision-v2 --out runs/m2-set-s0 --head set --device mps --seed 0 --epochs 2 --lr 2e-4 --lora 16 --batch 1 --accum 8 --ord-w 0 --p-none 0.1 --p-none-distract 0.12 --p-distract 0.15 --require-loss-decrease

uv run python -m hev.evaluate \
  --run runs/m2-pointer-s0 \
  --suite evals/decision-v2 \
  --transfer evals/transfer-v2 \
  --device mps \
  --seed 1 \
  --permutations 6 \
  --bootstrap-samples 10000 \
  --level-zero-ablation

uv run python -m hev.evaluate \
  --run runs/m2-set-s0 \
  --suite evals/decision-v2 \
  --transfer evals/transfer-v2 \
  --device mps \
  --seed 1 \
  --permutations 6 \
  --bootstrap-samples 10000

uv run python -m hev.compare \
  --pointer runs/m2-pointer-s0 \
  --set runs/m2-set-s0 \
  --kev-seed0 /Users/nafis/Documents/personal/kev/runs/ablation-v2/06-trial-6/result.json \
  --kev-seed1 /Users/nafis/Documents/personal/kev/runs/ablation-v2/07-trial-7/result.json \
  --out runs/m2-comparison-s0 \
  --bootstrap-samples 10000 \
  --seed 20260919
```

M3 commands (completed 2026-09-20; output paths are immutable and must not be reused):

```bash
uv run python -m hev.train --suite evals/decision-v2 --out runs/m3-pointer-s1 --head pointer --device mps --seed 1 --epochs 2 --lr 2e-4 --lora 16 --batch 1 --accum 8 --ord-w 0 --p-none 0.1 --p-none-distract 0.12 --p-distract 0.15 --require-loss-decrease
uv run python -m hev.evaluate --run runs/m3-pointer-s1 --suite evals/decision-v2 --transfer evals/transfer-v2 --device mps --seed 1 --permutations 6 --bootstrap-samples 10000
uv run python -m hev.replicate --seed0 runs/m2-pointer-s0 --seed1 runs/m3-pointer-s1 --out runs/m3-pointer-replication-v2 --bootstrap-samples 10000 --seed 20260920
uv run --extra serve python -m hev.serve --run runs/m3-pointer-s1 --port 8008

set -a && source .env && set +a && uv run --extra jev python -m hev.jev --suite evals/decision-v2 --split calibration --out runs/m3-jev-direct-calibration-v2-r1 --budget 0.05 --max-calls 500
set -a && source .env && set +a && uv run --extra jev python -m hev.jev --suite evals/decision-v2 --split development --out runs/m3-jev-direct-decision-v2 --budget 0.10 --max-calls 1300
set -a && source .env && set +a && uv run --extra jev python -m hev.jev --suite evals/transfer-v2 --split development --out runs/m3-jev-direct-transfer-v2 --budget 0.10 --max-calls 800

uv run python -m hev.three_way \
  --hev-seed0 runs/m2-pointer-s0 \
  --hev-seed1 runs/m3-pointer-s1 \
  --hev-set runs/m2-set-s0 \
  --kev-seed0 /Users/nafis/Documents/personal/kev/runs/ablation-v2/06-trial-6/result.json \
  --kev-seed1 /Users/nafis/Documents/personal/kev/runs/ablation-v2/07-trial-7/result.json \
  --jev-calibration runs/m3-jev-direct-calibration-v2-r1 \
  --jev-decision runs/m3-jev-direct-decision-v2 \
  --jev-transfer runs/m3-jev-direct-transfer-v2 \
  --out runs/m3-three-way-v2-r1 \
  --bootstrap-samples 10000 \
  --seed 20260920
```

The M3 Jev and three-way output directories above are immutable and must not be reused. Failed attempts retained under `runs/m3-jev-calibration-v2`, `runs/m3-jev-direct-calibration-v2`, and `runs/m3-three-way-v2` must also not be reused.

Hardware: Apple M4 Max, 36 GB. The pinned Qwen3-0.6B-Base tokenizer and model are cached from M1.

## Where things are

| Need | Look at |
|---|---|
| Roadmap and current milestone | PLAN.md |
| M2/M3 results and conclusions | docs/RESULTS.md, runs/m2-comparison-s0/result.json, runs/m3-pointer-replication-v2/result.json |
| Mask rule, positions, heads, invariance argument | docs/DESIGN.md |
| Illustrated walkthrough for newcomers (open in a browser) | docs/explainer.html |
| Why each choice was made, alternatives rejected | docs/DECISIONS.md |
| What kev has, what to port, kev/Jev baseline numbers | docs/KEV.md |
| Suite format, provenance, policy | docs/EVALS.md, evals/README.md |
| Original idea note and screenshots | .ideas/ |

## Style

Short modules, docstring at the top saying what the file is for. Compact but not golfed. Tests assert the architectural claims, not just that code runs.
