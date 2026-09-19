# AGENTS.md

Orientation for any agent or contributor starting a fresh session on this repo. Read this, then PLAN.md, then whichever doc the current milestone points at.

## What this is

hev is Nafis's research project: a Jev-style decision model (typed questions, probabilities out, single prefill, no decoding) whose one deliberate departure from kev is **option-level isolation** in the attention mask plus a permutation-equivariant readout. README.md has the pitch, docs/DESIGN.md has the mechanism, docs/DECISIONS.md has the why.

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

Hardware: Apple M4 Max, 36 GB. Base models are not cached yet; first real run downloads Qwen3-0.6B-Base (about 1.2 GB).

## Where things are

| Need | Look at |
|---|---|
| Roadmap and current milestone | PLAN.md |
| Mask rule, positions, heads, invariance argument | docs/DESIGN.md |
| Illustrated walkthrough for newcomers (open in a browser) | docs/explainer.html |
| Why each choice was made, alternatives rejected | docs/DECISIONS.md |
| What kev has, what to port, kev/Jev baseline numbers | docs/KEV.md |
| Suite format, provenance, policy | docs/EVALS.md, evals/README.md |
| Original idea note and screenshots | .ideas/ |

## Style

Short modules, docstring at the top saying what the file is for. Compact but not golfed. Tests assert the architectural claims, not just that code runs.
