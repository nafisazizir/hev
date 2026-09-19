# hev

An order-invariant, Jev-inspired decision model. Typed questions in, calibrated probabilities out, one forward pass, no decoding.

hev is a research project by Nafis, started 2026-09-19. It follows the shape of TypeSafe's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), as reconstructed in [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked) and reproduced at laptop scale by Jared Palmer's [kev](https://github.com/jaredpalmer/kev). It exists to test one architectural change that kev deferred:

> **kev isolates questions from each other. hev also isolates every option from its sibling options.**

In kev, a question's options are packed one after another under a causal mask, so option 3's hidden state has already read options 1 and 2. That is where option-order sensitivity comes from, and kev measures it at 7% argmax flips and a p90 probability spread of 0.25 under reordering. hev gives every option its own branch under the block mask, with all options of a question sharing the same start position, so the backbone representation of an option is a function of (state, instruction, that option's text) and nothing else. Order can then only enter through the readout, and the readouts are permutation-equivariant by construction. Flip rate is zero by design. The experiment is what that costs in accuracy, and whether a small listwise set-readout recovers it.

Full design: [docs/DESIGN.md](docs/DESIGN.md). Illustrated walkthrough with the end-to-end pipeline and computed mask grids: [docs/explainer.html](docs/explainer.html) (open in a browser). Roadmap: [PLAN.md](PLAN.md). Decision log: [docs/DECISIONS.md](docs/DECISIONS.md).

## Status

M1 complete. The model, mask and readout heads are covered by offline architectural tests; the pinned Qwen3 tokenizer passes the Hub context-limit checks; and both PointerHead and SetHead have immutable MPS smoke checkpoints with decreasing fixed-set loss, complete development evaluation and zero permutation flips. Smoke accuracy is a pipeline check, not a research result. See PLAN.md for linked artifacts and M2.

## Layout

```
hev/api.py       TypeSafe-compatible request/response schema (byte-compatible with kev's suites)
hev/model.py     packing, block mask, PointerHead / SetHead, DecisionModel
hev/data.py      labelled request -> internal record; augmentation ported from kev
hev/suite.py     checksummed frozen-suite loader; locked test split
evals/           frozen suites copied from kev (see evals/README.md for provenance)
tests/           offline tests: fake tokenizer + tiny random Qwen2 backbone
docs/            DESIGN, DECISIONS, KEV (what to reuse from kev and when), EVALS
runs/            training outputs (gitignored except ledgers)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). Tested on Apple Silicon (MPS); CUDA path mirrors kev's and is untested here.

```bash
uv sync --group dev
uv run pytest            # offline, ~10s
```

Hub and M1 smoke commands:

```bash
HEV_HUB_TESTS=1 uv run pytest tests/test_hub.py -v
uv run python -m hev.train --suite evals/smoke-v1 --out runs/smoke-pointer-s0 --head pointer --device mps --seed 0 --epochs 5 --lr 2e-4 --lora 16 --batch 1 --accum 4 --require-loss-decrease
uv run python -m hev.evaluate --run runs/smoke-pointer-s0 --suite evals/smoke-v1 --device mps --seed 1 --permutations 6
uv run python -m hev.train --suite evals/smoke-v1 --out runs/smoke-set-s0 --head set --device mps --seed 0 --epochs 5 --lr 2e-4 --lora 16 --batch 1 --accum 4 --require-loss-decrease
uv run python -m hev.evaluate --run runs/smoke-set-s0 --suite evals/smoke-v1 --device mps --seed 1 --permutations 6
```

Run directories and evaluation artifacts are immutable. Choose new `--out` paths for reruns; failed artifacts are preserved.

## Comparing against kev and Jev

hev's request and response shapes, delimiter tokens, and context limits match kev exactly, so kev's frozen suites in `evals/` load unchanged and any hev number is directly comparable to the kev and Jev numbers recorded in kev's `runs/` and `PLAN.md`. The rule is: never train on an eval-only source, never touch the `test` split for selection. `hev.suite.load_split` enforces both.

## Credits

Architecture lineage: TypeSafe (Jev), Archer Hume (reconstruction), Jared Palmer (kev, Apache-2.0). `hev/api.py` and the augmentation helpers in `hev/data.py` are adapted from kev. The eval suites are kev's, copied byte-for-byte.
