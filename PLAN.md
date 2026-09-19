# Plan

Started 2026-09-19. Each milestone has an exit criterion. Tick a box only with a linked artifact.

## Hypothesis

H1. Isolating options in the mask (docs/DESIGN.md) removes option-order sensitivity entirely, at a cost in held-out accuracy that is small relative to kev's measured order effect.

H2. A permutation-equivariant listwise readout (`SetHead`) recovers most of whatever accuracy the independent `PointerHead` loses, because it restores option interaction without restoring order.

H3. Calibration (ECE, Brier) is not worse than kev's on the same suites, because the readout is trained with the same proper scoring rule.

Kev baseline to beat or match, all on `evals/decision-v2` development, Qwen3-0.6B-Base, LoRA r=16, from kev's `runs/ablation-v2/`: development accuracy 81.6% / 79.3% over two seeds; transfer-v2 accuracy 62.0% / 62.1%. Kev's order sensitivity on the released 0.5B checkpoint (kev MODEL_CARD.md): 7% argmax flips, p90 spread 0.25. Exact paths in docs/KEV.md.

## Milestones

### M0. Scaffold  (done 2026-09-19)

- [x] Repo, package, pyproject, uv environment.
- [x] API schema byte-compatible with kev's suites (`hev/api.py`).
- [x] Model with option-isolating mask, shared option start positions, `PointerHead` and `SetHead`, Score level embedding (`hev/model.py`).
- [x] Frozen suites copied: decision-v2, transfer-v2, smoke-v1 (`evals/`).
- [x] Offline tests proving: mask rule, shared start positions, unforgeable delimiters, option hidden state independent of siblings, probability invariance under reordering for both heads, question isolation, packed equals separate, batched equals single (`tests/test_model.py`).

### M1. Training loop and smoke run  (done 2026-09-19)

Exit met by two immutable Qwen3-0.6B-Base MPS runs on `evals/smoke-v1`: [Pointer training](runs/smoke-pointer-s0/training_metrics.json) / [evaluation](runs/smoke-pointer-s0/eval.json) and [Set training](runs/smoke-set-s0/training_metrics.json) / [evaluation](runs/smoke-set-s0/eval.json). Both fixed-set objectives decreased, both evaluations covered all 30 development records / 40 questions, and both had zero Choice argmax flips. These tiny-suite accuracies are pipeline checks, not research results.

- [x] `hev/train.py`: LoRA + head + level embedding; cross-entropy; optional ranked-probability term for Score; refuses eval-only sources and existing run directories; writes `training_config.json` and `training_metrics.json`.
- [x] `hev/evaluate.py`: accuracy, NLL, Brier and 10-bin ECE per source and type; calibration-only global temperature; Choice permutation study; packed-vs-separate equality; auditable `eval_rows.json`.
- [x] Hub test: the real pinned Qwen3-0.6B-Base tokenizer strictly encodes every decision-v2 and transfer-v2 development record within kev's context limits (`HEV_HUB_TESTS=1 uv run pytest tests/test_hub.py -v`: 3 passed).
- [x] Pointer smoke: fixed objective 1.970 → 0.213; development accuracy 50.0%; flip rate 0; p90 correct-probability spread 0.000006; packed max difference 0.000007; 15.6 s training; 3.23 GB peak MPS allocation. [Metrics](runs/smoke-pointer-s0/training_metrics.json), [evaluation](runs/smoke-pointer-s0/eval.json).
- [x] Set smoke: fixed objective 1.807 → 0.153; development accuracy 30.0%; flip rate 0; p90 correct-probability spread 0.000007; packed max difference 0.000003; 13.9 s training; 3.25 GB peak MPS allocation. [Metrics](runs/smoke-set-s0/training_metrics.json), [evaluation](runs/smoke-set-s0/eval.json).

### M2. First real comparison  (done 2026-09-20)

Exit met by the immutable [PointerHead result](runs/m2-pointer-s0/result.json), [SetHead result](runs/m2-set-s0/result.json), and [aggregate comparison](runs/m2-comparison-s0/result.json). Both runs completed the predeclared recipe and full development coverage without test access. H1 and H3 were supported; H2 was inconclusive because PointerHead's loss to kev seed 0 was only 0.92 percentage points, below the material-loss threshold. SetHead improved decision accuracy by 0.25 points with a paired 95% CI of −1.17 to +1.58 points, so it showed no supported advantage. Full tables and interpretation: [docs/RESULTS.md](docs/RESULTS.md).

- [x] Train both heads, seed 0, kev's v2 recipe (2 epochs, r=16, lr 2e-4, effective batch 8, Qwen3-0.6B-Base pinned to the suite's revision). [Comparison](runs/m2-comparison-s0/result.json)
- [x] Temperature fit on calibration only, applied unchanged to transfer. [Pointer](runs/m2-pointer-s0/result.json), [Set](runs/m2-set-s0/result.json)
- [x] Ablation: `PointerHead` with Score level embedding zeroed. The embedding changed probabilities but was not proven useful. [Ablation](runs/m2-pointer-s0/result.json)
- [x] Write immutable result ledgers and summarise them in [docs/RESULTS.md](docs/RESULTS.md).
- [x] Decide H1/H2/H3 and D6 under the predeclared rules. H1 supported; H2 inconclusive/not needed; H3 supported for both heads; D6 not triggered. [Decision artifact](runs/m2-comparison-s0/result.json)

### M3. Serve and compare with Jev

PointerHead is the default selected by M2: it is simpler than SetHead, retained near-kev accuracy, and SetHead showed no supported gain. [M2 evidence](runs/m2-comparison-s0/result.json)

- [ ] `hev/serve.py`: FastAPI `POST /v1/systemone` plus `/v1/models`, same shapes as kev so the TypeSafe SDK and kev's playground work with a base_url change.
- [ ] Port kev's Jev client if a live three-way comparison is wanted (needs a TypeSafe key via AI Gateway; kev/jev.py).
- [ ] Replicate PointerHead with a second predeclared seed. Report both seeds, never only the better one.

### M4. Beyond the first result (pick after M2)

Candidates, in rough order of value:
- Extract question type: pointer over state spans, still no decoding. A capability Jev does not have.
- Dependent questions: a declared DAG where question B's branch may read question A's decide token.
- Calibration that transfers: per-question learned temperature or evidential head, scored on transfer-v2.
- Kev's v3/v4 compositional suites (kev `evals/v3`, `evals/v4`) for a harder transfer test.

## Not doing

- Not re-implementing dataset download and suite freezing until a new suite is needed.
- Not training on Qwen2.5-0.5B. Kev moved its baseline to Qwen3-0.6B-Base; comparisons use that.
- Not claiming anything about Jev's internals. Zero flips in Jev do not imply this architecture.
