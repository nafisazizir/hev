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

### M1. Training loop and smoke run

Exit: a `runs/smoke-*/` directory with a trained adapter on `evals/smoke-v1`, loss decreasing, and an eval JSON with accuracy and flip rate on smoke development.

- [ ] `hev/train.py`: LoRA + head + level embedding; cross-entropy; optional ranked-probability term for Score; refuses eval-only sources; refuses existing run dir; writes `training_config.json` and `training_metrics.json`. Model kev's loop, not copy it.
- [ ] `hev/evaluate.py`: accuracy, NLL, Brier, ECE (10 bins on top prob), per source and per type; permutation study (flip rate, prob spread of correct option) on Choice; packed-vs-separate equality. Port the metric functions from kev/evaluate.py, keep the same definitions so numbers are comparable.
- [ ] Hub test: real Qwen3-0.6B-Base tokenizer round-trips a decision-v2 development record within kev's context limits (marked `hub`).
- [ ] Smoke run on MPS, both heads. Record wall time and peak memory.

### M2. First real comparison

Exit: hev-pointer and hev-set trained on decision-v2 train, evaluated on decision-v2 development and transfer-v2 development, with kev's numbers side by side and bootstrap CIs.

- [ ] Train both heads, seed 0, kev's v2 recipe (2 epochs, r=16, lr 2e-4, effective batch 8, Qwen3-0.6B-Base pinned to the suite's revision).
- [ ] Temperature fit on calibration only, applied unchanged to transfer.
- [ ] Ablation: `PointerHead` with Score level embedding zeroed, to confirm the embedding is doing work on ordinal tasks.
- [ ] Write results to `runs/<name>/result.json` and summarise in docs/RESULTS.md with links.
- [ ] Decide: does H1 hold? Does H2 hold? If accuracy loss is large (more than about 5 points) on high-K sources like banking77, look at DECISIONS.md D6 for the fallback.

### M3. Serve and compare with Jev

- [ ] `hev/serve.py`: FastAPI `POST /v1/systemone` plus `/v1/models`, same shapes as kev so the TypeSafe SDK and kev's playground work with a base_url change.
- [ ] Port kev's Jev client if a live three-way comparison is wanted (needs a TypeSafe key via AI Gateway; kev/jev.py).
- [ ] Second seed for whichever configuration looks best. Never report only the better seed.

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
