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

### M3. Serve and replicate PointerHead  (done 2026-09-20)

Exit met by the TypeSafe-compatible server and the immutable [seed-1 result](runs/m3-pointer-s1/result.json) plus [two-seed aggregate](runs/m3-pointer-replication-v2/result.json). PointerHead averaged 80.00% decision-v2 accuracy and 60.71% transfer-v2 accuracy across seeds 0 and 1; both seeds passed every isolation/order mechanism check. Seed 1 minus seed 0 was −1.33 points on decision-v2 (paired 95% CI −3.08 to +0.42) and −0.71 points on transfer-v2 (−3.39 to +1.96), so the observed accuracy difference is not distinguishable from zero under the predeclared descriptive comparison. [Replication results](docs/RESULTS.md)

The optional live comparison is also complete against a direct TypeSafe snapshot resolving to `jev-1.13.0`. On the same development populations, two-seed means were Hev 80.00% / 60.71%, kev 80.46% / 62.05%, and Jev 83.50% / 85.36% for decision-v2 / transfer-v2. Paired Jev-minus-Hev accuracy intervals excluded zero for each Hev seed; kev remains point-only because its checked-in v2 artifacts have no rows. Hev alone had zero flips in its exhaustive six-order mechanism study; the smaller common clean-versus-one-permutation variant observed 4.17–5.56% kev decision flips and 1.39% Jev decision flips. Jev NLL is floor-sensitive because its API rounds probabilities to zero. [Comprehensive three-way artifact](runs/m3-three-way-v2-r1/result.json) and [interpretation](docs/RESULTS.md)

PointerHead is the default selected by M2: it is simpler than SetHead, retained near-kev accuracy, and SetHead showed no supported gain. [M2 evidence](runs/m2-comparison-s0/result.json)

### R1. Open-source research preview  (prepared 2026-09-20)

The local `v0.1.0` release is prepared as a development-only research preview: Apache-2.0 license and third-party notices, model card, citation metadata, offline CI, source/wheel builds, local-or-Hub checkpoint loading, and provenance-gated Hub publication tooling. Both predeclared PointerHead checkpoints are prepared as `seed-0` and `seed-1` revisions in one future model repository; neither is labelled best. The locked test split remains untouched. No Git remote, public repository, checkpoint upload or release tag was created during preparation.

Publication validation: `116 passed, 5 skipped`; sdist and wheel built; both immutable PointerHead runs passed no-network publication dry runs. Public namespace selection, repository creation, uploads and release tagging remain owner actions.

Predeclared replication protocol (2026-09-20, before the run): train PointerHead with seed 1 and the exact M2 recipe (2 epochs, r=16, lr 2e-4, effective batch 8, Qwen3-0.6B-Base at the suite-pinned revision, and the same augmentation probabilities). Evaluate decision-v2 and transfer-v2 development with evaluation seed 1, six permutations, 10,000 bootstrap draws, and no test access. The immutable model path is `runs/m3-pointer-s1`. The aggregate reports seed 0 and seed 1 separately, their arithmetic mean and range, and paired seed-1-minus-seed-0 intervals as descriptive variability. Replication mechanism checks are complete coverage, zero Choice argmax flips, p90 probability spread at most `1e-4`, and packed-vs-separate maximum difference at most `1e-4`; there is no accuracy-based seed-selection rule. The first aggregate attempt was retained as a failure because `evaluate.py` gained result-protocol metadata after seed 0. The successful `runs/m3-pointer-replication-v2` artifact admits only the two audited hashes, records that exception, and verifies identical model, data, suite, and training source hashes.

- [x] `hev/serve.py`: FastAPI `POST /v1/systemone` plus `/v1/models`, same shapes as kev so the TypeSafe SDK and kev's playground work with a base_url change.
- [x] Replicate PointerHead with predeclared seed 1. Report both seeds, never only the better one. [Aggregate](runs/m3-pointer-replication-v2/result.json)
- [x] Direct TypeSafe evaluation of `jev-1.13.0` on decision-v2 calibration/development and transfer-v2 development, followed by a comprehensive Hev / kev / Jev aggregate. [Three-way result](runs/m3-three-way-v2-r1/result.json)

### M4. Released kev-0.6b on v4 development  (done 2026-09-20)

Step 1 of the controlled comparison: evaluation only, no training. Protocol fixed in [D13](docs/DECISIONS.md). kev at HEAD 20fa626 ships its own `option_isolation` flag and a published 0.6B preview trained on decision-v4, so the earlier framing that only Hev has exact order invariance is out of date; see [docs/KEV.md](docs/KEV.md).

Exit met by the immutable [aggregate](runs/m4-released-v4-r2/result.json). The predictor sanity gate passed at exactly zero difference: re-scoring the released checkpoint reproduced kev's own published development accuracy, 0.8045886076 over 1,264 decision rows and 0.5975609756 over 656 transfer rows. On the primary public-source population kev led Hev seed 0 by 1.63 points on decision with a 95% interval of -0.38 to +3.75 (inconclusive) and seed 1 by 3.85 points with an interval of +1.73 to +5.96 (kev better). Transfer was inconclusive for both seeds. Under the exhaustive six-order protocol the released kev flipped 6.85% of 628 decision questions and 25.00% of 348 transfer questions, while both Hev seeds flipped none. kev's own one-permutation protocol reported 1.67% on the same checkpoint, so that cheaper protocol understates flips roughly fourfold. Full tables: [docs/RESULTS.md](docs/RESULTS.md).

- [x] Copy `decision-v4` and `transfer-v4` byte-for-byte from kev 20fa626 (`evals/`), development-only.
- [x] Vendored kev inference path (`hev/kev_model.py`) and `KevPredictor` (`hev/kev.py`) so a kev checkpoint runs under kev's exact packing through Hev's evaluator.
- [x] Evaluator seams: `--out`, `--checkpoint-kind kev`, `--allow-cross-suite`, each recorded in `result.json`; transfer-v4 holdout sources accepted as eval-only.
- [x] Evaluate released kev-0.6b, Hev seed 0, Hev seed 1 on decision-v4 and transfer-v4 development; predictor sanity gate against kev's own numbers. [kev](runs/m4-kev-0.6b-v4-r1/result.json), [seed 0](runs/m4-hev-pointer-s0-v4/result.json), [seed 1](runs/m4-hev-pointer-s1-v4/result.json)
- [x] Paired aggregate with predeclared equivalence margins, primary population = public sources. [Aggregate](runs/m4-released-v4-r2/result.json)
- [x] RESULTS.md, README results table, KEV.md, model card updated with artifact links.

Two earlier aggregate attempts are retained as failures and must never be reused. `runs/m4-released-v4` stopped because the aggregate validated each run's training suite hash instead of the suite actually evaluated, which a cross-suite Hev run correctly reports as decision-v2. `runs/m4-released-v4-r1` stopped because the evaluator did not record the scored kev checkpoint's architecture flags, so the artifact could not show that kev ran with `option_isolation` false. Both were code faults, not model results. The kev evaluation was then repeated as `runs/m4-kev-0.6b-v4-r1` with the flags recorded; it reproduced `runs/m4-kev-0.6b-v4` exactly, including every permutation statistic, and the first kev run is retained.

Step 2 is now M4b.

### M4b. Controlled retrain on decision-v4  (done 2026-09-21)

Step 2 of the controlled comparison: train Hev's PointerHead on kev's own training partition under kev's published v4 recipe, three predeclared seeds, so the D14 comparison no longer confounds architecture with training data. Protocol fixed in [D15](docs/DECISIONS.md). Precision and hardware still differ and always will on this machine, so this removes one confound, not all of them.

- [x] Fetch `decision-v4/train.jsonl` (10,896 records, sha256 `cb55b79e…`) from kev's pinned Hub mirror, verified against the manifest, git-ignored (`hev/suite.py`, opt-in `fetch=True`).
- [x] Read the trainable/eval-only policy from the suite manifest so v4's `legacy_policy` and `compositional` arms may train while `contrastive` may not, without changing decision-v2's policy (`hev/data.py::source_policy`).
- [x] Wire kev's none-pair minimal-pair augmentation into the loop with kev's accumulation weighting (`--p-none-pair`).
- [x] Record the deviations Hev cannot remove instead of pretending to match: `--recipe kev-v4` matches every shared knob or refuses to start, and writes precision, hardware, micro-batching and encoding differences into `training_config.json`.
- [x] Train seeds 0, 1 and 2, 2,724 optimizer steps each, all passing the loss-decrease and provenance gates. [seed 0](runs/m4b-eval-v4-s0/result.json), [seed 1](runs/m4b-eval-v4-s1/result.json), [seed 2](runs/m4b-eval-v4-s2/result.json). Seed 0 lives at `-s0-r1`: the predeclared directory is a killed launch that never trained (D15 addendum).
- [x] Evaluate each seed on decision-v4 and transfer-v4 development under D13's evaluation settings. [seed 0](runs/m4b-eval-v4-s0/result.json), [seed 1](runs/m4b-eval-v4-s1/result.json), [seed 2](runs/m4b-eval-v4-s2/result.json)
- [x] Three-seed aggregate against `runs/m4-kev-0.6b-v4-r1` with D13's populations and margins (`hev/controlled.py`), then RESULTS.md, README, KEV.md and the model card. [Aggregate](runs/m4b-v4-three-seed/result.json)

Exit met 2026-09-21 by the immutable [three-seed aggregate](runs/m4b-v4-three-seed/result.json). Sanity gate passed at zero difference; every seed passed the mechanism checks with zero flips. On the primary decision population kev leads by 0.67 points at every seed with every 95% interval crossing zero and the 90% upper bounds at 2.21 / 2.02 / 2.02 against the 2.0 margin, so all three are inconclusive rather than equivalent. On primary transfer kev leads by 2.71 / 5.42 / 1.46 points, kev-better at seed 1 and inconclusive at the others; three-seed mean +3.19. Calibration is comparable. The M4 step 1 decision gap was mostly data, as D14 expected; the transfer gap is consistent with kev's own isolation measurement. Precision and hardware still differ. Outcome recorded in [D16](docs/DECISIONS.md); tables in [docs/RESULTS.md](docs/RESULTS.md).

### M5. Beyond the first result (pick after M4)

Candidates, in rough order of value:
- Extract question type: pointer over state spans, still no decoding. A capability Jev does not have.
- Dependent questions: a declared DAG where question B's branch may read question A's decide token.
- Calibration that transfers: per-question learned temperature or evidential head, scored on transfer-v2.
- Kev's v3/v4 compositional suites (kev `evals/v3`, `evals/v4`) for a harder transfer test.

## Not doing

- Not re-implementing dataset download and suite freezing until a new suite is needed.
- Not training on Qwen2.5-0.5B. Kev moved its baseline to Qwen3-0.6B-Base; comparisons use that.
- Not claiming anything about Jev's internals. Zero flips in Jev do not imply this architecture.
