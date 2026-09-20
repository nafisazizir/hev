# Results

## M3: comprehensive Hev / kev / Jev comparison  (2026-09-20)

M3 now includes a same-suite comparison of the selected two-seed Hev PointerHead, the two checked-in kev Qwen3-0.6B seeds, and a live TypeSafe direct-API snapshot resolving consistently to `jev-1.13.0`. All models were scored on the exact checksum-matched `decision-v2` and `transfer-v2` development populations; Jev's temperature was fit only on `decision-v2` calibration and then applied unchanged. The locked test split was not accessed. The authoritative ledger is the [three-way result](../runs/m3-three-way-v2-r1/result.json); the live source ledgers are [Jev calibration](../runs/m3-jev-direct-calibration-v2-r1/result.json), [decision](../runs/m3-jev-direct-decision-v2/result.json), and [transfer](../runs/m3-jev-direct-transfer-v2/result.json).

### Headline comparison

Hev and kev values below are two-seed means with seed ranges; Jev is one hosted snapshot, not a seed average. Calibration preserves accuracy. Kev's checked-in v2 artifacts do not report calibrated transfer metrics.

| Model | Decision accuracy | Transfer accuracy | Decision raw NLL | Decision calibrated NLL / ECE / Brier | Transfer raw NLL |
|---|---:|---:|---:|---:|---:|
| Hev PointerHead | 80.00% (79.33–80.67) | 60.71% (60.36–61.07) | 0.606 | 0.502 / 0.020 / 0.269 | 0.989 |
| kev Qwen3-0.6B | 80.46% (79.33–81.58) | 62.05% (61.96–62.14) | 0.604 | 0.511 / 0.030 / 0.272 | 0.980 |
| Jev `1.13.0` | **83.50%** | **85.36%** | 0.813* | 0.544 / 0.103 / 0.274 | 0.782* |
| Hev SetHead, seed 0 ablation | 80.92% | 60.71% | 0.582 | 0.486 / 0.022 / 0.260 | 0.950 |

`*` Jev returns rounded probabilities, including exact zeros. Its reported NLL uses the common `1e-9` floor and is therefore floor-sensitive; accuracy, Brier, and ECE do not have that same logarithmic sensitivity. [Metric policy and floor analysis](../runs/m3-three-way-v2-r1/result.json)

Jev minus Hev accuracy was paired on identical examples with 10,000 source-stratified, record-clustered bootstrap draws. Temperature scaling does not change these accuracy differences.

| Hev target | Decision Jev − Hev (95% CI) | Transfer Jev − Hev (95% CI) |
|---|---:|---:|
| PointerHead seed 0 | +2.83 pp (+0.67, +5.00) | +24.29 pp (+20.36, +28.39) |
| PointerHead seed 1 | +4.17 pp (+2.00, +6.33) | +25.00 pp (+21.07, +28.93) |
| SetHead seed 0 | +2.58 pp (+0.42, +4.75) | +24.64 pp (+20.71, +28.57) |

These intervals quantify example-level uncertainty for each fixed model instance; they are not a Jev training-seed variance estimate. Kev's checked-in v2 artifacts contain no per-example rows, so only point differences are possible: Hev Pointer minus seed-matched kev was −0.92 / 0.00 pp on decision and −0.89 / −1.79 pp on transfer, while Jev minus kev was +1.92 / +4.17 pp on decision and +23.39 / +23.21 pp on transfer. [Pairwise evidence](../runs/m3-three-way-v2-r1/result.json)

### Where the difference comes from

Jev led every transfer task in the seed-0 Hev comparison. The largest gaps were MMLU (90.0% versus 36.3%), held-out authorization (100.0% versus 57.5%), and held-out deadline (92.5% versus 25.0%); it also led emotion, PAWS, QNLI, SciQ, and tweet-offensive. On familiar decision tasks the result was mixed but favored Jev overall: notable gains included BoolQ (92.5% versus Hev's 77.5% / 68.8%), MNLI (90.0% versus 75.0% / 70.0%), and Banking77 (82.5% versus 76.3% / 68.8%). Full per-task, per-source, and—where rows exist—per-type tables are embedded in the aggregate. Kev per-type metrics are unavailable because its v2 ledgers omit both rows and a type summary. [Task evidence](../runs/m3-three-way-v2-r1/result.json)

### Calibration and rounded-zero sensitivity

Jev's calibration-only macro-task NLL selected `T=3.364`, improving calibration macro-task NLL from 0.976 to 0.629. On decision development this reduced overall NLL from 0.813 to 0.544, but worsened ECE from 0.070 to 0.103 and Brier from 0.256 to 0.274. On transfer it reduced NLL from 0.782 to 0.533 while worsening ECE from 0.056 to 0.141 and Brier from 0.221 to 0.267. Temperature fitting optimizes calibration macro-task NLL, not ECE or Brier, and transfer calibration remains descriptive. Hev and kev had materially lower calibrated decision ECE at 0.020 and 0.030 respectively. [Calibration evidence](../runs/m3-three-way-v2-r1/result.json)

| Jev NLL floor | Decision NLL | Transfer NLL |
|---|---:|---:|
| `1e-3` | 0.525 | 0.462 |
| `1e-6` | 0.669 | 0.622 |
| `1e-9` | 0.813 | 0.782 |

### Option order

The protocols differ and must not be collapsed into one ranking. Hev's architectural study evaluated six orders for every eligible Choice question: both Pointer seeds and SetHead had zero flips over 696 decision and 348 transfer questions, with decision p90 correct-probability spread at most `1.45e-6`. The frozen common variant gives only clean versus one permutation on 72 decision and 36 transfer questions: kev flipped 5.56% / 4.17% on decision and 8.33% on transfer in both seeds; Jev flipped 1.39% on decision and 0% on transfer, while its maximum probability movement was 0.456 and 0.200. Jev's zero observed transfer flips does not prove architectural invariance. [Order evidence](../runs/m3-three-way-v2-r1/result.json)

### Conclusion and scope

On these development suites, Hev reproduces kev-level accuracy while uniquely delivering the tested exact option-order behavior. Jev is more accurate—especially on transfer—but its returned probabilities are coarsely rounded, its NLL is floor-sensitive, and decision-v2 temperature scaling did not improve ECE or Brier. SetHead remains an unsupported complexity increase: its seed-0 accuracy does not close the Jev gap and M2 found no paired advantage over PointerHead. This is a same-evaluation comparison, not a controlled training comparison: the systems differ in architecture, training data, compute, and availability. Jev is a hosted version snapshot rather than a reproducible checkpoint. The direct evaluation used 2,292 calls, 1,097,597 input tokens, and an estimated **$0.0461**. [Protocol, provenance, and caveats](../runs/m3-three-way-v2-r1/result.json)

## M3: PointerHead replication  (2026-09-20)

M3 repeated the selected PointerHead with predeclared training seed 1 and the unchanged M2 recipe, then evaluated the same decision-v2 and transfer-v2 development populations. The locked test split was not accessed. The authoritative artifacts are the [seed-1 result](../runs/m3-pointer-s1/result.json) and [two-seed aggregate](../runs/m3-pointer-replication-v2/result.json).

| Metric | Seed 0 | Seed 1 | Two-seed mean (range) |
|---|---:|---:|---:|
| Decision accuracy | 80.67% | 79.33% | 80.00% (79.33–80.67) |
| Transfer accuracy | 61.07% | 60.36% | 60.71% (60.36–61.07) |
| Decision calibrated ECE | 0.0148 | 0.0256 | 0.0202 (0.0148–0.0256) |
| Decision calibrated Brier | 0.2636 | 0.2754 | 0.2695 (0.2636–0.2754) |

The paired source-stratified, record-clustered seed-1-minus-seed-0 accuracy difference was −1.33 percentage points on decision-v2 (95% CI −3.08 to +0.42) and −0.71 points on transfer-v2 (−3.39 to +1.96). These are descriptive intervals over examples, not estimates from a population of two training seeds. They do not support selecting the better seed or a claim that seed 1 is less accurate. [Paired artifact](../runs/m3-pointer-replication-v2/result.json)

Seed 1 had zero Choice argmax flips on both suites. Its p90 correct-probability spread was `1.45e-6` on decision-v2 and `2.86e-6` on transfer-v2; its packed-vs-separate maximum was `6.38e-6` on decision-v2, while transfer records are single-question. All were below the predeclared `1e-4` mechanism tolerance, and coverage was complete. [Seed-1 diagnostics](../runs/m3-pointer-s1/result.json)

The first aggregate attempt correctly stopped because `evaluate.py` had changed after seed 0. Audit showed that the sole difference adds `evaluation_config` fields to `result.json` after predictions and metrics are computed. The final aggregate accepts exactly the recorded old and new hashes, preserves that exception in provenance, and still requires identical `data.py`, `model.py`, `suite.py`, and `train.py` hashes. [Provenance record](../runs/m3-pointer-replication-v2/result.json)

M3 therefore strengthens the development-only result: PointerHead's accuracy and exact-by-construction order behavior reproduced under a second training seed. It does not convert the result into a locked-test claim or a broad estimate of training-seed variance.

## M2: first real comparison  (2026-09-20)

M2 trained exactly the two predeclared seed-0 models on decision-v2 and evaluated them on clean decision-v2 and transfer-v2 development rows. Temperature scaling was fit only on decision-v2 calibration and transferred unchanged. The locked test split was not accessed. The authoritative aggregate artifact is [`runs/m2-comparison-s0/result.json`](../runs/m2-comparison-s0/result.json); the full PointerHead and SetHead ledgers are [`runs/m2-pointer-s0/result.json`](../runs/m2-pointer-s0/result.json) and [`runs/m2-set-s0/result.json`](../runs/m2-set-s0/result.json).

### Headline results

Accuracy intervals are 95% source-stratified, `(source, group_id)`-clustered bootstrap intervals with 10,000 draws. ECE and Brier are decision-v2 values after calibration. Kev values are hash-verified point baselines only because its checked-in artifacts do not contain per-example rows. [Protocol evidence](../runs/m2-comparison-s0/result.json)

| Model | Decision accuracy (95% CI) | Transfer accuracy (95% CI) | Decision calibrated ECE | Decision calibrated Brier | Evidence |
|---|---:|---:|---:|---:|---|
| hev PointerHead, seed 0 | 80.67% (78.33–82.92) | 61.07% (57.50–64.64) | 0.0148 | 0.2636 | [artifact](../runs/m2-comparison-s0/result.json) |
| hev SetHead, seed 0 | 80.92% (78.58–83.25) | 60.71% (57.14–64.29) | 0.0223 | 0.2603 | [artifact](../runs/m2-comparison-s0/result.json) |
| kev seed 0 | 81.58% | 61.96% | 0.0264 | 0.2595 | [hash-verified source in artifact](../runs/m2-comparison-s0/result.json) |
| kev seed 1 | 79.33% | 62.14% | 0.0339 | 0.2846 | [hash-verified source in artifact](../runs/m2-comparison-s0/result.json) |

PointerHead was 0.92 percentage points below kev seed 0 on decision-v2 and 0.89 points below it on transfer-v2. SetHead was 0.25 points above PointerHead on decision-v2, but the paired 95% CI was −1.17 to +1.58 points; on transfer it was 0.36 points lower, with a paired interval of −3.21 to +2.50 points. M2 therefore provides no evidence that the more complex SetHead improves accuracy. [Artifact](../runs/m2-comparison-s0/result.json)

### Order and packing diagnostics

Both heads had zero Choice argmax flips across the six evaluated orders on both suites. The remaining probability differences are fp32 numerical residuals far below the predeclared `1e-4` tolerance. [Diagnostic evidence](../runs/m2-comparison-s0/result.json)

| Head | Suite | Argmax flip rate | p90 correct-probability spread | Packed/separate max difference | Evidence |
|---|---|---:|---:|---:|---|
| Pointer | decision-v2 | 0 | 1.25e-6 | 6.20e-6 | [artifact](../runs/m2-comparison-s0/result.json) |
| Pointer | transfer-v2 | 0 | 2.74e-6 | n/a: transfer records are single-question | [artifact](../runs/m2-comparison-s0/result.json) |
| Set | decision-v2 | 0 | 1.13e-6 | 6.23e-6 | [artifact](../runs/m2-comparison-s0/result.json) |
| Set | transfer-v2 | 0 | 3.04e-6 | n/a: transfer records are single-question | [artifact](../runs/m2-comparison-s0/result.json) |

### Predeclared decisions

- **H1 supported.** PointerHead passed all eight predeclared order, packing, and accuracy checks on both development suites. This supports option-level isolation as an effective way to remove observed option-order sensitivity at a small seed-0 accuracy cost. [Decision artifact](../runs/m2-comparison-s0/result.json)
- **H2 inconclusive / not needed.** PointerHead's decision-v2 loss to kev seed 0 was 0.92 points, below the predeclared one-point material-loss threshold. SetHead's small point improvement was not statistically distinguishable from zero. [Decision artifact](../runs/m2-comparison-s0/result.json)
- **H3 supported for both heads.** PointerHead and SetHead both passed the predeclared calibrated ECE and Brier non-inferiority thresholds on decision-v2. Transfer calibration remains descriptive because kev has no matching calibrated transfer artifact. [Decision artifact](../runs/m2-comparison-s0/result.json)
- **The Score level embedding is used but not proven useful.** Zeroing it changed a Score probability by as much as 0.0286, but learned-minus-zeroed NLL was −0.00030 with a 95% CI of −0.00303 to +0.00233. [Pointer ablation artifact](../runs/m2-pointer-s0/result.json)
- **D6 was not triggered.** PointerHead and SetHead achieved 76.25% and 75.00% on clean banking77, both above the predeclared 71.25% fallback threshold. [Decision artifact](../runs/m2-comparison-s0/result.json)

### Conclusion

On this predeclared seed-0 comparison, option-isolated PointerHead retains essentially all of kev's accuracy while eliminating observed option-order flips. SetHead adds listwise interaction but shows no supported gain, so PointerHead is the preferred default for serving and second-seed replication. The current level embedding can remain in the checkpoint, but M2 does not justify claiming that it improves ordinal prediction. These conclusions are development-only and should not be upgraded to final test-set or multi-seed claims without a new predeclared protocol. [Aggregate evidence](../runs/m2-comparison-s0/result.json)
