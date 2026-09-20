# Results

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
