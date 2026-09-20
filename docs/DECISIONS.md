# Decisions

Append-only log. Newest at the bottom. Each entry: what was decided, why, what was rejected.

## D1. Start fresh rather than fork kev  (2026-09-19)

Own repo and own model code. Kev's eval suites and API contract are copied in from day one; other kev pieces are ported only when a milestone needs them.

Why: the model, mask and readout are the whole point of hev, and writing them clean avoids fighting kev's assumptions when the packing differs. Copying suites and schema immediately means every hev number is comparable to kev and Jev from the first run.

Rejected: forking kev and swapping model.py. Kev's train.py, evaluate.py and serve.py all call kev's `encode` and expect its checkpoint layout, so the swap would touch most files anyway.

## D2. Option-level isolation as the first experiment  (2026-09-19)

Why: it attacks kev's largest measured weakness (7% argmax flips, p90 spread 0.25), kev's PLAN.md explicitly lists "option-order architecture experiments" as deferred, the metric already exists in kev's evaluator, and the result is falsifiable either way.

Rejected for now: extract question type, dependent-question DAG, transfer-calibration heads. All listed in PLAN.md M4.

## D3. Same delimiters, limits, schema and augmentation as kev  (2026-09-19)

Why: identical packed token counts mean kev's context-limit admission carries over, so every record in the frozen suites is admissible in hev without re-freezing. Identical augmentation means a difference in results is attributable to the architecture, not the data pipeline.

## D4. Two readout heads, pointer first  (2026-09-19)

PointerHead is the null-interaction control. SetHead is the listwise, positionless alternative. Both are equivariant by construction. Train and report both.

Why: if pointer alone matches kev, the story is simple. If it does not and set recovers it, that isolates "interaction helps, order does not". If neither matches, H1 fails honestly.

Rejected: a single "best guess" head. Two heads cost one extra training run and turn a result into an explanation.

## D5. Learned level embedding for score questions only  (2026-09-19)

Why: ordinal levels are ordered by definition. Without any index signal the model can only infer order from level text, which is fragile. Zero-initialised so invariance is the default.

Rejected: injecting the index into the backbone (as a token or position offset). That would reintroduce order sensitivity inside the backbone, where it is hardest to bound. Readout-only keeps the backbone invariant for every type.

## D6. Fallback if pointer loses badly on high-K choice  (2026-09-19, provisional)

If banking77-style tasks drop by more than about 5 points with both heads, the fallback is a two-stage readout: independent option encoding as now, then a top-M shortlist re-scored by a positionless set layer over only those M. Still equivariant; recovers deeper comparison among plausible options. Not to be built until M2 shows it is needed.

## D7. Tests run offline against a tiny random backbone  (2026-09-19)

Why: the architectural claims (mask rule, invariance, isolation, packed equals separate) are properties of the code, not of trained weights, and must be checkable on any machine in seconds. Hub tests exist but are gated.

## D8. Qwen3-0.6B-Base is the backbone for comparisons  (2026-09-19)

Why: kev moved its research baseline there (81.6/79.3 dev vs 74.2/65.8 for Qwen2.5-0.5B on the same recipe), and the suites pin its revision. The released kev-0.5b checkpoint is a historical reference only.

## D9. M2 predeclared evaluation protocol  (2026-09-19, predeclared before training)

Recorded before any M2 training run. These rules are fixed; outcomes are appended in a later entry without editing this one.

**Runs.** Exactly two seed-0 training runs on the M4 Max, sequential, local MPS, fp32, eager attention, pinned `Qwen/Qwen3-0.6B-Base` revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, frozen decision-v2 train partition (manifest SHA-256 `e0388e1d13284f5e0a2e6cb33a0fcd34f0099bfc31a6a852bd5cb619f505094d`), kev v2 recipe: LoRA r=16, lr 2e-4, effective batch 8, 2 epochs, augmentation p_none 0.1 / p_none_distract 0.12 / p_distract 0.15. `runs/m2-pointer-s0` (PointerHead) and `runs/m2-set-s0` (SetHead), both with the learned Score level embedding. `runs/m2-comparison-s0` is a lightweight aggregate comparison; no third checkpoint exists.

**Evaluation.** Each checkpoint is evaluated once on decision-v2 development and transfer-v2 development (manifest SHA-256 `a9dd4485e3247014209ecbee28d9fe02d0556c69e95dc30cca0584d1e5c27e35`). The pointer checkpoint is additionally re-evaluated on clean Score questions with the learned level embedding temporarily zeroed at eval time and restored; it is not retrained. No test split access anywhere.

**Headline metrics are clean-only.** `variant == "clean"` rows only enter headline accuracy/NLL/Brier/ECE; none_present, none_absent and permuted variants are diagnostic reports, never headline. Temperature is fit once on decision-v2 calibration (clean rows, unweighted macro-task NLL, 81-point log grid [0.25, 4.0]) and applied unchanged to decision development, transfer development, and the ablation; transfer never fits its own.

**Uncertainty.** Deterministic percentile bootstrap: 10,000 draws, fixed seed 20260919, clustered on `(source, group_id)`, stratified by source (resample each source's clusters with replacement, keep every row in a sampled cluster). Intervals cover hev estimates and paired `set - pointer` deltas only. Kev seed-0/seed-1 results (`runs/ablation-v2/06-trial-6/result.json` and `07-trial-7/result.json`, hash-verified) are point baselines only — kev v2 artifacts have no per-example rows, so no paired hev-vs-kev CI is claimed.

**H1: option isolation removes order sensitivity at acceptable accuracy cost.** Supported only if, for PointerHead on both decision-v2 and transfer-v2 development: (1) Choice K>=3 argmax flip rate is exactly 0 across six deterministic orders; (2) p90 correct-option probability spread <= 1e-4; (3) packed-vs-separate max absolute probability difference <= 1e-4 where applicable; (4) clean micro accuracy >= 0.765833 on decision-v2 and >= 0.569643 on transfer-v2 (no more than 5 points below kev seed 0). The kev threshold is a fixed sourced point, not an interval.

**H2: SetHead recovers most of a material PointerHead loss.** Decision-v2 is primary; transfer-v2 is a secondary readout and cannot override it. With `L = kev_seed0_accuracy - pointer_accuracy` and `R = set_accuracy - pointer_accuracy` on clean decision-v2: if `L <= 0.01`, H2 is not needed / inconclusive; if `L > 0.01`, supported when `R / L >= 0.5` and the paired grouped 95% CI for `set - pointer` accuracy has lower bound > 0; contradicted when SetHead does not improve the point estimate or recovers <50%; if recovery >=50% but the CI crosses 0, inconclusive.

**H3: calibration is not materially worse than kev.** Per head on decision-v2: calibrated ECE <= 0.0464021 and calibrated Brier <= 0.2794607 (kev seed-0 values + 0.02 non-inferiority margin). Transfer calibration comparison to kev is descriptive; no calibrated transfer kev baseline exists in the checked-in artifact.

**Score level-embedding ablation.** The embedding is used if the maximum Score probability change under zeroing exceeds 1e-4; useful if learned-level Score NLL is lower and the paired grouped 95% CI for `(learned - zeroed)` NLL has upper bound < 0; otherwise learned-but-not-proven-useful or inconclusive, with no reinterpretation after results.

**D6 trigger.** Recommend pursuing D6 only if both hev heads score below 0.7125 clean accuracy on banking77 (matched kev seed-0 0.7625 minus 5 points). M2 records the trigger outcome; it does not implement the fallback.

## D10. M2 selects PointerHead as the default  (2026-09-20)

M2 supports H1 and H3 under D9. H2 is inconclusive/not needed because PointerHead's decision-v2 loss to kev seed 0 was 0.92 percentage points, below the predeclared material-loss threshold. SetHead's decision improvement was 0.25 points with a paired 95% CI of −1.17 to +1.58 points, and it was 0.36 points lower on transfer. [Aggregate comparison](../runs/m2-comparison-s0/result.json)

Use PointerHead as the default for M3 serving and second-seed replication. It is the simpler readout, passed all H1 checks, and SetHead supplied no supported accuracy gain. Retain SetHead as a research alternative rather than claiming it is worse. [Results](RESULTS.md), [decision artifact](../runs/m2-comparison-s0/result.json)

Do not build the D6 shortlist fallback: both heads remained above its banking77 threshold. Keep the existing Score level embedding in the M2 checkpoint, but do not claim it is beneficial; the ablation shows that it changes probabilities without a statistically supported NLL improvement. [Pointer ablation](../runs/m2-pointer-s0/result.json), [D6 outcome](../runs/m2-comparison-s0/result.json)

## D11. Keep PointerHead as the served default after replication  (2026-09-20)

The predeclared seed-1 replication reached 79.33% decision-v2 and 60.36% transfer-v2 accuracy, versus 80.67% and 61.07% for seed 0. Both paired accuracy intervals crossed zero, and both seeds passed every order, packing, and coverage check. Keep PointerHead as the default and report the two-seed mean and range rather than selecting seed 0 as the headline model. [Replication aggregate](../runs/m3-pointer-replication-v2/result.json)

Serving applies each run's calibration-only fitted temperature and exposes the TypeSafe-compatible `/v1/systemone` and `/v1/models` shapes. A live Jev comparison remains optional and must not be implied without an explicit AI Gateway run artifact.
