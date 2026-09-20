# Decisions

Append-only log. Newest at the bottom. Each entry: what was decided, why, what was rejected.

## D1. Start fresh rather than fork kev  (2026-09-19)

Own repo and own model code. Kev's eval suites and API contract are copied in from day one; other kev pieces are ported only when a milestone needs them.

Why: the model, mask and readout are the whole point of Hev, and writing them clean avoids fighting kev's assumptions when the packing differs. Copying suites and schema immediately means every Hev number is comparable to kev and Jev from the first run.

Rejected: forking kev and swapping model.py. Kev's train.py, evaluate.py and serve.py all call kev's `encode` and expect its checkpoint layout, so the swap would touch most files anyway.

## D2. Option-level isolation as the first experiment  (2026-09-19)

Why: it attacks kev's largest measured weakness (7% argmax flips, p90 spread 0.25), kev's PLAN.md explicitly lists "option-order architecture experiments" as deferred, the metric already exists in kev's evaluator, and the result is falsifiable either way.

Rejected for now: extract question type, dependent-question DAG, transfer-calibration heads. All listed in PLAN.md M4.

## D3. Same delimiters, limits, schema and augmentation as kev  (2026-09-19)

Why: identical packed token counts mean kev's context-limit admission carries over, so every record in the frozen suites is admissible in Hev without re-freezing. Identical augmentation means a difference in results is attributable to the architecture, not the data pipeline.

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

**Uncertainty.** Deterministic percentile bootstrap: 10,000 draws, fixed seed 20260919, clustered on `(source, group_id)`, stratified by source (resample each source's clusters with replacement, keep every row in a sampled cluster). Intervals cover Hev estimates and paired `set - pointer` deltas only. Kev seed-0/seed-1 results (`runs/ablation-v2/06-trial-6/result.json` and `07-trial-7/result.json`, hash-verified) are point baselines only — kev v2 artifacts have no per-example rows, so no paired Hev-vs-kev CI is claimed.

**H1: option isolation removes order sensitivity at acceptable accuracy cost.** Supported only if, for PointerHead on both decision-v2 and transfer-v2 development: (1) Choice K>=3 argmax flip rate is exactly 0 across six deterministic orders; (2) p90 correct-option probability spread <= 1e-4; (3) packed-vs-separate max absolute probability difference <= 1e-4 where applicable; (4) clean micro accuracy >= 0.765833 on decision-v2 and >= 0.569643 on transfer-v2 (no more than 5 points below kev seed 0). The kev threshold is a fixed sourced point, not an interval.

**H2: SetHead recovers most of a material PointerHead loss.** Decision-v2 is primary; transfer-v2 is a secondary readout and cannot override it. With `L = kev_seed0_accuracy - pointer_accuracy` and `R = set_accuracy - pointer_accuracy` on clean decision-v2: if `L <= 0.01`, H2 is not needed / inconclusive; if `L > 0.01`, supported when `R / L >= 0.5` and the paired grouped 95% CI for `set - pointer` accuracy has lower bound > 0; contradicted when SetHead does not improve the point estimate or recovers <50%; if recovery >=50% but the CI crosses 0, inconclusive.

**H3: calibration is not materially worse than kev.** Per head on decision-v2: calibrated ECE <= 0.0464021 and calibrated Brier <= 0.2794607 (kev seed-0 values + 0.02 non-inferiority margin). Transfer calibration comparison to kev is descriptive; no calibrated transfer kev baseline exists in the checked-in artifact.

**Score level-embedding ablation.** The embedding is used if the maximum Score probability change under zeroing exceeds 1e-4; useful if learned-level Score NLL is lower and the paired grouped 95% CI for `(learned - zeroed)` NLL has upper bound < 0; otherwise learned-but-not-proven-useful or inconclusive, with no reinterpretation after results.

**D6 trigger.** Recommend pursuing D6 only if both Hev heads score below 0.7125 clean accuracy on banking77 (matched kev seed-0 0.7625 minus 5 points). M2 records the trigger outcome; it does not implement the fallback.

## D10. M2 selects PointerHead as the default  (2026-09-20)

M2 supports H1 and H3 under D9. H2 is inconclusive/not needed because PointerHead's decision-v2 loss to kev seed 0 was 0.92 percentage points, below the predeclared material-loss threshold. SetHead's decision improvement was 0.25 points with a paired 95% CI of −1.17 to +1.58 points, and it was 0.36 points lower on transfer. [Aggregate comparison](../runs/m2-comparison-s0/result.json)

Use PointerHead as the default for M3 serving and second-seed replication. It is the simpler readout, passed all H1 checks, and SetHead supplied no supported accuracy gain. Retain SetHead as a research alternative rather than claiming it is worse. [Results](RESULTS.md), [decision artifact](../runs/m2-comparison-s0/result.json)

Do not build the D6 shortlist fallback: both heads remained above its banking77 threshold. Keep the existing Score level embedding in the M2 checkpoint, but do not claim it is beneficial; the ablation shows that it changes probabilities without a statistically supported NLL improvement. [Pointer ablation](../runs/m2-pointer-s0/result.json), [D6 outcome](../runs/m2-comparison-s0/result.json)

## D11. Keep PointerHead as the served default after replication  (2026-09-20)

The predeclared seed-1 replication reached 79.33% decision-v2 and 60.36% transfer-v2 accuracy, versus 80.67% and 61.07% for seed 0. Both paired accuracy intervals crossed zero, and both seeds passed every order, packing, and coverage check. Keep PointerHead as the default and report the two-seed mean and range rather than selecting seed 0 as the headline model. [Replication aggregate](../runs/m3-pointer-replication-v2/result.json)

Serving applies each run's calibration-only fitted temperature and exposes the TypeSafe-compatible `/v1/systemone` and `/v1/models` shapes. A live Jev comparison remains optional and must not be implied without an explicit AI Gateway run artifact.

## D12. Release a development-only research preview  (2026-09-20)

Prepare `v0.1.0` for open source without opening the locked test. The contribution is the option-isolation mechanism and its reproduced development evidence, not a production-readiness or state-of-the-art claim.

Publish both predeclared PointerHead checkpoints as separate `seed-0` and `seed-1` revisions in one future Hub repository. Do not select the higher-accuracy seed, publish SetHead as the default, or use a locked-test result to market the preview. Keep weights out of git; require successful training/evaluation status and exact provenance agreement before upload.

Keep detailed coding-agent instructions under `.devin/agent-notes/`, with only a standard root pointer. Credit TypeSafe, Archer Hume, Jared Palmer and kev, Qwen, dataset contributors, and Devin/Cognition explicitly. No affiliation or endorsement is implied.

## D13. M4 protocol: released kev-0.6b versus Hev on v4 development  (2026-09-20, predeclared before any run)

**Why.** kev moved 62 commits past the pinned cc954f2 and now ships its own `option_isolation` flag and a published 0.6B research preview trained on decision-v4. Hev's checked-in kev comparison is unpaired (kev's v2 artifacts have no rows) and uses mismatched order protocols. M4 step 1 is evaluation only: the same items, three checkpoints, one evaluator. It is a same-items, different-training comparison and must not be read as an architecture claim. See [KEV.md](KEV.md).

**Models.** Hev PointerHead seed 0 (`runs/m2-pointer-s0`) and seed 1 (`runs/m3-pointer-s1`), both trained on decision-v2 (3,432 records). Released `jaredpalmer/kev-0.6b` at Hub snapshot `83e05fabf7ef08e343bb4daf144e08777f99713d`, kev trial `v4-06b-hardened/00-trial-0`, seed 0 of 3, trained on decision-v4 (10,896 records, minimal pairs on, bf16 H100). SetHead is excluded because it is not a release candidate. No training in this step.

**Suites.** `evals/decision-v4` (manifest SHA-256 `1b33e566d114f9eafeff55b36c221fadb2a4ae358a1b9cc68006e82c7cfad8f1`) calibration and development; `evals/transfer-v4` (`31677c2256b406222e7d94ffdc0a02a70ce05746b9efe307876024c4e77291d1`) development. No test access. Copied byte-for-byte from kev HEAD 20fa626.

**Procedure.** `hev.evaluate` for all three, device mps, fp32, eager attention, evaluation seed 1, six deterministic orders, 10,000 bootstrap draws, temperature fit once per model on decision-v4 calibration clean rows and applied unchanged to transfer. kev is loaded through the vendored kev inference path (`hev/kev_model.py`, kev commit 20fa626) with `--checkpoint-kind kev`; Hev runs use `--allow-cross-suite` because their training suite is decision-v2, and the flag use is recorded in each `result.json`. Outputs: `runs/m4-kev-0.6b-v4`, `runs/m4-hev-pointer-s0-v4`, `runs/m4-hev-pointer-s1-v4`; aggregate `runs/m4-released-v4`. All immutable; failures are kept and never reused.

**Predictor sanity gate.** The kev re-evaluation must reproduce the checkpoint's own development accuracy recorded in the Hub `result.json` (clean 0.805 decision, 0.598 transfer) within 1.0 percentage point each under the same clean-row definition. If it does not, the predictor is presumed wrong: the artifact is kept as a failure, the discrepancy is explained in RESULTS.md, and no comparison is claimed until it is resolved.

**Primary and secondary populations.** Primary: the ten public sources of decision-v4 development (banking77, boolq, agnews, mnli, sst5, yelp, trec, dbpedia14, amazon, imdb) and the six public eval-only sources of transfer-v4 development. Both models trained on those decision sources and neither trained on the transfer sources. Secondary, reported separately and flagged: the policy arms (`legacy_policy`, `compositional`, `legacy_holdout`, `composition_holdout`), which kev trained on or was designed for and Hev never saw.

**Statistics.** Paired on identical rows, source-stratified, `(source, group_id)`-clustered percentile bootstrap, 10,000 draws, seed 20260920. Deltas are kev minus Hev for each Hev seed separately; the two-seed mean is reported as the mean of the two point deltas with both intervals, never as a pooled interval. Decision headline metrics are clean micro accuracy and calibrated NLL, ECE, Brier; transfer headline metrics are accuracy and raw NLL, Brier. Per-source, per-task, per-type, none-option and contrastive-pair tables are descriptive.

**Decision rules.** Margins are fixed now: 2.0 points on decision accuracy, 3.0 points on transfer accuracy. "Equivalent" requires the 90% paired interval to lie entirely inside the margin (two one-sided tests at 5%). "kev better" or "Hev better" requires the 95% interval to exclude zero. Anything else is inconclusive. Rules apply to the primary population; secondary results carry no verdict.

**Order study.** All three checkpoints get the exhaustive six-order protocol. Hev is expected to show zero flips and p90 spread at most 1e-4 (a mechanism check, failure is a bug). kev's flip rate and spread are reported as measured; this is the first exhaustive-protocol order measurement of a released kev checkpoint. A nonzero kev flip rate is not evidence about Jev.

**What may be claimed.** Accuracy and calibration differences between a released kev checkpoint and the Hev checkpoints on identical items, with intervals, and the order behaviour of each. What may not: any attribution of a difference to architecture rather than training data, any locked-test statement, or any seed-selection.

### D13 addendum: contamination verified before any M4 result existed  (2026-09-20)

Both models' training rows were checked against every M4 evaluation split, by exact state digest and by exact record digest with `_meta` stripped. Hev's training partition is `evals/decision-v2/train.jsonl` (3,432 records). kev's is decision-v4 `train.jsonl` (10,896 records), fetched from the Hub mirror `jaredpalmer/kev-suites` at pinned revision `a3318ddc1f630c5673232efacd8123a84de3f480` and verified against the decision-v4 manifest SHA-256; it was not copied into `evals/`, so decision-v4 keeps no local train partition.

| Evaluation split | States shared with Hev's training rows | States shared with kev's training rows |
|---|---:|---:|
| decision-v4 development (1,204) | 0 | 0 |
| transfer-v4 development (764) | 0 | 0 |
| decision-v4 calibration (728) | 1 | 0 |

Neither model has seen any headline evaluation row. The single decision-v4 calibration state shared with Hev's training data affects only temperature fitting, a one-parameter fit, and is recorded rather than excluded so the two models keep an identical calibration population.

The two training sets share 1,303 states, as expected from the common public pools. That is a fairness note, not contamination: both models trained on overlapping public material and kev trained on roughly three times as many records. It is one more reason M4 step 1 cannot attribute any difference to architecture.
