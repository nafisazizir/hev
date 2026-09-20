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

### D13 addendum: the artifact paths that were actually used  (2026-09-20)

D13 predeclared the output directories `runs/m4-kev-0.6b-v4` and `runs/m4-released-v4`. Runs are immutable and failures are kept, so two code faults moved the authoritative artifacts to retry paths. The kev evaluation is `runs/m4-kev-0.6b-v4-r1` and the aggregate is `runs/m4-released-v4-r2`. Cite those.

`runs/m4-released-v4` stopped because the aggregate validated each run's training suite hash rather than the suite actually evaluated, which a cross-suite Hev run correctly reports as decision-v2. `runs/m4-released-v4-r1` stopped because the evaluator did not record the scored kev checkpoint's architecture flags, so the artifact could not demonstrate that kev ran with `option_isolation` false; since the order-sensitivity result is only interpretable against the packing that produced it, the evaluator was fixed and the kev evaluation repeated. The repeat reproduced `runs/m4-kev-0.6b-v4` exactly, including every permutation statistic, and the original is retained. Neither fault touched a model or a datum; no predeclared rule, margin or population was changed after seeing any result.

## D14. M4 outcome: kev leads on the fair population, and exact order invariance is no longer unique  (2026-09-20)

M4 step 1 ran under D13 without amendment. The predictor sanity gate passed at exactly zero difference on both suites, which validates the vendored kev inference path. [Aggregate](../runs/m4-released-v4-r2/result.json)

On the primary public-source population, kev led Hev seed 0 by 1.63 points on decision with a 95% interval of −0.38 to +3.75, and seed 1 by 3.85 points with an interval of +1.73 to +5.96. Under the predeclared rules seed 0 is inconclusive and seed 1 is a kev win. Transfer was inconclusive for both seeds. No result met the equivalence criterion, because no 90% interval fell entirely inside its margin. Record this as kev ahead on decision and undecided on transfer, not as parity.

The comparison is confounded. kev trained on roughly three times as many records with an augmentation Hev does not implement. Do not cite any M4 number as evidence about option isolation.

**Two framings in the repository were wrong and are now corrected.** First, Hev's order-invariance claim implied uniqueness; kev at HEAD has its own `option_isolation` flag and measured it at 0.6B on these suites, so the mechanism is independently implemented and, by kev's controlled measurement, costs about half a point on decision and 1.7 points on transfer. Second, the M3 table compared Hev's exhaustive six-order study against kev's and Jev's single-permutation studies. Measured on one checkpoint, the single-permutation protocol reports 1.67% where six orders report 6.85%. Never place flip rates from different numbers of orders in one table again.

**Implications for the next milestone.** Step 2 should retrain PointerHead on decision-v4 with kev's recipe across three seeds to remove the data confound, and should not retrain SetHead. Expect it to confirm that isolation is roughly accuracy-neutral on decision and slightly costly on transfer rather than to overturn kev's finding. Its value is removing a confound from Hev's own published table and testing a genuinely different isolation design, not a decisive verdict on the mechanism.

Hev seed 1 is materially weaker than seed 0 on the primary decision population, 77.50% against 79.71%. Two seeds cannot separate that from training noise. Step 2's three seeds should be reported as a mean with a range, and no seed may be selected on accuracy.

## D15. M4 step 2 protocol: retrain PointerHead on decision-v4 with kev's recipe, three seeds  (2026-09-20, predeclared before any run)

**Why.** D14's comparison is confounded: kev's published 0.6B trained on 10,896 decision-v4 records with a minimal-pair augmentation Hev did not implement, while both Hev checkpoints trained on 3,432 decision-v2 records. Step 2 removes the data confound by training Hev's PointerHead on kev's exact training partition under kev's published recipe. It does not remove the precision and hardware confound, and it is not a clean architecture isolation; kev's own `option_isolation` trials remain the closer architecture control and must be cited beside any Hev number.

**Four changes made before the first run, all committed before any training started.**

1. `hev.suite.fetch_partition` downloads `decision-v4/train.jsonl` from kev's Hub mirror `jaredpalmer/kev-suites` at the pinned revision `a3318ddc1f630c5673232efacd8123a84de3f480` and verifies it against the local manifest before it is written into place. `load_split(..., fetch=True)` is opt-in, so the offline read path never reaches the network; the partition is git-ignored and never enters this repository. Fetched sha256 `cb55b79e037c9f5a0ef2de4efe79ec6ef5d7d8d21aa64e94eee7731d7eddce74`, 10,896 records, matching `evals/README.md`.
2. Source policy now comes from the suite manifest (`hev.data.source_policy`), not from a module constant. decision-v4 declares `legacy_policy` and `compositional` trainable (448 records each) and `contrastive` eval-only; decision-v2 keeps its own policy unchanged, so M1-M4 runs stay reproducible. The trainable check is on record sources, the eval-only refusal is on exact names at both record and question level, and a suite that declares an empty trainable list may not train at all.
3. `--p-none-pair` wires `hev.data.none_pair` into the training loop exactly as `kev/train.py:170-171` does, including kev's accumulation weighting, so a record that emits pair siblings does not take a larger share of the gradient than one that does not.
4. `--recipe kev-v4` refuses to start unless base, epochs, lr, LoRA rank, effective batch, `ord_w` and all four augmentation probabilities match kev's published trial and the suite hash is decision-v4's, and it records the deviations that remain in `training_config.json`.

**Accepted deviations, recorded in every run.** Precision: Hev fp32, kev bf16 autocast. Hardware: Apple MPS, kev an NVIDIA H100 80GB. Micro-batching: batch 1 with accumulation 8 against kev's batch 8 with accumulation 1, the same effective batch of eight records. Encoding: Hev's option-isolating mask with shared option start positions against kev's plain packing, which is the object of study rather than a deviation to remove. kev's `perm_kl`, `anchor_w`, `head_dim` and `special_embeddings` knobs are not implemented in Hev; the published trial used the values that make them inert.

**Recipe.** `evals/decision-v4` train, 10,896 records, Qwen3-0.6B-Base at the suite-pinned revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, PointerHead only, LoRA r=16, lr 2e-4 OneCycle, AdamW wd 0.01, grad clip 1.0, 2 epochs, effective batch 8, p_none 0.1 / p_none_distract 0.12 / p_distract 0.15 / p_none_pair 0.25, `ord_w` 0. Seeds 0, 1 and 2, predeclared now; no seed may be selected on accuracy and all three are reported. Expected 2,724 optimizer steps per seed, matching kev's trial. The loss-decrease gate is measured on a deterministic 1,024-record probe, chosen by record-id digest so all three seeds share it; `training_metrics.json` records the probe size. SetHead is not retrained: M2 gave it no supported advantage.

**Runs.** `runs/m4b-pointer-v4-s0`, `-s1`, `-s2`, trained in that order; evaluations `runs/m4b-eval-v4-s0`, `-s1`, `-s2`; aggregate `runs/m4b-v4-three-seed`. Immutable; a failed run is kept and never reused.

**Evaluation.** Identical to D13 so the numbers are comparable: `hev.evaluate` on decision-v4 development and transfer-v4 development, device mps, fp32, eager attention, evaluation seed 1, six deterministic orders, 10,000 bootstrap draws, temperature fit once on decision-v4 calibration clean rows and applied unchanged to transfer, no test access. These runs train on decision-v4, so `--allow-cross-suite` is not needed.

**Contamination, verified before any training.** The fetched partition shares zero exact states with decision-v4 development (1,204), transfer-v4 development (764) and decision-v4 calibration (728). It shares 1,303 states with decision-v2 train, as expected from the common public pools. The one calibration state the v2-trained Hev checkpoints had seen is therefore not shared by these runs.

**Statistics and rules.** Three-seed mean with the full range, never a selected seed. Paired, source-stratified, `(source, group_id)`-clustered percentile bootstrap, 10,000 draws, seed 20260920, on identical rows against `runs/m4-kev-0.6b-v4-r1`. Primary population and margins are D13's unchanged: the ten public decision sources and six public transfer sources; 2.0 points on decision accuracy, 3.0 on transfer; "equivalent" needs the 90% interval inside the margin, "better" needs the 95% interval to exclude zero, everything else is inconclusive. The policy arms stay secondary and carry no verdict, but they are now a trained population for Hev as well, which is a change from D13 and must be stated wherever they are shown.

**Mechanism checks.** Every seed: complete coverage, zero Choice argmax flips, p90 correct-probability spread at most 1e-4, packed-versus-separate maximum difference at most 1e-4. Failure is a bug, not a result.

**What may be claimed.** With the training partition, recipe and augmentation held equal, the accuracy, calibration and order behaviour of Hev's isolating encoding against the released kev checkpoint on identical items. **What may not.** That any remaining difference is caused by the encoding alone, since precision and hardware still differ; anything about the locked test; anything about Jev; any seed selection.

### D15 addendum: the artifact paths that were actually used  (2026-09-20)

D15 predeclared `runs/m4b-pointer-v4-s0`. That directory exists and holds only `training_config.json` (status `configured`, written 16:08, the minute the D15 code was committed) with no `training_metrics.json` and no `failure.json`: the first launch was killed from outside before its first optimizer step, so no model and no gate were ever involved. Runs are immutable and failures are kept, so the stub stays and seed 0 was trained as `runs/m4b-pointer-v4-s0-r1` with the identical command. Seeds 1 and 2 kept their predeclared paths. The evaluation and aggregate paths are unchanged: `runs/m4b-eval-v4-s0/-s1/-s2` and `runs/m4b-v4-three-seed`. Cite `-s0-r1` for seed 0.

## D16. M4b outcome: with data held equal, isolation is within a point on decision and behind on transfer  (2026-09-21)

M4b ran under D15 without amendment. All three seeds completed kev's 2,724 steps, passed both training gates and every mechanism check, and the kev predictor sanity gate passed again at zero difference. [Aggregate](../runs/m4b-v4-three-seed/result.json)

On the primary decision population kev leads each seed by 0.67 points, every 95% interval crosses zero, and the 90% upper bounds are 2.21, 2.02 and 2.02 against the 2.0 point margin. Under the predeclared rule all three are inconclusive. Record this as "within a point, equivalence not demonstrated", and do not round it up to parity: the rule was fixed before the runs and the bar was missed, however narrowly. On primary transfer kev leads by 2.71, 5.42 and 1.46 points; seed 1 is a kev win and the others are inconclusive; the three-seed mean is +3.19 with no pooled interval.

**What this changes.** The M4 step 1 decision gap of 1.63 / 3.85 points was mostly training data; D14's expectation held. Hev's order-invariance claim now rests on checkpoints that match kev's data and recipe and stay within a point of it on decision, which is a materially stronger position than M4 step 1's. The policy arms are now trained for both models, so the M4 step 1 caveat about untrained arms no longer applies to the v4 checkpoints.

**What it does not change.** Precision, hardware and micro-batching still differ, so the transfer gap cannot be attributed to the encoding alone. kev's own `option_isolation` measurement showed the same direction and roughly the same size, which makes the encoding the likeliest explanation, but "likeliest" is not a controlled result. Only kev's seed 0 has rows, so every paired interval is one kev checkpoint against three Hev seeds. SetHead was not retrained and nothing here bears on it.

**Publication.** The v4 checkpoints are not published and the served default remains the v2-trained `seed-0` (D11, D12). Whether to replace the Hub revisions with the v4 checkpoints is a separate decision; if taken, all three seeds go up together and none is selected.

**Next.** M5 candidates stand. The one experiment M4b motivates directly is a transfer-focused one: whether the isolation cost on out-of-source questions is a capacity effect (kev saw it grow at 4B), a data effect, or an artifact of the pointer readout that SetHead-style interaction could recover. That is a new predeclared protocol, not an extension of D15.
