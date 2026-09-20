# Evaluation suites

## Format

Each suite is a directory under `evals/` with `manifest.json` and four jsonl splits: `train`, `calibration`, `development`, `test`. A record is a labelled TypeSafe request:

```json
{"state": "...", 
 "questions": {"intent": {"type": "choice", "instructions": "...", "criteria": {"k": "desc or null"}, "label": "k", "src": "banking77"}},
 "_meta": {"id": "banking77/train/123", "source": "banking77", "repo": "...", "revision": "...", "row": 123,
           "text_sha256": "...", "row_sha256": "...", "group_id": "...", "variant": "clean"}}
```

`label` is an option key for choice, a bool for noul, a level index for score. `_meta.group_id` is the bootstrap unit (contrastive siblings share one). `_meta.variant` is `clean`, `none_present`, `none_absent` or `permuted`.

The manifest records sha256 and record count per file, dataset and base-model revisions, the trainable/eval-only policy, and the context-admission rule (every record fits kev's limits under both Qwen tokenizers). `hev.suite.load_split` refuses a file whose checksum or count disagrees.

## Bundled suites

| Suite | Purpose | train / cal / dev / test | Sources |
|---|---|---|---|
| decision-v2 | in-distribution training and development | 3432 / 448 / 1176 / 1176 | banking77, boolq, agnews, mnli, sst5, yelp, trec, dbpedia14, amazon, imdb, contrastive |
| transfer-v2 | out-of-distribution, eval only | 0 / 0 / 668 / 668 | mmlu, emotion, tweet_offensive, qnli, paws, sciq, contrastive |
| decision-v4 | kev's current in-distribution suite (added 2026-09-20; see "v4 suites" below) | (10896, Hub mirror only) / 728 / 1204 / 1176 | ten public sources, legacy_policy, compositional (dev/cal); contrastive (test) |
| transfer-v4 | out-of-distribution, eval only, with held-out rule structures | 0 / 0 / 764 / 764 | mmlu, emotion, tweet_offensive, qnli, paws, sciq, legacy_holdout, composition_holdout (dev); contrastive, composition_holdout (test) |
| smoke-v1 | pipeline check in seconds | 8 / 8 / 30 / 30 | six original sources, mnli and sst5 held out |

Kev also has decision-v1, transfer-v1, public-pool-v4/v5/v6, v3/ and v5/ through v8/ (later training mixes; their development and test partitions are byte-identical to v4). Copy them from the kev clone if a milestone needs them; do not regenerate.

## v4 suites (added 2026-09-20)

`evals/decision-v4/` and `evals/transfer-v4/` are byte-for-byte copies of kev's `evals/v4/` at commit `20fa626` (provenance and digests in `evals/README.md`). They are kev's current selection and transfer suites: every kev number published after its overnight autoresearch run (kev `PLAN.md`, "Overnight autoresearch"), including the `jaredpalmer/kev-0.6b` preview, is on these partitions. Hev's M2/M3 results are on v2 and remain valid; v4 exists so that Hev and kev can be compared on the same rows kev now reports.

**The locked test is untouched.** `evals/decision-v4/test.jsonl` is byte-identical to `evals/decision-v2/test.jsonl` (sha256 `cd7d129a84232e0c7a4e92b4840a6dfe14c8bb93c5d1904b526fa8c085325d2d` in both manifests; `evals/decision-v4/manifest.json` records the v2 files under `parent_files`). The manifest's `protocol.legacy_test` says the inherited bytes were retained without inspecting examples. `evals/transfer-v4/test.jsonl` contains every transfer-v2 test row plus new rows (below); nothing was removed or edited. The `allow_test` rule applies unchanged.

What changed relative to v2, counted from the files (records per `_meta.source`; questions per `src` where noted):

- **decision-v4 development** (`evals/decision-v4/development.jsonl`, 1204 records / 1468 questions vs 1176 / 1440 in `evals/decision-v2/development.jsonl`): the ten public sources are present with identical counts (banking77 116, boolq 80, agnews 116, mnli 116, sst5 80, yelp 80, trec 116, dbpedia14 116, amazon 80, imdb 80, each with the same clean / none_present / none_absent / permuted mix). The v2 `contrastive` arm (196 records) is replaced by `legacy_policy` (96 records, question `src` `contrastive_return_window` / `_spend_threshold` / `_age_eligibility` / `_quantity_limit`, 24 each, all `variant == "clean"`) and `compositional` (128 records, question `src` `composition_atom`, `_negation`, `_conjunction`, `_disjunction`, `_exception`, `_conditional`, `_nested_and`, `_nested_or`, 16 each, all clean). With `_meta` stripped, 920 of the v2 development rows are shared with v4 (1104 distinct v2 payloads, 1112 distinct v4 payloads; 184 v2-only, 192 v4-only).
- **decision-v4 calibration** (`evals/decision-v4/calibration.jsonl`, 728 records / 908 questions vs 448 / 568 in v2): 60 records per public source (v2: 40) plus `legacy_policy` 64 and `compositional` 64 (v2: `contrastive` 48). The manifest's `protocol.calibration` says it is shared across arms and stratified by family and group. Temperature fit on v4 therefore uses different rows than on v2; do not compare a v2-fitted temperature to a v4-fitted one.
- **decision-v4 test** (`evals/decision-v4/test.jsonl`): identical bytes to v2, 1176 records / 1440 questions, including the v2 `contrastive` arm (196 records).
- **transfer-v4 development** (`evals/transfer-v4/development.jsonl`, 764 records vs 668 in `evals/transfer-v2/development.jsonl`): the six public eval-only sources are unchanged (mmlu 116, emotion 116, tweet_offensive 80, qnli 80, paws 80, sciq 116). The v2 `contrastive` rows (80, families authorization and deadline) are replaced by `legacy_holdout` (80 new records, same question `src` names `contrastive_authorization` / `contrastive_deadline`, 40 each) and `composition_holdout` (96 records, question `src` `composition_held_and_or` / `_held_or_not` / `_held_conditional`, 32 each). With `_meta` stripped, 552 development rows are shared with v2 (632 distinct v2 payloads, 704 distinct v4 payloads; 80 v2-only, 152 v4-only).
- **transfer-v4 test** (`evals/transfer-v4/test.jsonl`, 764 records vs 668): every transfer-v2 test row (632 distinct payloads, including `contrastive` 80) plus `composition_holdout` 96 records (question `src` `composition_final_combination` / `_final_negation` / `_final_exception`, 32 each). Those 96 rows are 72 distinct payloads once `_meta` is stripped, because each compositional group holds a relevant pair (a, b) and an irrelevant pair (a, b') and the shared sibling a is emitted twice with different `_meta` (`pair_id`, `pair_kind`). Score complete `pair_id` siblings, as for contrastive pairs.
- **Manifest shape** (`evals/decision-v4/manifest.json`, `evals/transfer-v4/manifest.json`): `"version": 3`; `base_revisions` adds `Qwen/Qwen3-4B-Base`; `parent_files` records the sha256 of every decision-v2, transfer-v2 and public-pool-v4 file; a `protocol` block lists the trained rule shapes (`atom` ... `nested_or`), transfer shapes (`held_and_or`, `held_or_not`, `held_conditional`), locked shapes (`final_combination`, `final_negation`, `final_exception`), train render styles 0 and 1, locked render style 2, 10,000 public training records at 1,000 per source from `evals/public-pool-v4`, and 448 synthetic records per arm. `trainable_sources` / `eval_only_sources` sit at the top level; the v2 `policy`, `admission` and `contrastive` blocks are gone. `files` keeps the same `sha256` / `records` / `questions` schema, so `hev.suite.load_split` works unchanged.
- **New `_meta` keys.** `compositional` and `composition_holdout` records carry `certificate` (the rule tree, atoms, facts and their order, from which the label was computed), `pair_kind` (`relevant` or `irrelevant`), `render_style` (0 or 1 in development, 2 in the locked test), `family` and `family_id` (the rule shape), alongside the usual `id`, `group_id`, `pair_id`, `sibling`, `text_sha256`, `variant`. They have no `repo` / `revision` / `row` / `row_sha256` / `split` because they are generated, not sampled from a dataset. `legacy_policy` and `legacy_holdout` records use the v2 `contrastive` schema (with `repo` and `revision` null, `split: "generated"`). `group_id` remains the bootstrap unit for all of them.
- **Training partition.** `decision-v4/train.jsonl` (10,896 records / 13,896 questions per the manifest) is not in the repo; see `evals/README.md`. Hev has not trained on v4.

Metrics and the bootstrap protocol are unchanged. When reporting v4 numbers, add the new `src` task names to per-task tables and keep `legacy_*` and `composition_*` rows separate from the public sources in macro summaries so v2 and v4 macro-source figures are not silently averaged over different source sets.

## Policy

- Trainable: banking77, boolq, agnews, mnli, sst5, yelp, trec, dbpedia14, amazon, imdb.
- Eval-only, permanently: mmlu (knowledge probe), emotion and tweet_offensive (noisy-label honesty), qnli, paws, sciq (reading transfer).
- Excluded entirely: rotten tomatoes (SST parent), snli (MNLI sibling).
- `test` is locked. Development is for selection. Calibration is for fitting temperature only.
- v4 source names (from `evals/decision-v4/manifest.json` and `evals/transfer-v4/manifest.json`): `legacy_policy` and `compositional` are trainable synthetic policy arms; `contrastive`, `legacy_holdout` and `composition_holdout` are eval-only. The six public eval-only sources are unchanged.

## Metrics to report (same definitions as kev so numbers compare)

Headline metrics are computed on `variant == "clean"` questions only. The `none_present`, `none_absent` and `permuted` variants are diagnostic populations: they are reported separately and never enter headline accuracy/NLL/Brier/ECE. Temperature is fit once on clean calibration rows of the training suite (unweighted macro-task NLL over a deterministic 81-point log grid in [0.25, 4.0]) and the same scalar is applied unchanged to every other split or suite, including transfer and ablation evaluations.

- Accuracy, NLL, Brier per source and per question type; micro on clean rows plus unweighted macro-source and macro-task summaries, each labeled.
- ECE with 10 equal-width bins on the top probability, raw and after temperature fit on calibration.
- Score questions: also MAE of the expected level and ranked probability score.
- Permutation study on choice with K ≥ 3: argmax flip rate and p90 spread of the correct option's probability over six orders, aligned by semantic option key. For Hev this must be 0 and ~1e-6; report it anyway as the check that the implementation matches the design.
- Contrastive pairs: complete `pair_id` siblings only — relevant-pair flip rate and both-correct rate, invariance and both-correct rates for invariant pairs; duplicate or incomplete pairs are rejected. None diagnostics report count, mean none mass, none selection rate, and accuracy separately for none-present and none-absent.
- Bootstrap CIs: deterministic percentile bootstrap, 10,000 draws, fixed seed 20260919. The resampling unit is the `(source, group_id)` cluster so sibling questions and contrastive siblings stay together; draws are stratified by source — each draw resamples that source's clusters with replacement at the original count and includes every row in each sampled cluster. Clean rows only unless the metric explicitly names an ablation subset. Paired model intervals require identical `(id, question)`, group, keys and label populations and reuse the same sampled clusters for both sides.
