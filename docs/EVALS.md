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
| smoke-v1 | pipeline check in seconds | 8 / 8 / 30 / 30 | six original sources, mnli and sst5 held out |

Kev also has decision-v1, transfer-v1, public-pool-v4, v3/ and v4/ (compositional synthetic policies). Copy them from the kev clone if M4 needs them; do not regenerate.

## Policy

- Trainable: banking77, boolq, agnews, mnli, sst5, yelp, trec, dbpedia14, amazon, imdb.
- Eval-only, permanently: mmlu (knowledge probe), emotion and tweet_offensive (noisy-label honesty), qnli, paws, sciq (reading transfer).
- Excluded entirely: rotten tomatoes (SST parent), snli (MNLI sibling).
- `test` is locked. Development is for selection. Calibration is for fitting temperature only.

## Metrics to report (same definitions as kev so numbers compare)

- Accuracy, NLL, Brier per source and per question type; macro over sources.
- ECE with 10 equal-width bins on the top probability, raw and after temperature fit on calibration.
- Score questions: also MAE of the expected level and ranked probability score.
- Permutation study on choice with K ≥ 3: argmax flip rate and p90 spread of the correct option's probability over six orders. For hev this must be 0 and ~1e-6; report it anyway as the check that the implementation matches the design.
- Contrastive pairs: both-correct rate for relevant pairs, invariance rate for irrelevant pairs, none-option mass and selection reported separately.
- Bootstrap CIs clustered by `_meta.group_id`.
