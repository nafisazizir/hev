---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3-0.6B-Base
base_model_relation: adapter
pipeline_tag: text-classification
tags:
  - decision-model
  - calibration
  - lora
  - multiple-choice
  - order-invariance
  - typesafe
  - prefill-only
datasets:
  - legacy-datasets/banking77
  - google/boolq
  - fancyzhx/ag_news
  - nyu-mll/multi_nli
  - SetFit/sst5
  - Yelp/yelp_review_full
  - CogComp/trec
  - fancyzhx/dbpedia_14
  - SetFit/amazon_reviews_multi_en
  - stanfordnlp/imdb
metrics:
  - accuracy
  - expected_calibration_error
  - nll
  - brier_score
model-index:
  - name: hev-0.6b PointerHead seed 0
    results:
      - task: { type: text-classification, name: typed decision }
        dataset: { type: mixed, name: decision-v4 development, public sources }
        metrics:
          - { type: accuracy, value: 0.7971 }
          - { type: expected_calibration_error, value: 0.0221, name: calibrated ECE }
      - task: { type: text-classification, name: out-of-source typed decision }
        dataset: { type: mixed, name: transfer-v4 development, public sources }
        metrics:
          - { type: accuracy, value: 0.6438 }
  - name: hev-0.6b PointerHead seed 1
    results:
      - task: { type: text-classification, name: typed decision }
        dataset: { type: mixed, name: decision-v4 development, public sources }
        metrics:
          - { type: accuracy, value: 0.7750 }
          - { type: expected_calibration_error, value: 0.0492, name: calibrated ECE }
      - task: { type: text-classification, name: out-of-source typed decision }
        dataset: { type: mixed, name: transfer-v4 development, public sources }
        metrics:
          - { type: accuracy, value: 0.6396 }
---

# Model Card: hev-0.6b

`hev-0.6b` is a small, prefill-only decision model. It takes one state and typed questions, then returns a probability distribution for each question in one forward pass. It does not generate text.

The model is a LoRA adapter and PointerHead on `Qwen/Qwen3-0.6B-Base`. Its defining change from the kev revision it forked from is option-level isolation: every option receives the same positions and cannot attend to sibling options. Choice outputs are therefore permutation-equivariant by construction. kev has since added an `option_isolation` flag of its own, so the mechanism is no longer unique to Hev.

This is a **development-only research prototype**. It is not a production decision system, is not Jev and is not affiliated with TypeSafe.

- Code, design, training and evaluation: [github.com/nafisazizir/hev](https://github.com/nafisazizir/hev)
- Planned Hub layout: `OWNER/hev-0.6b`, revisions `seed-0` and `seed-1`
- Current result ledger (M4b, controlled retrain on v4): [`runs/m4b-v4-three-seed/result.json`](https://github.com/nafisazizir/hev/blob/main/runs/m4b-v4-three-seed/result.json)
- Released-checkpoint ledger (M4 step 1, v4 development): [`runs/m4-released-v4-r2/result.json`](https://github.com/nafisazizir/hev/blob/main/runs/m4-released-v4-r2/result.json)
- Earlier three-way ledger (M3, v2 development): [`runs/m3-three-way-v2-r1/result.json`](https://github.com/nafisazizir/hev/blob/main/runs/m3-three-way-v2-r1/result.json)

## Model Details

| | |
|---|---|
| Developed by | Nafis Riza, with Devin (Cognition) |
| Model type | Causal transformer, prefill-only, option-isolating mask, pointer readout |
| Base model | `Qwen/Qwen3-0.6B-Base` |
| Base revision | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| Adapter | LoRA rank 16, alpha 32, dropout 0.05, all attention and MLP projections |
| Head | `1024 → 256` query/key projections, scaled dot product, softmax over options |
| Score signal | Zero-initialized learned level embedding, up to 64 levels |
| Precision | fp32 training and evaluation on Apple MPS |
| Training context | 384 state tokens, 1,024 tokens per question branch, 2,048 packed |
| Serving context | Up to 8,192 state / branch tokens, bounded by the backbone context |
| Question types | `noul`, `choice`, `score` |
| Language | English |
| License | Apache-2.0 for Hev code, adapter and head; dependencies and data retain their terms |
| Version | 0.1.0 research preview |

The head input size is the Qwen3-0.6B hidden size. The model reuses five existing Qwen special tokens as delimiters; it adds no vocabulary rows.

## Checkpoints

The two checkpoints use the same predeclared recipe. They are reported together; neither was selected as best. The figures below are the `decision-v2` / `transfer-v2` development results those checkpoints were evaluated on at training time, with the temperature fitted on `decision-v2` calibration. The current headline evaluation is on the v4 development suites and is in [Evaluation](#evaluation).

| Revision | Training seed | decision-v2 accuracy | transfer-v2 accuracy | decision-v2 calibration temperature |
|---|---:|---:|---:|---:|
| `seed-0` | 0 | 80.67% | 61.07% | 1.932 |
| `seed-1` | 1 | 79.33% | 60.36% | 1.866 |
| **Mean** | — | **80.00%** | **60.71%** | — |

Weights are not stored in git. Each Hub revision contains the PEFT adapter, `readout.pt`, training/evaluation ledgers and a SHA-256 manifest. Load a published revision with:

```bash
uv run --extra serve python -m hev.serve \
  --run hf://OWNER/hev-0.6b@seed-0 \
  --port 8008
```

## Intended Use

**Intended.** Research on direct probability readouts, option-order robustness, isolated attention branches, calibration, compact local models and TypeSafe-compatible APIs. Local demos and teaching.

**Not intended.** Production or consequential decisions in medicine, law, employment, credit, insurance, fraud, moderation, public services or safety-critical control. Do not treat returned confidence as a verified probability on a new workflow.

## Architecture

One request is packed as:

```text
<state> state
<q> instructions <decide>
<opt> option 1 </opt>
<opt> option 2 </opt>
...
```

Each token carries a question segment and option subsegment. An option may attend to:

1. the shared state,
2. its question instructions,
3. its own earlier tokens.

It cannot attend to sibling options. Every option starts at the same position. The selected PointerHead computes:

```text
logit_j = <Wq h_decide, Wk h_option_j> / sqrt(256)
```

Choice and Noul receive no positional option signal. Score receives a learned level embedding because level order is semantic.

The mathematical mechanism is permutation-equivariant; observed differences around `1e-6` are floating-point residuals.

## Training Data

The **published** `seed-0` and `seed-1` checkpoints use the frozen `decision-v2` train partition: 3,432 records and 4,332 questions. The unpublished v4 retrains behind this card's headline evaluation use kev's own `decision-v4` train partition instead; see [The v4 controlled retrain](#the-v4-controlled-retrain) below.

| Source | Records | Conversion |
|---|---:|---|
| Banking77 | 300 | Choice, 77 intents |
| BoolQ | 300 | Noul |
| AG News | 300 | Choice plus two Noul questions |
| MNLI | 300 | Choice, 3 labels |
| SST-5 | 300 | Score, 5 levels |
| Yelp Review Full | 300 | Score plus Noul |
| TREC | 300 | Choice, 6 labels |
| DBpedia14 | 300 | Choice, 14 labels |
| Amazon Reviews | 300 | Score, 5 levels |
| IMDb | 300 | Noul |
| Deterministic contrastive policies | 432 | Choice and Noul minimal pairs |

Question totals are 1,740 Choice, 1,692 Noul and 900 Score.

The contrastive policies are generated from executable rules without an LLM. Choice augmentation always shuffles options and sometimes adds varied none-of-the-above or irrelevant distractor options. Every training request passes through the same rendering path used at serving time.

Eval-only sources are permanently refused during training: MMLU, Emotion, TweetEval offensive, QNLI, PAWS and SciQ. Dataset revisions, provenance and split checksums are recorded in the suite manifests. Datasets retain their own licenses and known biases.

## Training Procedure

| | |
|---|---|
| Objective | Record-mean cross-entropy; ranked-probability weight 0 for these checkpoints |
| Optimizer | AdamW, lr `2e-4`, weight decay 0.01, OneCycle schedule, 10% warm-up |
| Batch | 1 record, gradient accumulation 8, effective batch 8 |
| Epochs | 2 |
| Steps | 858 |
| Exposure | 6,864 records, 8,664 questions |
| Hardware | Apple M4 Max, 36 GB unified memory, PyTorch MPS |
| Wall time | 19.0 minutes seed 0; 22.2 minutes seed 1 |
| Peak MPS allocation | 3.27 GB |
| Seeds | 0 and 1, both reported |

Training data is augmented independently each epoch from deterministic per-record seeds. Runs refuse existing output directories and retain failures.

### The v4 controlled retrain

The three checkpoints behind this card's headline evaluation are **not published**. They exist to remove the training-data confound from the kev comparison, and they are reported together with no seed selected.

Data is kev's `decision-v4` train partition: 10,896 records and 13,896 questions (sha256 `cb55b79e…`), fetched from kev's pinned Hub mirror and verified against the suite manifest. Which arms may train is read from the manifest rather than hard-coded, so the synthetic `legacy_policy` and `compositional` arms train while `contrastive` never does. The same six eval-only sources remain permanently refused.

| | |
|---|---|
| Recipe guard | `--recipe kev-v4`; refuses to start unless every shared knob matches kev's `v4-06b-hardened` trial |
| Objective | Record-mean cross-entropy; ranked-probability weight 0 |
| Optimizer | AdamW, lr `2e-4`, weight decay 0.01, OneCycle schedule, 10% warm-up |
| Batch | 1 record, gradient accumulation 8, effective batch 8 |
| Epochs | 2 |
| Steps | 2,724 per seed |
| Exposure | 21,792 records, ~33,000 questions per seed, including ~5,300 none-pair minimal-pair records |
| Augmentation | kev's v4 settings: `p_none` 0.1, `p_none_distract` 0.12, `p_distract` 0.15, `p_none_pair` 0.25 |
| Hardware | Apple M4 Max, 36 GB unified memory, PyTorch MPS, fp32 |
| Wall time | 105.6 / 88.9 / 81.5 minutes for seeds 0 / 1 / 2 |
| Peak MPS allocation | 3.42 / 3.13 / 3.22 GB |
| Calibration temperature | 1.682 / 1.682 / 1.803, fit on `decision-v4` calibration only |
| Seeds | 0, 1 and 2, all reported |

Four deviations from kev cannot be removed on this machine and are recorded in each run's `training_config.json` rather than glossed: fp32 against kev's bf16 autocast, Apple MPS against an H100, batch 1 with accumulation 8 against batch 8 with no accumulation, and the option-isolating encoding itself, which is the object of study. Four kev knobs Hev does not implement (`perm_kl`, `anchor_w`, `head_dim`, `special_embeddings`) were all at kev's default in the reference trial. [Seed 0](runs/m4b-eval-v4-s0/result.json), [seed 1](runs/m4b-eval-v4-s1/result.json), [seed 2](runs/m4b-eval-v4-s2/result.json)

## Evaluation

All reported results use frozen **development** partitions. The locked test split has not been accessed. No number in this card is a test-set result.

### v4 development, controlled retrain: current headline

The published checkpoints trained on `decision-v2`. To remove that confound, the same PointerHead recipe was retrained on kev's own `decision-v4` partition (10,896 records, none-of-the-above minimal pairs on 25% of Choice records) under kev's published v4 recipe for three predeclared seeds, then scored against the released `jaredpalmer/kev-0.6b` on identical `decision-v4` and `transfer-v4` development rows. Those three checkpoints are **not published**; they are the controlled measurement behind this card's claims. Precision (fp32 against bf16), hardware (Apple MPS against an H100) and micro-batching still differ.

| Model | Decision accuracy, n=1040 | Transfer accuracy, n=480 | Decision calibrated ECE | Decision calibrated Brier |
|---|---:|---:|---:|---:|
| kev-0.6b released, seed 0 | 81.35% (78.85–83.75) | 65.42% (61.46–69.38) | 0.0391 | 0.2702 |
| Hev PointerHead v4, seed 0 | 80.67% (78.17–83.17) | 62.71% (58.75–66.67) | 0.0400 | 0.2645 |
| Hev PointerHead v4, seed 1 | 80.67% (78.17–83.17) | 60.00% (55.83–63.96) | 0.0394 | 0.2699 |
| Hev PointerHead v4, seed 2 | 80.67% (78.27–83.08) | 63.96% (60.00–67.92) | 0.0280 | 0.2631 |

Paired kev minus Hev on identical rows: decision +0.67 points at every seed, 95% intervals (−1.15, +2.50), (−0.96, +2.31) and (−0.87, +2.31), all inconclusive under the predeclared rule and each missing the 2.0 point equivalence bar by at most 0.21 points of the 90% bound; transfer +2.71 (−1.88, +7.29), +5.42 (+1.04, +9.79) and +1.46 (−3.12, +6.04), a kev win at seed 1 and inconclusive otherwise. The three identical decision figures are distinct models with equal correct counts; they disagree on 82 to 97 rows pairwise. All three seeds flip zero Choice questions under the exhaustive six-order protocol. Full tables: [RESULTS](https://github.com/nafisazizir/hev/blob/main/docs/RESULTS.md) and [`runs/m4b-v4-three-seed/result.json`](https://github.com/nafisazizir/hev/blob/main/runs/m4b-v4-three-seed/result.json).

### v4 development: the published checkpoints against the released kev

Both published checkpoints were re-scored on kev's frozen `decision-v4` and `transfer-v4` development splits alongside the released `jaredpalmer/kev-0.6b` checkpoint (Hub snapshot `83e05fabf7ef08e343bb4daf144e08777f99713d`, seed 0 of 3), using one evaluator and identical rows. The primary population is the ten public decision sources and the six public transfer sources: both models trained on the decision ten and neither trained on the transfer six. Intervals are 95% source-stratified, clustered bootstrap intervals with 10,000 draws. Temperature was fitted per model on `decision-v4` calibration only and applied unchanged to transfer.

| Model | Decision accuracy, n=1040 | Transfer accuracy, n=480 | Decision calibrated ECE | Decision calibrated Brier |
|---|---:|---:|---:|---:|
| kev-0.6b released, seed 0 | 81.35% (78.85–83.75) | 65.42% (61.46–69.38) | 0.0391 | 0.2702 |
| **Hev PointerHead, seed 0** | 79.71% (77.12–82.12) | 64.38% (60.42–68.33) | 0.0221 | 0.2798 |
| **Hev PointerHead, seed 1** | 77.50% (74.90–80.00) | 63.96% (60.00–67.92) | 0.0492 | 0.3036 |

Paired kev minus Hev differences on identical rows: on decision, +1.63 points at Hev seed 0 with a 95% interval of (−0.38, +3.75), and +3.85 points at seed 1 with a 95% interval of (+1.73, +5.96); on transfer, +1.04 points at seed 0 with (−3.75, +5.63) and +1.46 points at seed 1 with (−3.33, +6.04). Under the protocol fixed before the runs, seed 0 on decision is inconclusive, seed 1 on decision is a kev win, and both transfer comparisons are inconclusive. Neither seed meets the equivalence bar.

**Hev is behind the released kev on this population.** The comparison is confounded and cannot be read as an architecture result: kev trained on 10,896 `decision-v4` records with none-of-the-above minimal pairs on 25% of Choice records, in bf16 on one H100, while Hev trained on 3,432 `decision-v2` records with no such pairs, in fp32 on Apple MPS. The base model, its pinned revision, LoRA rank, learning rate, epoch count and effective batch size are the same. The two training sets share 1,303 states.

The v4 suites also contain policy arms (`compositional`, `legacy_policy`, `composition_holdout`, `legacy_holdout`) that kev's v4 data was built to teach and Hev has never seen. They are reported separately, carry no verdict and support no comparison: on decision policy arms (n=224) kev scores 76.34% against Hev's 68.75% and 66.96%; on transfer policy arms (n=176) kev scores 44.32% against Hev's 47.16% and 42.61%. Mixing them into an all-rows figure gives 80.46% / 77.77% / 75.63% on decision over 1,264 rows and 59.76% / 59.76% / 58.23% on transfer over 656 rows.

Contamination was checked before any result existed: zero states are shared between either model's training rows and either graded split. One state is shared between Hev's training rows and `decision-v4` calibration, affecting only the one-parameter temperature fit.

The evaluator was validated first. Re-scoring the released kev checkpoint reproduced kev's own published development accuracy exactly, 0.8045886075949367 on decision and 0.5975609756097561 on transfer, an absolute difference of 0.0 on both.

Full tables and the artifact are in [RESULTS](https://github.com/nafisazizir/hev/blob/main/docs/RESULTS.md) and [`runs/m4-released-v4-r2/result.json`](https://github.com/nafisazizir/hev/blob/main/runs/m4-released-v4-r2/result.json).

### v2 development: earlier comparison

The earlier M3 comparison ran on the different `decision-v2` and `transfer-v2` suites and included a hosted Jev snapshot. It is a separate result on separate items and is not comparable to the v4 table above.

| Model | decision-v2 accuracy | transfer-v2 accuracy | decision-v2 calibrated ECE |
|---|---:|---:|---:|
| Hev PointerHead, two-seed mean | 80.00% | 60.71% | 0.020 |
| kev Qwen3-0.6B, two-seed mean | 80.46% | 62.05% | 0.030 |
| Jev `1.13.0` | 83.50% | 85.36% | 0.103 |

Jev returns rounded probabilities, so its NLL is floor-sensitive. Details are in [RESULTS](https://github.com/nafisazizir/hev/blob/main/docs/RESULTS.md) and [`runs/m3-three-way-v2-r1/result.json`](https://github.com/nafisazizir/hev/blob/main/runs/m3-three-way-v2-r1/result.json).

### Option Order

All three v4 checkpoints received the same exhaustive protocol: every distinct order of six, on every eligible Choice question.

| Model | Decision flips of 628 | Transfer flips of 348 | Decision p90 correct-probability spread |
|---|---:|---:|---:|
| **Hev Pointer seed 0** | 0 (0%) | 0 (0%) | `1.33e-6` |
| **Hev Pointer seed 1** | 0 (0%) | 0 (0%) | below the `1e-4` mechanism tolerance |
| kev-0.6b released, seed 0 | 43 (6.85%) | 87 (25.00%) | 0.1334 |

kev's largest single-question correct-probability spread across the six orders was 0.8988 on decision, and its transfer p90 spread was 0.3139.

Two qualifications matter. First, protocol choice changes the measurement: kev's own clean-versus-one-permutation protocol on this same checkpoint reported a 1.67% decision flip rate over 60 cases, roughly a quarter of the exhaustive figure, so any table that places six-order results beside one-permutation results is not like-for-like. Earlier versions of this card did exactly that and the figures it quoted for kev and Jev understate flips. Second, exact order invariance is **not unique to Hev**: kev now has its own `option_isolation` flag and measured a 0.0 flip rate with it at 0.6B, at a cost by its own controlled measurement of roughly half a point of decision accuracy and 1.7 points of transfer accuracy. The kev checkpoint evaluated here ran with `option_isolation` false.

Exact order invariance is not shown to improve accuracy.

## Limitations

- **No locked-test result.** Every headline number is development-only.
- **The published checkpoints are behind the released kev.** Trained on `decision-v2`, they are 1.63 and 3.85 points behind `kev-0.6b` on the public v4 decision sources. The controlled v4 retrains close that to 0.67 points at every seed, but those retrains are not the published weights.
- **Transfer cost of isolation.** With data and recipe matched, kev leads the v4 retrains by 1.46 to 5.42 points on the public transfer sources. Precision, hardware and micro-batching still differ, so this cannot be attributed to the encoding alone, though it matches kev's own isolation measurement.
- **Exact order invariance is not unique to this model.** kev implements its own `option_isolation` and has measured a 0.0 flip rate with it at 0.6B. Invariance is a property of the mask, not evidence of better accuracy.
- **One kev checkpoint only.** Only kev's published seed-0 checkpoint can be re-scored. Its other two seeds are point-only figures from kev's own evaluator with no per-example rows, so they can never enter a paired interval.
- **Policy and compositional structures are untrained in the published checkpoints.** They never saw the v4 policy arms; results there are descriptive. The v4 retrains did train on them.
- **Seed variance is material.** The published pair spans 79.71% and 77.50% on primary v4 decision accuracy; the three v4 retrains span 60.00% to 63.96% on primary transfer. Differences smaller than those spans are not meaningful.
- **Transfer gap.** Out-of-source accuracy is 64.38% and 63.96% for the published checkpoints and 62.71% / 60.00% / 63.96% for the v4 retrains on the public v4 transfer sources, against kev's 65.42%.
- **Small backbone.** Factual knowledge, arithmetic and multi-step reasoning are limited.
- **Narrow supervision.** English classification datasets and deterministic policies do not cover arbitrary business workflows.
- **Pointer bottleneck.** Options are scored independently. Comparative reasoning can be difficult, especially for close or high-cardinality choices.
- **Ordinal learning is weak.** The Score level embedding changes probabilities but has no supported NLL improvement; the ranked-probability loss was disabled.
- **Calibration is local.** A temperature fitted on one calibration split does not guarantee calibrated transfer or deployment probabilities.
- **Score confidence is a stand-in.** TypeSafe's exact formula is unpublished.
- **No mobile claim.** PyTorch/Transformers inference is tested on Apple Silicon, not iPhone or Core ML.
- **No production hardening.** The server has no authentication, permissive CORS and single-process local serving semantics.

## Bias, Risks And Recommendations

The model inherits biases and label noise from Qwen3 and every training dataset, including English banking terminology, product reviews, news taxonomies and crowd-sourced language-inference labels.

Direct probabilities can appear more authoritative than generated prose. A value such as `0.92` is the model's mass on one supplied option after temperature scaling, not proof that the answer is correct. Measure calibration on representative labelled outcomes before applying thresholds.

Option isolation prevents one option from changing a sibling option's backbone state. It does not prevent misleading state text, ambiguous criteria, data leakage, distribution shift or prompt injection through allowed content, and it has not been shown to improve accuracy.

Use human review for consequential decisions. Log model version, full criteria and distributions. Provide abstention and rollback paths.

## Environmental Impact

Each published checkpoint's training run took 19–22 minutes on one Apple M4 Max; the three v4 retrains took 82–106 minutes each. Energy consumption was not measured. Evaluation runs, the hosted Jev comparison and the M4 re-scoring of the released kev checkpoint are documented in the immutable ledgers.

## Citation

```bibtex
@software{hev2026,
  title  = {Hev: an option-order-invariant decision model},
  author = {Riza, Nafis},
  year   = {2026},
  url    = {https://github.com/nafisazizir/hev}
}

@software{kev2026,
  title  = {kev: a laptop-scale reconstruction of a Jev-style decision model},
  author = {Palmer, Jared},
  year   = {2026},
  url    = {https://github.com/jaredpalmer/kev}
}

@misc{hume2026jev,
  title  = {Jev's Architecture Unmasked},
  author = {Hume, Archer},
  year   = {2026},
  url    = {https://archerhume.com/posts/jevs-architecture-unmasked}
}
```

## Acknowledgements

Created by Nafis Riza with engineering and research assistance from Devin by Cognition.

The work builds on TypeSafe's public description of Jev and System One, Archer Hume's architectural reconstruction, Jared Palmer's kev implementation and evaluation suites, the Qwen3 base model, and the public datasets listed above. Their inclusion does not imply endorsement or affiliation.

See [THIRD_PARTY_NOTICES](https://github.com/nafisazizir/hev/blob/main/THIRD_PARTY_NOTICES.md) for provenance and licensing notes.

## Contact

Open an issue at [github.com/nafisazizir/hev](https://github.com/nafisazizir/hev/issues).
