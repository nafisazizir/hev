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
  - name: hev-0.6b PointerHead development mean
    results:
      - task: { type: text-classification, name: typed decision }
        dataset: { type: mixed, name: decision-v2 development }
        metrics:
          - { type: accuracy, value: 0.8000 }
          - { type: expected_calibration_error, value: 0.0202, name: calibrated ECE }
      - task: { type: text-classification, name: out-of-source typed decision }
        dataset: { type: mixed, name: transfer-v2 development }
        metrics:
          - { type: accuracy, value: 0.6071 }
---

# Model Card: hev-0.6b

`hev-0.6b` is a small, prefill-only decision model. It takes one state and typed questions, then returns a probability distribution for each question in one forward pass. It does not generate text.

The model is a LoRA adapter and PointerHead on `Qwen/Qwen3-0.6B-Base`. Its defining change from kev is option-level isolation: every option receives the same positions and cannot attend to sibling options. Choice outputs are therefore permutation-equivariant by construction.

This is a **development-only research prototype**. It is not a production decision system, is not Jev and is not affiliated with TypeSafe.

- Code, design, training and evaluation: [github.com/nafisazizir/hev](https://github.com/nafisazizir/hev)
- Planned Hub layout: `OWNER/hev-0.6b`, revisions `seed-0` and `seed-1`
- Full result ledger: [`runs/m3-three-way-v2-r1/result.json`](https://github.com/nafisazizir/hev/blob/main/runs/m3-three-way-v2-r1/result.json)

## Model Details

| | |
|---|---|
| Developed by | Nafis Azizi Riza, with Devin (Cognition) |
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

The two checkpoints use the same predeclared recipe. They are reported together; neither was selected as best.

| Revision | Training seed | Decision accuracy | Transfer accuracy | Calibration temperature |
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

Training uses the frozen `decision-v2` train partition: 3,432 records and 4,332 questions.

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

## Evaluation

All reported results use frozen **development** partitions. The locked test split has not been accessed.

### Accuracy And Calibration

| Model | Decision accuracy | Transfer accuracy | Decision raw NLL | Decision calibrated NLL / ECE / Brier |
|---|---:|---:|---:|---:|
| **Hev PointerHead** | **80.00%** | **60.71%** | 0.606 | 0.502 / 0.020 / 0.269 |
| kev Qwen3-0.6B | 80.46% | 62.05% | 0.604 | 0.511 / 0.030 / 0.272 |
| Jev `1.13.0` | 83.50% | 85.36% | 0.813* | 0.544 / 0.103 / 0.274 |

Hev and kev are two-seed means. Jev is one hosted snapshot. `*` Jev returns rounded probabilities, including zeros, so NLL depends on the `1e-9` floor.

Hev versus Jev is not a controlled training comparison. Training data, architecture, compute and availability differ. The Hev and kev recipes share data and model size, but the recorded Kev baselines trained on H100/bf16/batch 8 while Hev trained on MPS/fp32 with gradient accumulation.

### Choice Accuracy

| Model | Decision Choice | Transfer Choice |
|---|---:|---:|
| Hev PointerHead, two-seed mean | 85.52% | 61.67% |
| Jev `1.13.0` | 89.79% | 82.50% |
| kev | unavailable in its checked-in v2 artifacts |

Exact order invariance does not imply higher Choice accuracy.

### Option Order

| Model | Protocol | Decision flips | Transfer flips |
|---|---|---:|---:|
| **Hev Pointer seed 0** | six orders, all eligible Choice | **0 / 696** | **0 / 348** |
| **Hev Pointer seed 1** | six orders, all eligible Choice | **0 / 696** | **0 / 348** |
| kev seed 0 / 1 | clean plus one fixed permutation | 5.56% / 4.17% of 72 | 8.33% / 8.33% of 36 |
| Jev `1.13.0` | clean plus one fixed permutation | 1.39% of 72 | 0% of 36 |

Hev's decision p90 correct-probability spread is at most `1.45e-6`; transfer is at most `2.86e-6`. Jev's maximum aligned probability movement is 0.456 on decision and 0.200 on transfer despite few observed argmax flips.

The protocols differ and must not be ranked as identical experiments. Hev's exhaustive result verifies its implementation and architectural claim. Zero observed Jev transfer flips do not establish architectural invariance.

## Limitations

- **No locked-test result.** Every headline number is development-only.
- **Transfer gap.** Hev trails Jev substantially out of source. MMLU and held-out deadline are major weaknesses.
- **Small backbone.** Factual knowledge, arithmetic and multi-step reasoning are limited.
- **Narrow supervision.** English classification datasets and deterministic policies do not cover arbitrary business workflows.
- **Pointer bottleneck.** Options are scored independently. Comparative reasoning can be difficult, especially for close or high-cardinality choices.
- **Ordinal learning is weak.** The Score level embedding changes probabilities but has no supported NLL improvement; the ranked-probability loss was disabled.
- **Calibration is local.** A fitted temperature on decision-v2 does not guarantee calibrated transfer or deployment probabilities.
- **Score confidence is a stand-in.** TypeSafe's exact formula is unpublished.
- **No mobile claim.** PyTorch/Transformers inference is tested on Apple Silicon, not iPhone or Core ML.
- **No production hardening.** The server has no authentication, permissive CORS and single-process local serving semantics.

## Bias, Risks And Recommendations

The model inherits biases and label noise from Qwen3 and every training dataset, including English banking terminology, product reviews, news taxonomies and crowd-sourced language-inference labels.

Direct probabilities can appear more authoritative than generated prose. A value such as `0.92` is the model's mass on one supplied option after temperature scaling, not proof that the answer is correct. Measure calibration on representative labelled outcomes before applying thresholds.

Option isolation prevents one option from changing a sibling option's backbone state. It does not prevent misleading state text, ambiguous criteria, data leakage, distribution shift or prompt injection through allowed content.

Use human review for consequential decisions. Log model version, full criteria and distributions. Provide abstention and rollback paths.

## Environmental Impact

Each local training run took 19–22 minutes on one Apple M4 Max. Energy consumption was not measured. Evaluation and the hosted Jev comparison are documented in the immutable ledgers.

## Citation

```bibtex
@software{hev2026,
  title  = {hev: an option-order-invariant decision model},
  author = {Azizi Riza, Nafis},
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

Created by Nafis Azizi Riza with engineering and research assistance from Devin by Cognition.

The work builds on TypeSafe's public description of Jev and System One, Archer Hume's architectural reconstruction, Jared Palmer's kev implementation and evaluation suites, the Qwen3 base model, and the public datasets listed above. Their inclusion does not imply endorsement or affiliation.

See [THIRD_PARTY_NOTICES](https://github.com/nafisazizir/hev/blob/main/THIRD_PARTY_NOTICES.md) for provenance and licensing notes.

## Contact

Open an issue at [github.com/nafisazizir/hev](https://github.com/nafisazizir/hev/issues).
