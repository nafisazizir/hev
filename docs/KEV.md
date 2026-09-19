# kev: what it is, what we reuse, what we don't

Local clone: `/Users/nafis/Documents/personal/kev` at commit `cc954f2` (2026-09-19). Upstream: github.com/jaredpalmer/kev, Apache-2.0. Read this before porting anything.

## One-paragraph summary

kev is a LoRA adapter (r=16) plus a 256-dim pointer head on Qwen 0.5B/0.6B. A request's state and all its questions are packed into one sequence; a block-causal mask lets each question read the state but not sibling questions; a `<decide>` token at the end of each question is dotted against each option's `</opt>` vector; softmax gives the answer. Trained with cross-entropy on about 13k records from public classification datasets. The model file is 134 lines. The other ~2,800 lines are evaluation discipline.

## File map

| kev file | What | hev status |
|---|---|---|
| `kev/model.py` | encode, block-causal mask, PointerHead, DecisionModel | **Rewritten** in `hev/model.py` with option-level isolation. Same delimiters and limits. |
| `kev/api.py` | TypeSafe schema, render, to_record, to_answers | **Copied** to `hev/api.py`, `qtype` added to records. |
| `kev/data.py` | dataset conversion, augment, none_pair, materialize, source policy | **Partly ported** to `hev/data.py`: materialize, augment, none_pair, policy. Dataset download/build not ported. |
| `kev/suite.py` | freeze suites, load_split with checksums, contrast_cases | **load_split ported** to `hev/suite.py`. freeze and contrast_cases not ported. |
| `kev/train.py` | training loop, perm-KL, ranked probability score, batching | Not ported. M1 writes hev's own; reuse the loss definitions. |
| `kev/evaluate.py` | accuracy/ECE/permutation/IIA/isolation/latency studies | Not ported. M1 ports the metric functions verbatim so numbers are comparable. |
| `kev/benchmark.py`, `kev/compare.py`, `kev/plot.py` | bootstrap CIs, kev-vs-jev comparison, figures | Port at M2 for CIs. |
| `kev/jev.py` | client for TypeSafe's Jev via AI Gateway | Port at M3 if a live Jev comparison is wanted. |
| `kev/serve.py` | FastAPI `/v1/systemone`, `/permute`, `/separate` | M3 writes hev's own with the same routes. |
| `kev/experiment.py`, `modal_app.py` | config-only trial runner, Modal H100 | Port if MPS becomes the bottleneck. |
| `kev/composition.py`, `kev/study_v3.py`, `kev/contrastive.py` | synthetic compositional policy data, v3 study | Not needed until M4. |
| `playground/` | Next.js UI with isolation/forgery probes and chess | Works against hev unchanged once M3 serve exists. |
| `tests/` | unit, api, research, v3 tests | Read `test_unit.py` for what kev asserts; hev's tests are a superset for the mask. |

## Baseline numbers (quote with these paths)

All from the kev clone.

- Released kev-0.5b (Qwen2.5-0.5B), held-out in-distribution: ECE 0.065 raw, 0.031 after temperature; 7% argmax flips and p90 spread 0.25 under option reordering. `MODEL_CARD.md`.
- kev vs Jev, familiar sources: 79.7% vs 81.1% micro accuracy, macro difference -1.8 [-5.5, +1.7]. `runs/kev-vs-jev-v1.json`.
- kev vs Jev, transfer-v1: 63.3% vs 82.3%, -19.1 [-23.1, -15.0]. `runs/kev-vs-jev-transfer-v1.json`.
- Qwen3-0.6B-Base on the v2 recipe, decision-v2 development: 81.6% / 79.3% (two seeds); transfer-v2: 62.0% / 62.1%. `runs/ablation-v2/06-trial-6/result.json`, `07-trial-7/result.json`.
- Qwen2.5-0.5B same recipe: 74.2% / 65.8% dev; 60.5% / 48.2% transfer. `runs/ablation-v2/results.jsonl`.
- v3 capacity study: Qwen3-4B beats 0.6B by +17.5 pp on transfer-v3 [+13.0, +22.1]. `runs/v3-data-capacity-s0/results.jsonl`.
- Calibration warning: one 0.6B trial scored 50% on transfer authorization with 99.6% mean confidence. `runs/ablation-v2/06-trial-6/result.json`.

## Kev's v2 training recipe (what M2 reproduces)

From `PLAN.md` and `train.py` defaults: Qwen3-0.6B-Base pinned to the suite revision, LoRA r=16 alpha=32 dropout 0.05 on all attention and MLP projections, lr 2e-4 OneCycle with 10% warmup, AdamW wd 0.01, grad clip 1.0, effective batch 8 records, 2 epochs, fp32 on MPS, augmentation p_none 0.1 / p_none_distract 0.12 / p_distract 0.15, fresh permutation each epoch. Training on decision-v2 `train.jsonl` (3,432 records). Temperature fit on `calibration.jsonl` (448) only.

## Things kev learned the hard way (do not repeat)

- "None of the above" with a single wording becomes a shortcut. Vary wording and use it as both right and wrong answer. (`data.py` NONE_OPTIONS comment.)
- Minimal pairs must share sentence order and option order; change one fact only. (`PLAN.md` v3 corrections.)
- Compare semantic answer keys, not option positions, when scoring pairs.
- Low aggregate ECE on one transfer set does not mean calibration transfers.
- Zero argmax flips observed in Jev do not prove Jev's architecture is order-invariant; its probabilities do move.
- Eager attention on MPS, SDPA on CUDA; the float 4D mask is only known-good on eager for MPS.
- Evaluate in fp32 with TF32 off on CUDA for exactness.

## What kev deferred that hev picks up

`PLAN.md` "Status and deferred work", last item: "Deferred: option-order architecture experiments. Do not infer Jev's architecture from zero argmax flips." That item is hev.
