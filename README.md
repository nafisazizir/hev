# Hev

Small, option-order-invariant decision model. Typed questions in, calibrated probabilities out, one prefill, no decoding.

<p>
  <a href="https://github.com/nafisazizir/hev/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/nafisazizir/hev/ci.yml?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="MODEL_CARD.md"><img alt="Model card" src="https://img.shields.io/badge/MODEL%20CARD-read-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
</p>

Hev is a LoRA adapter and PointerHead on `Qwen/Qwen3-0.6B-Base`. It reads one state, answers many typed questions in parallel and returns probability distributions without generating text.

Its experiment is narrow: the kev version Hev forked from isolates questions; Hev also isolates every option from its sibling options. Every option receives the same positions and can attend only to the state, its question and itself. Reordering Choice options therefore cannot change their backbone representations. The readout is permutation-equivariant, so semantic answers are invariant by construction. kev has since added its own `option_isolation` flag, so the mechanism is no longer unique to Hev.

Hev is an independent research project. It is not Jev, is not affiliated with TypeSafe and does not claim to reproduce Jev's private implementation.

## Highlights

- **Small on purpose.** A 0.6B backbone, LoRA rank 16 and a 256-dimensional pointer readout.
- **Exact option isolation.** An option never reads a sibling option. Choice order is nuisance, not signal.
- **One pass, many answers.** State and questions share one prefill; there is no decoding loop.
- **Probabilities, not prose.** Cross-entropy training plus calibration-only temperature scaling.
- **Three question types.** `noul`, `choice` and ordinal `score` use one scoring primitive.
- **Auditable research.** Frozen suites, locked tests, immutable runs and claim-linked result ledgers.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). The research runs used Apple Silicon MPS. CUDA is implemented but not validated here.

```bash
git clone https://github.com/nafisazizir/hev.git
cd hev
uv sync --group dev --extra serve
```

Model weights are not committed to git. The release layout uses one Hugging Face repository with `seed-0` and `seed-1` revisions; neither is presented as the better seed. Until a Hub namespace is chosen, replace `OWNER` below.

```bash
uv run --extra serve python -m hev.serve --run hf://OWNER/hev-0.6b@seed-0 --port 8008
```

A local evaluated run works too:

```bash
uv run --extra serve python -m hev.serve --run runs/m3-pointer-s1 --port 8008
```

## Quick Start

```bash
curl -s localhost:8008/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "The parcel arrived late and the card was charged twice.",
    "model": "hev-latest",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "shipping": "Delivery and tracking",
          "billing": "Charges and payments",
          "returns": "Refunds and exchanges"
        }
      },
      "urgent": {
        "type": "noul",
        "instructions": "Does this need urgent attention?"
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"]
      }
    }
  }'
```

The server also works with the official TypeSafe SDK by changing `base_url` to `http://127.0.0.1:8008` and using model `hev-latest`.

## API

### `POST /v1/systemone`

| Type | Input | Output |
|---|---|---|
| `noul` | instructions, optional true/false criteria | `p(yes)` |
| `choice` | instructions, keyed criteria | semantic key, full distribution, confidence |
| `score` | instructions, ordered levels | expected level, full distribution, confidence |

Structured state, instructions and criteria are rendered to labelled text. Reserved delimiters are rewritten out of user content. Validation errors return `422`.

### `GET /v1/models`

Returns the loaded run, base model, head and calibration temperature.

The server has no authentication and permissive CORS. It binds to `127.0.0.1` and is intended for local research only.

## How It Works

```mermaid
flowchart LR
    A[state + typed questions] --> B[render]
    B --> C[pack one sequence]
    C --> D[option-isolated causal mask]
    D --> E[Qwen3-0.6B + LoRA]
    E --> F[PointerHead]
    F --> G[softmax + temperature]
    G --> H[typed probabilities]
```

For query token `i` and key token `j`, attention is allowed only when `j <= i` and the key is in the shared state, the current question instruction, or the same option branch. Every option starts at the same position. `h(option)` is therefore a function of `(state, instruction, option text)`, not list position or siblings.

The selected PointerHead scores each option independently:

```text
logit_j = <Wq h_decide, Wk h_option_j> / sqrt(256)
```

See [DESIGN](docs/DESIGN.md) for the mask, positions and invariance argument. Open [the illustrated explainer](docs/explainer.html) in a browser for a worked request.

## Results

All numbers are on frozen **development** splits. The locked test split has not been accessed.

The current headline is M4b: a controlled retrain. Hev's PointerHead was trained on kev's own `decision-v4` partition (10,896 records) under kev's published v4 recipe, minimal-pair augmentation included, for three predeclared seeds, and scored by one evaluator against the released `jaredpalmer/kev-0.6b` checkpoint on identical `decision-v4` and `transfer-v4` development rows. Data, recipe and augmentation match. Precision (fp32 against bf16), hardware (Apple MPS against an H100) and micro-batching still differ, so a remaining difference is not attributable to the encoding alone.

Primary population: the ten public decision sources and six public transfer sources. Intervals are 95% source-stratified, clustered bootstrap intervals with 10,000 draws. All three seeds are reported; none was selected.

| Model | Decision accuracy, n=1040 | Transfer accuracy, n=480 | Decision calibrated ECE |
|---|---:|---:|---:|
| kev-0.6b released, seed 0 | **81.35%** (78.85–83.75) | **65.42%** (61.46–69.38) | 0.0391 |
| Hev PointerHead v4, seed 0 | 80.67% (78.17–83.17) | 62.71% (58.75–66.67) | 0.0400 |
| Hev PointerHead v4, seed 1 | 80.67% (78.17–83.17) | 60.00% (55.83–63.96) | 0.0394 |
| Hev PointerHead v4, seed 2 | 80.67% (78.27–83.08) | 63.96% (60.00–67.92) | **0.0280** |

Paired on identical rows, kev leads by 0.67 points on decision at every seed, with every 95% interval crossing zero and the 90% upper bounds at 2.21 / 2.02 / 2.02 against the predeclared 2.0 point equivalence margin: inconclusive three times, a hair short of equivalence. On transfer kev leads by 2.71 / 5.42 / 1.46 points, a kev win at seed 1 and inconclusive at the others. That is the same shape kev measured for its own `option_isolation` flag: near-neutral in-distribution, a few points out-of-source. The identical decision figures are three different models that happen to get 839 of 1,040 right; they disagree on 82 to 97 rows pairwise.

The earlier evaluation-only M4 step 1, which scored the v2-trained checkpoints on the same rows, had kev ahead by 1.63 and 3.85 points on decision. Most of that gap was training data.

### Option order

Exhaustive six-order protocol, every distinct order on every eligible Choice question, the same protocol for all four checkpoints.

| Model | Decision flips of 628 | Transfer flips of 348 | Decision p90 correct-probability spread |
|---|---:|---:|---:|
| **Hev PointerHead v4, seeds 0 / 1 / 2** | **0 / 0 / 0** | **0 / 0 / 0** | `8.9e-7` / `1.1e-6` / `7.3e-7` |
| kev-0.6b released, seed 0 | 43 (6.85%) | 87 (25.00%) | 0.1334 |

Two qualifications. Protocol matters: kev's own clean-versus-one-permutation protocol on this same checkpoint reported 1.67% over 60 cases, so one-permutation figures understate flips by roughly a factor of four. And exact order invariance is **not unique to Hev**: kev at HEAD ships its own `option_isolation` flag and measured a 0.0 flip rate with it at 0.6B.

The M3 three-way comparison with Jev is a separate, earlier result on the `decision-v2` and `transfer-v2` suites with mismatched order protocols; it is kept in [RESULTS](docs/RESULTS.md).

Full tables, calibration, the training ledgers and the remaining confounds are in [RESULTS](docs/RESULTS.md). The authoritative M4b aggregate is [`runs/m4b-v4-three-seed/result.json`](runs/m4b-v4-three-seed/result.json); the M4 step 1 aggregate is [`runs/m4-released-v4-r2/result.json`](runs/m4-released-v4-r2/result.json).

## Reproduce

Offline tests:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest -p no:cacheprovider
```

Train the selected seed-0 recipe:

```bash
uv run python -m hev.train \
  --suite evals/decision-v2 \
  --out runs/pointer-s0 \
  --head pointer \
  --device mps \
  --seed 0 \
  --epochs 2 \
  --lr 2e-4 \
  --lora 16 \
  --batch 1 \
  --accum 8 \
  --ord-w 0 \
  --p-none 0.1 \
  --p-none-distract 0.12 \
  --p-distract 0.15 \
  --require-loss-decrease
```

Evaluate development only:

```bash
uv run python -m hev.evaluate \
  --run runs/pointer-s0 \
  --suite evals/decision-v2 \
  --transfer evals/transfer-v2 \
  --device mps \
  --seed 1 \
  --permutations 6 \
  --bootstrap-samples 10000
```

Run directories and evaluation outputs are immutable. Use new paths. Never use the locked test split for selection.

### Publish checkpoints

Validate a bundle without network access:

```bash
uv run python -m hev.publish \
  --run runs/m2-pointer-s0 \
  --repo OWNER/hev-0.6b \
  --revision seed-0 \
  --dry-run
```

After `hf auth login`, omit `--dry-run`. Repeat with `runs/m3-pointer-s1` and `--revision seed-1`. Publication refuses failed runs, SetHead runs, test-evaluated runs and mismatched seed/provenance metadata.

## Limitations

- **Development-only.** No locked-test claim has been made.
- **Within a point of the released kev on decision, behind on transfer.** With data and recipe matched, kev leads by 0.67 points on the public decision sources at every seed with intervals crossing zero, and by 1.46 to 5.42 points on the public transfer sources. Equivalence was not demonstrated under the predeclared rule. [Evidence](runs/m4b-v4-three-seed/result.json)
- **Precision and hardware still differ from kev.** Hev trains in fp32 on Apple MPS with accumulation; kev trained in bf16 on an H100. The transfer gap is consistent with kev's own isolation measurement but cannot be attributed to the encoding alone. [Evidence](runs/m4b-v4-three-seed/result.json)
- **Exact order invariance is not unique to Hev.** kev implements its own `option_isolation` and measured a 0.0 flip rate with it at 0.6B. Invariance is also not shown to improve accuracy. [Evidence](runs/m4b-v4-three-seed/result.json)
- **The published checkpoints are the v2-trained ones.** The `seed-0` and `seed-1` Hub revisions trained on 3,432 decision-v2 records and are 1.63 and 3.85 points behind kev on the same rows. The v4 retrains above are not published. [Evidence](runs/m4-released-v4-r2/result.json)
- **Transfer remains weak.** Primary transfer accuracy is 62.71% / 60.00% / 63.96% across the v4 seeds against kev's 65.42%. [Evidence](runs/m4b-v4-three-seed/result.json)
- **Seed variance is real.** Three v4 seeds of one recipe span 60.00% to 63.96% on primary transfer accuracy and 55.34% to 59.30% on all transfer rows. No seed is selected as best. [Evidence](runs/m4b-v4-three-seed/result.json)
- **Small backbone.** 0.6B parameters limit knowledge and multi-step reasoning.
- **English and classification shaped.** Code, long workflows, multilingual use and open-ended generation are outside training.
- **Calibration does not transfer automatically.** Measure it on labelled outcomes from the target workflow.
- **Score ordering is not proven helpful.** The level embedding changes predictions but has no supported NLL benefit.
- **No mobile runtime.** The current implementation is PyTorch/Transformers on MPS, CUDA or CPU; iPhone deployment is untested.
- **No production safety claim.** Do not use it for consequential decisions without independent validation and human oversight.

## Development

```bash
uv sync --group dev --extra serve
PYTHONDONTWRITEBYTECODE=1 uv run pytest -p no:cacheprovider
uv build
```

See [CONTRIBUTING](CONTRIBUTING.md), [SECURITY](SECURITY.md) and the append-only [decision log](docs/DECISIONS.md).

## Authors And Acknowledgements

Created by [Nafis Riza](https://github.com/nafisazizir). Built with [Devin](https://devin.ai) by Cognition.

Hev exists because of work by:

- [TypeSafe](https://typesafe.ai) for Jev and the public System One API contract.
- [Archer Hume](https://archerhume.com/posts/jevs-architecture-unmasked) for the public architectural reconstruction.
- [Jared Palmer](https://github.com/jaredpalmer) for [kev](https://github.com/jaredpalmer/kev), its evaluation discipline and the frozen suites adapted here.
- The [Qwen team](https://huggingface.co/Qwen/Qwen3-0.6B-Base) for the base model.
- The dataset creators, maintainers and annotators listed in the [model card](MODEL_CARD.md#training-data) and [third-party notices](THIRD_PARTY_NOTICES.md).

Names and trademarks belong to their owners. Acknowledgement does not imply endorsement or affiliation.

## License

[Apache-2.0](LICENSE) for Hev code, adapter and readout. The base model, datasets and adapted materials retain their own licenses and terms; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md).
