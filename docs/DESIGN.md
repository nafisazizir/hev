# Design

## The primitive

Everything is one operation: score K option vectors against one decision vector and softmax. Three API question types map onto it (hev/api.py):

| type | options | answer |
|---|---|---|
| noul | `[no, yes]` | p(yes) |
| choice | one per criteria key, text `key` or `key: description` | argmax, full distribution, confidence |
| score | one per ordered level | expected level, distribution, confidence |

The backbone is a causal LM with its vocabulary head removed. We never generate.

## Packing

One sequence per request:

```
<state> s1 … sS
<q> i1 … iI <decide>                       question 1, instruction branch
<opt> o1 … </opt>                          question 1, option 1
<opt> o1 … </opt>                          question 1, option 2
…
<q> … <decide> <opt> … </opt> …            question 2
```

Delimiters are five existing Qwen special tokens (same five as kev, so packed lengths match kev's and the suites' context limits carry over). User text is rewritten so it cannot contain any `<|…|>` token; boundaries are unforgeable.

Every token carries `(seg, sub)`: `seg` is 0 for the state and k for question k; `sub` is 0 for the instruction branch and j for option j.

## Mask

For query i and key j, in packed order, attention is allowed iff

```
j <= i  and  (
    seg[j] == 0                                       -- anyone reads the state
 or seg[j] == seg[i] and sub[j] == 0                  -- an option reads its own instruction
 or seg[j] == seg[i] and sub[j] == sub[i]             -- a token reads its own branch
)
```

Consequences:

- A question never sees a sibling question (kev's property, kept).
- An option never sees a sibling option, in either direction. Earlier siblings are excluded by the segment rule, later siblings by causality.
- The `<decide>` token sits at the end of the instruction and sees state plus instruction only. It is a learned query, not an aggregator.

Padded query rows keep their diagonal so no softmax row is all `-inf`; the mask uses `finfo.min` rather than `-inf`.

## Positions

Position ids are 0…S-1 for the state. Each question's instruction branch restarts at S. Every option of that question starts at S + I_k, where I_k is that question's instruction length. So option j's tokens see exactly the same (token, position, mask) triple regardless of j.

## Invariance argument

The hidden state at option j's `</opt>` token is a deterministic function of the token ids and positions it can attend to, transitively. By the mask and position rules those are: the state tokens at positions 0…S-1, the instruction tokens at S…S+I_k-1, and option j's own tokens at S+I_k…. None of these depend on which other options exist or where option j sits in the list. Therefore `h(option j)` is invariant to option order and to the presence of siblings.

Tested mechanically in `tests/test_model.py::test_option_hidden_state_invariance` with a random backbone: the vector for "alpha" is identical whether it is first of three or second of two, to 1e-5.

Floating-point note: attention sums over the same key set in a different physical order, so results agree to roughly 1e-6 in fp32 rather than bit-exactly. That is the tolerance the tests use.

## Readout heads

Both take `h_decide` [d] and `h_opts` [K, d] and return logits [K].

**PointerHead** (default). `logit_j = <W_q h_decide, W_k h_opt_j> / sqrt(dp)`, dp = 256. Options are scored independently; exactly permutation-invariant; no option interaction at all. This is the strictest test of H1.

**SetHead.** Project `[h_decide; h_opt_1 … h_opt_K]` to dp, run two pre-norm transformer encoder layers with **no positional encoding**, then pointer-score the refined decide vector against the refined option vectors. Options interact (which archerhume showed Jev's do, via the irrelevant-option log-odds shift), and the whole thing is permutation-equivariant because nothing in it knows an index. This tests H2.

Temperature scaling is a single scalar fit on the calibration split after training, as in kev.

## Score questions and order

Ordinal levels are ordered by meaning, so for `score` questions order is information, not a nuisance. A learned level embedding `E[j]` is added to `h_opt_j` before the readout, for score questions only. It is zero-initialised so a fresh model is invariant everywhere and learns to break invariance only where the labels reward it. `tests/test_model.py::test_score_levels_break_invariance` checks that a nonzero embedding does change the answer under reversal.

Choice and noul get no index signal anywhere. Noul's `[no, yes]` order is fixed by the API, so the readout distinguishes them by text alone.

## What is deliberately the same as kev

- Delimiter tokens, `MAX_STATE`=384, `MAX_BRANCH`=1024, `MAX_PACKED`=2048.
- Request and response JSON, confidence formulas, structured-content rendering.
- LoRA targets and rank, cross-entropy objective, optional ranked-probability term for score.
- Augmentation: option permutation, varied "none of the above", distractors, none minimal pairs. `hev.train --recipe kev-v4` refuses to start unless every shared knob matches kev's published v4 trial, and writes the deviations Hev cannot remove (precision, device, micro-batching, the encoding itself) into `training_config.json`.

Keeping these identical is what makes kev's frozen suites and recorded numbers a valid baseline.

## Cost

Token count per request is identical to kev's. The mask is denser in zeros (each option attends to fewer keys), so attention FLOPs are slightly lower, not higher. Memory is the same L×L mask. No extra forward passes.

## Known risks

- ~~High-K choice (banking77, K=77) may suffer most from independent scoring.~~ **Resolved by M2.** Both heads stayed above D6's banking77 threshold, so the shortlist fallback was not built ([D10](DECISIONS.md), [aggregate](../runs/m2-comparison-s0/result.json)). Independent scoring is not the bottleneck it was expected to be.
- **Out-of-source transfer is where isolation costs something.** With training data, recipe and augmentation matched to kev, the three v4 retrains trail the released kev by 1.46 to 5.42 points on the public transfer sources while staying within 0.67 points on decision ([M4b](RESULTS.md), [aggregate](../runs/m4b-v4-three-seed/result.json)). kev's own `option_isolation` measurement moved the same direction by about the same amount, which makes the encoding the likeliest cause, but precision, hardware and micro-batching still differ, so it is not a controlled attribution.
- The `<decide>` vector cannot see options, so with PointerHead all comparison happens in the dot product. If that is too weak, a cheap upgrade is a bilinear form or an MLP on `[h_decide; h_opt_j; h_decide ⊙ h_opt_j]`, still invariant. Whether SetHead-style interaction recovers the transfer gap is untested: SetHead was never retrained on v4.

## What this design does not claim

Exact order invariance is a property of the mask, independently implementable and independently implemented: kev at HEAD has its own `option_isolation` flag and measures a 0.0 flip rate with it at 0.6B. Nothing here shows invariance improves accuracy. The claim is that the property is available by construction rather than by augmentation, and at no measured decision-accuracy cost once data and recipe are held equal.
