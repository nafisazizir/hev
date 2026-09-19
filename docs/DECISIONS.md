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
