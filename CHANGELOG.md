# Changelog

## 0.1.0 — Unreleased

- Option-isolated Qwen3-0.6B decision model with PointerHead.
- Two-seed development replication and Hev / kev / Jev comparison.
- TypeSafe-compatible local API.
- Checksummed suites, offline mechanism tests and immutable result ledgers.
- Kev's `decision-v4` and `transfer-v4` suites copied byte-for-byte from kev `20fa626`, with on-demand, checksum-verified fetch of the git-ignored training partition from kev's Hub mirror.
- Vendored kev inference path (`hev/kev_model.py`, `hev/kev.py`) so a released kev checkpoint runs under kev's own packing through Hev's evaluator; validated by a predictor sanity gate that reproduces kev's published accuracy at zero difference.
- Evaluation-only comparison against the released `jaredpalmer/kev-0.6b` on v4 development (`hev.released`), under an exhaustive six-order protocol applied identically to every checkpoint.
- Controlled retrain on kev's decision-v4 partition under kev's published recipe, three seeds, with minimal-pair augmentation and a recipe guard (`hev.train --recipe kev-v4`, `hev.controlled`).
- Corrected two earlier claims: exact option-order invariance is not unique to Hev, since kev ships its own `option_isolation`; and flip rates from different numbers of orders were previously tabulated together, which understated kev's and Jev's figures roughly fourfold.
