# Frozen suites

Copied byte-for-byte on 2026-09-19 from `/Users/nafis/Documents/personal/kev/evals/` at kev commit `cc954f2f66d86943688fdbfa07b6db18f82ed065` (github.com/jaredpalmer/kev, Apache-2.0).

Do not edit anything here. `hev.suite.load_split` checksums every file against its manifest. New suites go in a new directory.

See `docs/EVALS.md` for format, policy and metrics.

## v4 suites (added 2026-09-20)

`decision-v4/` and `transfer-v4/` were copied byte-for-byte on 2026-09-20 from `/Users/nafis/Documents/personal/kev/evals/v4/decision-v4/` and `/Users/nafis/Documents/personal/kev/evals/v4/transfer-v4/` at kev commit `20fa6268c8ceb226530be2fb5266ab2c36b37724` (2026-09-19). They live flat here, not under a `v4/` subdirectory, so `hev.suite.load_split("evals/decision-v4", ...)` works like the v2 suites. Every copied file was compared with `cmp` against the kev original and its sha256 checked against `manifest.json`.

Manifest digests (`shasum -a 256 evals/<suite>/manifest.json`):

- `decision-v4/manifest.json`: `1b33e566d114f9eafeff55b36c221fadb2a4ae358a1b9cc68006e82c7cfad8f1`
- `transfer-v4/manifest.json`: `31677c2256b406222e7d94ffdc0a02a70ce05746b9efe307876024c4e77291d1`

`decision-v4/train.jsonl` is intentionally absent. Its manifest entry (sha256 `cb55b79e037c9f5a0ef2de4efe79ec6ef5d7d8d21aa64e94eee7731d7eddce74`, 10,896 records) is real, but kev does not keep training partitions over 10 MB in git; they are mirrored on the Hub dataset `jaredpalmer/kev-suites` and fetched on first use by kev's `kev/suite.py::fetch_partition` (kev README, "Frozen research suites"). Evaluation on decision-v4 needs only `calibration.jsonl` and `development.jsonl` (and the locked `test.jsonl`), all present. Do not fabricate or regenerate `train.jsonl`; if Hev ever trains on v4, fetch it from the mirror and verify it against the manifest digest above.

`decision-v4/test.jsonl` is byte-identical to `decision-v2/test.jsonl` (sha256 `cd7d129a84232e0c7a4e92b4840a6dfe14c8bb93c5d1904b526fa8c085325d2d`). `transfer-v4/train.jsonl` and `transfer-v4/calibration.jsonl` are empty files, as in transfer-v2. See `docs/EVALS.md` for what v4 changes relative to v2.
