# Third-Party Notices

Hev is original research code with clearly identified adaptations and external assets.

## kev

Portions of `hev/api.py`, the augmentation helpers in `hev/data.py`, evaluation and bootstrap methodology, and the frozen suites under `evals/` are adapted or copied from [kev](https://github.com/jaredpalmer/kev) by Jared Palmer at commit `cc954f2f66d86943688fdbfa07b6db18f82ed065`.

kev is licensed under Apache-2.0. Copyright 2026 Jared Palmer. Modified Hev files retain source-level attribution where applicable.

`hev/kev_model.py` contains a verbatim, unmodified copy of kev's `kev/model.py` at commit `20fa6268c8ceb226530be2fb5266ab2c36b37724` (Apache-2.0, Copyright 2026 Jared Palmer), placed after the file's `# ---- verbatim kev/model.py below this line ----` marker. The sha256 of the copied upstream file is `839dd7632ec291ac4740035c86f780f8f13da2b74b54a146ce90ff0ce8de527b`; `tests/test_kev.py` checks the vendored text against it. The copy exists only so that released kev checkpoints (for example `jaredpalmer/kev-0.6b`) can be evaluated under kev's exact token packing (`<decide>` after the options, sequential positions, kev's `head.pt` schema); Hev's own model is `hev/model.py`, which does not derive from it.

`evals/decision-v4/` and `evals/transfer-v4/` are byte-for-byte copies of kev's `evals/v4/` at the same commit `20fa6268c8ceb226530be2fb5266ab2c36b37724` (provenance and digests in `evals/README.md`).

## Jev And System One

The project studies the public shape of TypeSafe's Jev and implements a compatible subset of the public System One API contract. No TypeSafe source code or private implementation is included. Jev, System One and TypeSafe names belong to their owner. Hev is not affiliated with or endorsed by TypeSafe.

The architectural starting point is Archer Hume's public article, [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked). The article is cited for ideas and observations; its text and figures are not redistributed here.

## Base Model

Hev adapters require `Qwen/Qwen3-0.6B-Base` at revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. The base weights are downloaded separately and are not redistributed in this repository. Use is governed by the model repository's license and terms.

## Datasets

The frozen suites contain converted records derived from these public datasets:

- `legacy-datasets/banking77`
- `google/boolq`
- `fancyzhx/ag_news`
- `nyu-mll/multi_nli`
- `SetFit/sst5`
- `Yelp/yelp_review_full`
- `CogComp/trec`
- `fancyzhx/dbpedia_14`
- `SetFit/amazon_reviews_multi_en`
- `stanfordnlp/imdb`
- `cais/mmlu`
- `dair-ai/emotion`
- `cardiffnlp/tweet_eval`
- `nyu-mll/glue` (QNLI)
- `google-research-datasets/paws`
- `allenai/sciq`

Dataset revisions and per-record provenance are recorded in each suite manifest. Dataset content remains subject to its source license and terms; the Hev Apache-2.0 license does not replace them.

The contrastive policy records were generated deterministically in kev from executable rules without an LLM and copied with the frozen suites.

## Dependencies

Python dependencies are installed separately through `uv.lock` and retain their own licenses. See `pyproject.toml` and `uv.lock` for exact packages and versions.
