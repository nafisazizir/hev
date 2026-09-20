# Contributing

Hev is a research repository. Keep changes small, auditable and reproducible.

## Development

```bash
uv sync --group dev --extra serve
PYTHONDONTWRITEBYTECODE=1 uv run pytest -p no:cacheprovider
uv build
```

Hub-dependent tests are opt-in with `HEV_HUB_TESTS=1`.

## Research rules

- Never edit a frozen suite under `evals/`.
- Never use the locked test split for model selection.
- Never train on sources listed in `hev.data.EVAL_ONLY`.
- Never overwrite a run directory.
- Link numerical claims to immutable artifacts.
- Preserve option-order and question-isolation tests when changing model code.

Open an issue before changing an evaluation protocol or public API.
