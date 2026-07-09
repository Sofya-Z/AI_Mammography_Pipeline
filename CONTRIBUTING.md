# Contributing

## Branches
- `main` — stable, released.
- `develop` — integration branch for the next release.
- `feature/*` — one branch per step or experiment.

Open PRs against `develop`.

## Before a PR
```bash
pip install -e ".[dev]"
ruff check mammokit scripts tests
pytest -q tests/test_smoke.py
```

## Conventions
- New logic goes in `mammokit/`; CLI entry points in `scripts/`.
- Keep every config value overridable from the command line (`key.subkey=value`).
- Never commit patient data, weights, or anything under `data/`.
