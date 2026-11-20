# Contributing

## Workflow

1. Open (or pick) an issue. Experiments use the *Experiment proposal* template: hypothesis, setup, decision metric.
2. Branch from `main`: `feat/…`, `fix/…`, `exp/…`, `docs/…`, `build/…`.
3. Keep PRs to one concern. Link the issue with `Closes #N`.
4. `make check` must pass locally; CI runs the same steps and is required for merge.
5. Squash-merge with a [Conventional Commits](https://www.conventionalcommits.org) title.

## Development setup

```bash
uv sync --extra dev --extra data --extra viz   # CPU torch from the PyTorch index
uv run pre-commit install
make check
```

## Experiment hygiene

- Every reported number must come from a file in `results/` produced by a committed config.
- Change one factor at a time against the stated baseline; say what was held fixed.
- Report negative and null results too.
- Never load third-party checkpoints with `weights_only=False`.
