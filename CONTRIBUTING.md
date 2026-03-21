# Contributing

Thanks for contributing to EpiLink.

## Development setup

Create or activate an environment with Python 3.10 or newer, then install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

If you are using the local conda environment:

```bash
conda activate epilink
python -m pip install -e ".[dev]"
```

## Before opening a change

- Run `ruff check .`
- Run `pytest`
- Add or update tests for behavior changes
- Keep public examples and docs in sync with the code

## Project layout

- `src/epilink/`: package source code
- `tests/`: unit tests
- `docs/`: figures, manuscript assets, and usage notebook

## Testing expectations

- Add focused unit tests when changing scenario scoring, natural-history parameters, or transmission profiles.
- Prefer deterministic fixtures for logic-heavy tests.
- Keep Monte Carlo-heavy examples in docs or notebooks unless they validate a concrete behavior.
- When randomness matters, use fixed RNG seeds and assert against stable contracts rather than incidental draws.

## Documentation

If you change user-facing behavior, update the relevant documentation:

- `README.md` for installation or high-level API usage
- `docs/epilink_usage_notebook.ipynb` for worked examples and figures
- `docs/assets/epilink.pdf` or source assets when the mathematical description changes
- Document public return types and input guarantees whenever the API contract changes.
- Update `docs/performance.md` when changing caching, vectorization, or benchmark guidance.

## Style

- Follow the existing code style and type-hinting conventions.
- Prefer small, composable tests over broad end-to-end cases.
- Avoid reverting unrelated work already present in the repository.
