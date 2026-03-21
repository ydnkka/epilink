These instructions apply to the entire repository.
- `epilink` is a Python package for epidemiological linkage inference from sampling-time differences and consensus genetic distances.
- The repository uses a `src/` layout and targets Python 3.10 or newer.
- Core code is split between scoring/model logic and simulation helpers.
- `src/epilink/model/`: scoring logic, transmission profiles, and model parameters.
- `src/epilink/simulation/`: outbreak and genome simulation helpers.
- `tests/`: unit tests for model, profiles, parameters, and simulation utilities.
- `docs/`: manuscript assets, figures, and notebooks.
- `README.md`: installation, quick start, and public API examples.
- `CONTRIBUTING.md`: contributor workflow and validation expectations.
- Use Python 3.10 or newer.
- Preferred development install: `python -m pip install -e ".[dev]"`.
- Conda workflow: `conda env create -f environment.yml` then `conda activate epilink`.
- Before finalizing a change, make sure the dependencies needed for the affected workflow are installed.
1. Inspect the relevant implementation and tests before editing.
2. Make the smallest change that fully solves the task.
3. Update or add focused tests when behavior changes.
4. Update documentation if the public API, usage, or outputs change.
5. Run targeted validation first, then broader checks as needed.
- Keep changes focused, minimal, and consistent with the existing codebase.
- Do not modify unrelated files or revert user changes outside the task.
- Preserve the public API exposed from `src/epilink/__init__.py` unless the task explicitly requires an API change.
- Match the existing style: readable NumPy-oriented code, small helper functions, and straightforward typing where already present.
- Prefer root-cause fixes over superficial patches.
- Avoid broad refactors unless they are clearly required by the task.
- Run targeted tests first for the area you changed, then broader validation if needed.
- Standard validation commands:
  - `ruff check .`
  - `pytest`
- Add or update tests when changing scoring behavior, parameters, profiles, or simulation helpers.
- Prefer deterministic tests and fixed seeds for stochastic logic.
- Avoid Monte Carlo-heavy tests unless they validate a specific regression or contract.
- Update `README.md` when installation steps, public usage, or top-level behavior change.
- Update relevant files under `docs/` when mathematical descriptions, figures, notebooks, or worked examples change.
- Keep code examples aligned with the current public API.
- Do not commit caches, virtual environments, build artifacts, notebook checkpoints, or other generated files.
- Before finishing a code change, confirm `.gitignore` still covers any newly introduced generated artifacts.
- Do not commit `__pycache__` contents or local tool caches.
- Prefer `rg` and `rg --files` for search.
- Read large files in chunks.
- Prefer small, verifiable edits over large rewrites.
- When changing correctness-sensitive behavior, prioritize updating tests in `tests/` alongside the implementation.
- If a task changes user-facing behavior, update both code and documentation in the same pass.
