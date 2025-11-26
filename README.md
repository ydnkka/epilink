# epilink: Epidemiological Linkage from Temporal and Genetic Data

[![codecov](https://codecov.io/gh/ydnkka/epilink/branch/master/graph/badge.svg)](https://codecov.io/gh/ydnkka/epilink)

Estimate the probability that two cases are epidemiologically linked from their temporal and genetic distances. Implements a mechanistic SARS‑CoV‑2 infectiousness model (E/P/I) with optional Numba acceleration. Usable from Python or the command line.

## Features

- Estimate P(link | genetic distance g, temporal gap t)
- Parameterised infectiousness profiles (e.g. TOST; configurable)
- Fast simulation kernels with optional JIT (Numba)
- Python API and CLI for pipelines and scripts

---

## Installation

Recommended (conda/mamba):

```bash
# Create a fresh env (uses compiled deps from conda-forge)
conda create -n epilink -c conda-forge python=3.11 numpy scipy numba pip
conda activate epilink
 git clone https://github.com/ydnkka/epilink.git
 cd epilink

# Install the package from source without touching conda-managed deps
pip install -e . --no-deps

# (Optional) Dev tools: tests, linting, docs
pip install "pytest>=7.3" "pytest-cov>=4.0" "mypy>=1.4" "ruff>=0.5" "black>=24.1" \
            "pre-commit>=3.3" "mkdocs>=1.5" "mkdocs-material>=9.5" "mkdocstrings[python]>=0.24"
```

Alternative (pip + venv):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

Notes:
- Prefer conda-forge for NumPy/SciPy/Numba (especially on Apple Silicon).
- In a conda env, avoid pip-installing compiled deps; use `pip install -e . --no-deps`.

---

## Quickstart

### Python API

```python
import numpy as np
from epilink import (
    estimate_linkage_probability,
    pairwise_linkage_probability_matrix,
    InfectiousnessParams,
)

# Probability that a pair with 2 SNPs and 4 days apart is directly linked (m=0)
p = estimate_linkage_probability(
    genetic_distance=2,
    sampling_interval=4,
    intermediate_generations=(0,),
    num_simulations=10_000,
    infectiousness_profile=InfectiousnessParams(),
)
print("P(link):", p)

# Grid over genetic distances (SNPs) and temporal gaps (days)
gd = np.arange(0, 6)       # 0..5 SNPs
td = np.arange(0, 15, 3)   # 0..12 days, step 3
mat = pairwise_linkage_probability_matrix(gd, td, num_simulations=10_000)
print(mat)
```

### CLI

```bash
# Help
epilink --help
epilink point --help
epilink grid --help

# Single pair
epilink point -g 2 -t 4 --nsims 200

# Multiple pairs (CSV to stdout)
epilink point -g 0 1 2 -t 0 2 5 --nsims 500 > out.csv

# Grid (CSV to file)
epilink grid --g-start 0 --g-stop 5 --g-step 1 \
             --t-start 0 --t-stop 12 --t-step 3 \
             --nsims 10000 --out grid.csv
```

Commonly used options (see `--help` for full list):
- `-m, --intermediate-generations` e.g. `"0,1,2"`
- `--relax-rate`
- `--subs-rate 1e-3`
- `--subs-rate-sigma 0.33`
- `--seed 12345`

---

## Mutation accumulation models

`epilink` supports two ways of modelling how mutations accumulate along the
transmission tree, conditional on the molecular clock:

- **Deterministic** (default):
  - Uses the legacy, time-based genetic kernel.
  - Effectively treats the expected number of mutations along each path as
    a fixed quantity.
  - Backwards‑compatible with previous versions of the package.
- **Poisson**:
  - Uses a mutation‑count kernel where the observed SNP distance is modelled
    as a Poisson random variable with mean equal to the expected number of
    mutations along the path.
  - Provides a more explicit mutation‑likelihood formulation.

You choose the model via the ``mutation_model`` argument in the Python API.

### Python API examples

Deterministic (legacy behaviour):

```python
from epilink import estimate_linkage_probability

p_det = estimate_linkage_probability(
    genetic_distance=2,
    temporal_distance=4,
    intermediate_generations=(0,),
    num_simulations=10_000,
    mutation_model="deterministic",   # default
    mutation_tolerance=0,              # use legacy time-based kernel
)
``

Deterministic in mutation space with an integer tolerance around the expected
mutation count:

```python
p_det_tol = estimate_linkage_probability(
    genetic_distance=2,
    temporal_distance=4,
    intermediate_generations=(0, 1),
    num_simulations=10_000,
    mutation_model="deterministic",
    mutation_tolerance=1,  # accept ±1 mutation around expected count
)
```

Poisson mutation accumulation:

```python
p_pois = estimate_linkage_probability(
    genetic_distance=2,
    temporal_distance=4,
    intermediate_generations=(0, 1, 2),
    num_simulations=10_000,
    mutation_model="poisson",
)
```

The same options are available in the lower‑level
:func:`epilink.genetic_linkage_probability` function via the ``mutation_model``
and ``mutation_tolerance`` keyword arguments.

> Note: if you do not pass ``mutation_model``, the default behaviour remains
> deterministic with the original time‑based genetic kernel, so existing code
> continues to work unchanged.

---

## Development

Run tests:

```bash
pytest --cov=epilink --cov-report=term-missing
# To count Python-side coverage of JIT kernels:
# NUMBA_DISABLE_JIT=1 pytest
```

Code quality:

```bash
ruff check .
black .
mypy src/epilink
```

Pre-commit:

```bash
pre-commit install
pre-commit run -a
```

Docs:

```bash
mkdocs serve
```

---

## Examples

Examples in `examples/`:
- Generate grid CSV: `python examples/generate_grid.py ...`
- Plot heatmap: `python examples/plot_grid.py grid.csv --out grid.png`

Install plotting deps if needed:

```bash
mamba install -c conda-forge matplotlib seaborn
```

---

## License

MIT License (see [LICENSE](LICENSE))

---

## Contact

- Questions or issues: open an [issue](https://github.com/ydnkka/epilink/issues)
- Maintainer: [@ydnkka](https://github.com/ydnkka)

---

## Reference

Hart WS, Maini PK, Thompson RN (2021). High infectiousness immediately before COVID‑19 symptom onset highlights the importance of continued contact tracing. eLife, 10:e65534. https://doi.org/10.7554/eLife.65534
