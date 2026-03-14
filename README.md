# epilink: Epidemiological Linkage from Temporal and Genetic Data

[![codecov](https://codecov.io/gh/ydnkka/epilink/branch/master/graph/badge.svg)](https://codecov.io/gh/ydnkka/epilink)

Estimate the probability that two cases are epidemiologically linked from their temporal and genetic distances. Implements a mechanistic SARS‑CoV‑2 infectiousness model (E/P/I) with optional Numba acceleration. Usable from Python or the command line.

By default, linkage estimates follow the appendix definition of recent transmission and use only `M=0` genetic compatibility, i.e. `included_intermediate_counts=(0,)`.

## Features

- Estimate P(link | genetic distance g, temporal gap t)
- Parameterised infectiousness profiles (InfectiousnessToTransmissionTime/SymptomOnsetToTransmissionTime; configurable)
- Fast simulation kernels with optional JIT (Numba)
- Python API plus a lightweight CLI for batch runs

---

## Installation

From PyPI:

```bash
pip install epilink
```

Recommended (conda/mamba):

```bash
# Create a fresh env (uses compiled deps from conda-forge)
conda create -n epilink -c conda-forge python=3.11 numpy scipy numba networkx pandas pip
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
from epilink import InfectiousnessToTransmissionTime, MolecularClock, estimate_linkage_probability, estimate_linkage_probability_grid

toit = InfectiousnessToTransmissionTime(rng_seed=123)
clock = MolecularClock(use_relaxed_clock=False, rng_seed=123)

# Probability that a pair with 2 SNPs and 4 days apart is directly linked (m=0)
p = estimate_linkage_probability(
    transmission_profile=toit,
    clock=clock,
    genetic_distance=2,
    temporal_distance=4,
    included_intermediate_counts=0,
    num_simulations=10_000,
)
print("P(link):", p)

# Grid over genetic distances (SNPs) and temporal gaps (days)
gd = np.arange(0, 6)       # 0..5 SNPs
td = np.arange(0, 15, 3)   # 0..12 days, step 3
mat = estimate_linkage_probability_grid(
    transmission_profile=toit,
    clock=clock,
    genetic_distances=gd,
    temporal_distances=td,
    num_simulations=10_000,
)
print(mat)
```

### CLI

```bash
# Help
epilink --help
epilink point --help
epilink grid --help

# Single pair
epilink point -g 2 -t 4 --num-simulations 200

# Multiple pairs (CSV to stdout)
epilink point -g 0 1 2 -t 0 2 5 --num-simulations 500 > out.csv

# Custom infectiousness profile and numerical grid bounds
epilink point -g 2 -t 4 --num-simulations 200 \
             --grid-min-days 0 --grid-max-days 40 \
             --incubation-shape 5.0 --incubation-scale 1.1 \
             --latent-shape 2.0 --symptomatic-rate 0.4 \
             --symptomatic-shape 1.2 --rel-presymptomatic-infectiousness 2.0

# Grid (CSV to file)
epilink grid --genetic-start 0 --genetic-stop 5 --genetic-step 1 \
             --temporal-start 0 --temporal-stop 12 --temporal-step 3 \
             --num-simulations 10000 --output grid.csv
```

Commonly used options (see `--help` for full list):
- `-m, --included-intermediate-counts` default `"0"`; e.g. `"0,1,2"` to include longer chains
- `--relaxed-clock`
- `--substitution-rate 1e-3`
- `--relaxed-clock-sigma 0.33`
- `--seed 12345`
- `--grid-min-days 0 --grid-max-days 60` (numerical grid bounds)
- `--incubation-shape`, `--incubation-scale`, `--latent-shape`, `--symptomatic-rate`, `--symptomatic-shape`, `--rel-presymptomatic-infectiousness`

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
- `python examples/quickstart.py`
- `python examples/grid_to_csv.py --output grid.csv`

Install plotting deps if needed:

```bash
mamba install -c conda-forge matplotlib seaborn
```

---

## License

MIT License (see [LICENSE](LICENSE))

---

## Development & Contributing

For developers:
- **Testing releases**: See [TESTPYPI.md](TESTPYPI.md) for instructions on testing package releases on TestPyPI before publishing to PyPI
- **Contributing**: Pull requests welcome! Please ensure tests pass and coverage remains high

---

## Contact

- Questions or issues: open an [issue](https://github.com/ydnkka/epilink/issues)
- Maintainer: [@ydnkka](https://github.com/ydnkka)

---

## Reference

Hart WS, Maini PK, Thompson RN (2021). High infectiousness immediately before COVID‑19 symptom onset highlights the importance of continued contact tracing. eLife, 10:e65534. https://doi.org/10.7554/eLife.65534
