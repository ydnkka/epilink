# epilink

## Overview

`epilink` estimates probabilities that two cases are epidemiologically linked from their temporal and genetic distances. It implements a mechanistic SARS‑CoV‑2 E/P/I infectiousness model, with optional Numba acceleration.

Key features:
- Estimate P(link | genetic distance g, temporal gap t)
- Parameterised infectiousness profiles (TOIT, TOST)
- High‑performance kernels with optional JIT
- Python API and CLI for pipelines and scripts

---

## Installation

Recommended (conda/mamba):
```bash
# Create a fresh env with compiled deps from conda-forge
conda create -n epilink -c conda-forge python=3.11 numpy scipy numba pip
conda activate epilink

# Install your package in editable mode without touching conda-managed deps
pip install -e . --no-deps

# Dev tools (tests, lint, docs)
pip install "pytest>=7.3" "pytest-cov>=4.0" "mypy>=1.4" "ruff>=0.5" "black>=24.1" \
            "pre-commit>=3.3" "mkdocs>=1.5" "mkdocs-material>=9.5" "mkdocstrings[python]>=0.24"
```

Alternative (pip/venv):
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

Notes:
- Prefer conda-forge for NumPy/SciPy/Numba, especially on Apple Silicon.
- In a conda env, avoid pip-installing compiled deps; use pip install -e . --no-deps for your package.

---

## Usage

### Python API

```python
import numpy as np
from epilink import (
    estimate_linkage_probability,
    pairwise_linkage_probability_matrix,
    InfectiousnessParams,
)

# Probability that a pair with 2 SNPs and 4 days apart is linked (directly, m=0)
p = estimate_linkage_probability(
    genetic_distance=2,
    sampling_interval=4,
    intermediate_generations=(0,),
    num_simulations=10_000,
)
print("P(link):", p)

# Grid over distances and temporal gaps
gd = np.arange(0, 6)       # 0..5 SNPs
td = np.arange(0, 15, 3)   # 0..12 days
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

Common options:
- -m / --intermediate-generations "0,1,2"
- -M / --no-intermediates 10
- --relax-rate
- --subs-rate 1e-3
- --subs-rate-sigma 0.33
- --seed 12345

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

Simple scripts are provided in examples/:
- Generate grid CSV: python examples/generate_grid.py ...
- Plot heatmap: python examples/plot_grid.py grid.csv --out grid.png

Install plotting deps if needed:
```bash
mamba install -c conda-forge matplotlib seaborn
```

---

## License

MIT License — see [LICENSE](LICENSE).

## Reference

Hart WS, Maini PK, Thompson RN (2021). High infectiousness immediately before COVID-19 symptom onset highlights the importance of continued contact tracing. eLife, 10:e65534. https://doi.org/10.7554/eLife.65534
```

Optional:
- Replace YOUR_CODECOV_TOKEN with your actual token or remove the Codecov badge until you configure uploads.
- If you use an environment.yml, you can add a short “Install via environment.yml” subsection.