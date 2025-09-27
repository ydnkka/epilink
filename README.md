# epilink

[![CI](https://github.com/ydnkka/epilink/actions/workflows/CI.yml/badge.svg)](https://github.com/ydnkka/epilink/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/ydnkka/epilink/branch/master/graph/badge.svg?token=YOUR_CODECOV_TOKEN)](https://codecov.io/gh/ydnkka/epilink)

## Overview
`epilink` estimates epidemiological linkage probabilities from temporal and genetic data.  
It implements the SARS-CoV-2 **E/P/I infectiousness model** using **Numba** for fast computations.

**Key features:**

- Estimates transmission linkage probabilities from genetic distances and sampling intervals.
- Parameterised infectiousness profiles (TOIT, TOST) for flexibility.
- High-performance computation with optional JIT for large-scale datasets.
- CLI and Python API for integration in pipelines or analysis scripts.

---

## Installation

**Using pip (editable, with dev dependencies):**

```bash
git clone https://github.com/ydnkka/epilink.git
cd epilink
pip install -e .[dev]
````

**Optional: Using a conda environment**

```bash
conda create -n epilink python=3.11
conda activate epilink
pip install -e .[dev]
```

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
# Show help
epilink --help

# Example command
epilink --genetic-distance 2 --sampling-interval 4 --num-simulations 200
```

Common options:
- -m/--intermediate-generations "0,1,2"
- -M/--no-intermediates 10
- --relax-rate (use relaxed molecular clock)
- --subs-rate 1e-3, --subs-rate-sigma 0.33
- --seed 12345

## Development

### Run tests
```bash
pytest --cov=epilink --cov-report=term-missing
```

### Enforce code style
```bash
ruff check .
black --check .
mypy src/epilink
```

### Pre-commit hooks
```bash
pre-commit install
pre-commit run -a
```

## License

MIT License — see [LICENSE](LICENSE) for details.


## Reference

Hart WS, Maini PK, Thompson RN (2021). High infectiousness immediately before COVID-19 symptom onset highlights the importance of continued contact tracing. eLife, 10:e65534. https://doi.org/10.7554/eLife.65534




# epilink

Epidemiological linkage probabilities from genetic and temporal distances using a mechanistic SARS-CoV-2 infectiousness model (E/P/I) and Numba-accelerated kernels.

- Infectiousness model: Hart et al. (2021) E/P/I with variable infectiousness (TOST, TOIT)
- Monte Carlo simulations for epidemiological quantities
- Numba-accelerated probability kernels, with a safe pure-Python fallback

## Install

```bash
# development install
pip install -e .[dev]
```

## Quick start

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

## Development

- Run tests: `pytest`
- Lint/format: `ruff check .` and `black .`
- Type-check: `mypy src/epilink`
- Pre-commit hooks: `pre-commit install` then `pre-commit run -a`

## CLI

Install/activate your conda env, then:

- Single pair:
  epilink point -g 2 -t 4 --nsims 10000

- Multiple pairs (CSV to stdout):
  epilink point -g 0 1 2 -t 0 2 5 --nsims 500 > out.csv

- Grid (CSV file):
  epilink grid --g-start 0 --g-stop 5 --g-step 1 --t-start 0 --t-stop 12 --t-step 3 --nsims 10000 --out grid.csv

Common options:
- -m/--intermediate-generations "0,1,2"
- -M/--no-intermediates 10
- --relax-rate (use relaxed molecular clock)
- --subs-rate 1e-3, --subs-rate-sigma 0.33
- --seed 12345

## Reference

Hart WS, Maini PK, Thompson RN (2021). High infectiousness immediately before COVID-19 symptom onset highlights the importance of continued contact tracing. eLife, 10:e65534. https://doi.org/10.7554/eLife.65534
