# EpiLink

[![PyPI version](https://img.shields.io/pypi/v/epilink.svg)](https://pypi.org/project/epilink/)
[![Python versions](https://img.shields.io/pypi/pyversions/epilink.svg)](https://pypi.org/project/epilink/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![codecov](https://codecov.io/gh/ydnkka/epilink/branch/master/graph/badge.svg)](https://codecov.io/gh/ydnkka/epilink)
[![DOI](https://zenodo.org/badge/1065508228.svg)](https://doi.org/10.5281/zenodo.20402075)

EpiLink scores how compatible pairs of sampled infections are with recent
transmission scenarios, using the difference in sampling dates and the consensus
genetic distance between the samples.

It is designed for epidemiological linkage workflows where you want to compare
observed pairs against hypotheses such as direct transmission, transmission with
hidden intermediates, or recent shared ancestry. Scores are compatibility
summaries, not posterior probabilities unless you combine them with an explicit
prior model.

## Features

- Scenario-based linkage scores for ancestor-descendant and common-ancestor
  hypotheses.
- Deterministic and stochastic mutation-process models.
- Scalar and NumPy-broadcasted scoring for batch workflows.
- Typed result objects with attribute access and lightweight dictionary-style
  compatibility.
- Simulation helpers for epidemic dates, genomic sequences, and pairwise case
  tables.
- A small CSV command-line interface for batch scoring.

## Installation

EpiLink requires Python 3.10 or newer.

Install the published package from PyPI:

```bash
python -m pip install epilink
```

Install from a local checkout for development:

```bash
git clone https://github.com/ydnkka/epilink.git
cd epilink
python -m pip install -e ".[dev]"
```

Alternatively, create the repository conda environment:

```bash
conda env create -f environment.yml
conda activate epilink
```

## Quick Start

```python
from epilink import EpiLink, InfectiousnessToTransmission

profile = InfectiousnessToTransmission(rng_seed=2026)

model = EpiLink(
    transmission_profile=profile,
    maximum_depth=2,
    mc_samples=20000,
    target=["ad(0)", "ca(0,0)"],
    mutation_process="stochastic",
)

result = model.score_pair(
    sample_time_difference=3.0,
    genetic_distance=2.0,
)

print(result.target_labels)
print(result.target_compatibility)
print(result.scenario_scores["ad(0)"].compatibility)
```

## Core Concepts

EpiLink scores observations against generated transmission scenarios.
`maximum_depth` controls how many hidden generations are included.

- `ad(0)`: direct ancestor-descendant transmission.
- `ad(1)`: ancestor-descendant transmission with one hidden intermediate.
- `ca(0,0)`: two sampled cases with a recent shared common ancestor.
- `ca(m_i,m_j)`: a common-ancestor scenario with `m_i` and `m_j` hidden
  generations on each branch.

Each individual scenario compatibility lies in `[0, 1]`. If `target` contains
multiple scenarios, `target_compatibility` is the sum across that target subset
and can be greater than `1`.

## Common Workflows

### Score a Target Subset

Use `score_target` when you only need the combined compatibility score:

```python
score = model.score_target(
    sample_time_difference=3.0,
    genetic_distance=2.0,
    target=["ad(0)", "ad(1)", "ca(0,0)"],
)

print(score)
```

You can also pass `Scenario` objects instead of string labels:

```python
from epilink import Scenario

score = model.score_target(
    sample_time_difference=3.0,
    genetic_distance=2.0,
    target=[
        Scenario.ancestor_descendant(0),
        Scenario.common_ancestor(0, 0),
    ],
)
```

### Score Many Pairs

`score_target` and `pairwise_model` broadcast NumPy inputs, so you can score a
grid or batch efficiently:

```python
import numpy as np

pairwise = model.pairwise_model(target=["ad(0)", "ca(0,0)"])

time_differences = np.array([[0.0], [2.0], [4.0]])
genetic_distances = np.array([[0.0, 1.0, 2.0, 3.0]])

scores = pairwise(time_differences, genetic_distances)
print(scores.shape)  # (3, 4)
```

### Simulate a Pairwise Case Table

The simulation helpers are useful for synthetic examples, tests, and downstream
benchmarking:

```python
import networkx as nx

from epilink import (
    InfectiousnessToTransmission,
    build_pairwise_case_table,
    simulate_epidemic_dates,
    simulate_genomic_sequences,
)

profile = InfectiousnessToTransmission(rng_seed=2026)
tree = nx.DiGraph(
    [
        ("case-0", "case-1"),
        ("case-0", "case-2"),
    ]
)

dated_tree = simulate_epidemic_dates(profile, tree, fraction_sampled=1.0)
simulated = simulate_genomic_sequences(profile, dated_tree, genome_length=500)
pair_table = build_pairwise_case_table(simulated, dated_tree)

print(pair_table.head())
```

`simulate_genomic_sequences(...)` returns a `SimulationResult` with `packed` and
optional `raw` sequence sets. Each sequence set exposes `deterministic` and
`stochastic` members.

### Score a CSV File

The command-line interface expects an input CSV with these columns:

- `sample_time_difference`
- `genetic_distance`

```bash
epilink observations.csv \
  --output scored_observations.csv \
  --target "ad(0)" "ca(0,0)" \
  --maximum-depth 2 \
  --mc-samples 20000 \
  --mutation-process stochastic
```

The output CSV includes an `epilink_score` column.

## Mutation Models

- `mutation_process="deterministic"` compares the observation with expected
  mutation counts.
- `mutation_process="stochastic"` compares the observation with Poisson
  mutation-count draws.

The stochastic option is usually preferable when mutation-count variability
should contribute to the score.

## Reproducibility

- Randomness is controlled by the transmission profile RNG, or by the explicit
  `rng=` argument passed to `EpiLink`.
- For reproducible scores and simulations, construct profiles with a fixed
  `rng_seed` and reuse the resulting model instance.
- `score_pair(...)` and `pairwise_model(...)` use cached Monte Carlo draws, so
  repeated evaluations on the same model are stable unless you replace
  `draws_by_scenario`.

## Performance

- `EpiLink(...)` precomputes scenario draws, so model construction is the main
  fixed cost.
- Reuse one model instance for repeated scoring instead of rebuilding it inside
  loops.
- Use `pairwise_model(...)` when scoring many observations against the same
  target subset.
- See [docs/performance.md](docs/performance.md) for benchmark guidance.

## Documentation

- Model derivation: [docs/epilink.md](docs/epilink.md)
- Usage guide: [docs/usage_guide.ipynb](docs/usage_guide.ipynb)
- Workflow figure: [docs/epilink_schematic.pdf](docs/epilink_schematic.pdf)
- Performance guide: [docs/performance.md](docs/performance.md)

## Development

Run the test suite:

```bash
python -m pytest
```

Run formatting and lint checks:

```bash
python -m ruff check src tests
python -m black --check src tests
python -m mypy src
```

## Citation

If you use EpiLink in research, please cite the software metadata in
[CITATION.cff](CITATION.cff). The underlying infectiousness model is:

1. Hart WS, Maini PK, Thompson RN. High infectiousness immediately before
   COVID-19 symptom onset highlights the importance of continued contact
   tracing. *eLife*. 2021;10:e65534. <http://dx.doi.org/10.7554/eLife.65534>

## License

EpiLink is distributed under the MIT license. See [LICENSE](LICENSE).
