# EpiLink

[![codecov](https://codecov.io/gh/ydnkka/epilink/branch/master/graph/badge.svg)](https://codecov.io/gh/ydnkka/epilink)

EpiLink scores how compatible a pair of samples is with recent transmission scenarios using sampling-time differences and consensus genetic distance.

It is useful when you have:

- a sampling-time difference in days
- a consensus genetic distance in mutations
- a question like "is this pair more compatible with direct transmission or a recent shared ancestor?"

EpiLink returns typed score results with per-scenario compatibility summaries and can also sum scores across a user-defined target subset such as `["ad(0)", "ca(0,0)"]`.

## Installation

From PyPI:

```bash
pip install epilink
```

Clone the repository first if you are starting from GitHub:

```bash
git clone https://github.com/ydnkka/epilink.git
cd epilink
```

The repository environment is the easiest way to get everything needed for the package, examples, and simulation helpers:

```bash
conda env create -f environment.yml
conda activate epilink
```

If you prefer `pip`:

```bash
python -m pip install -e .
python -m pip install networkx pandas
```

EpiLink requires Python 3.10 or newer.

## Scenario labels

- `ad(0)`: direct ancestor-descendant transmission
- `ad(1)`: ancestor-descendant transmission with one hidden intermediate
- `ca(0,0)`: a recent shared common ancestor with one branch to each sampled case
- `ca(m_i,m_j)`: a common-ancestor scenario with `m_i` and `m_j` hidden generations on each branch

`maximum_depth` controls how many of these latent scenarios are generated.

## Which method to use

- `score_pair(...)`: one observed pair, plus a full per-scenario breakdown
- `score_target(...)`: only the target score, for scalar or array inputs
- `pairwise_model(...)`: a cached scorer for repeatedly evaluating the same target subset

Each individual scenario compatibility lies in `[0, 1]`. If `target` contains multiple scenarios, `target_compatibility` is the sum across that subset, so it can be greater than `1`.

## API guarantees

- `score_pair(...)` returns a `PairCompatibilityResult` object with attribute access such as `result.target_labels` and `result.scenario_scores["ad(0)"].compatibility`.
- `simulate_genomic_sequences(...)` returns a `SimulationResult` object with `packed` and optional `raw` sequence sets, each exposing `deterministic` and `stochastic` members.
- `score_target(...)` returns a scalar `float` for scalar inputs and a NumPy array for broadcastable array inputs.
- Result objects retain lightweight dictionary-style access for common keys to ease migration from earlier releases.

## Reproducibility

- Randomness is controlled by the transmission profile RNG, or by the explicit `rng=` argument passed to `EpiLink`.
- For reproducible scores and simulations, construct profiles with a fixed `rng_seed` and reuse the resulting model instance.
- `score_pair(...)` and `pairwise_model(...)` use cached Monte Carlo draws, so repeated evaluations on the same model are stable unless you replace `draws_by_scenario`.

## Performance and caching

- `EpiLink(...)` precomputes scenario draws up front, so model construction is the main fixed cost.
- Reuse one model instance for repeated scoring instead of rebuilding it inside loops.
- Use `pairwise_model(...)` when scoring many observations against the same target subset.
- For benchmarking and a fuller discussion, see [docs/performance.md](docs/performance.md) and run `python -m docs.benchmark_api`.

## Quick start

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

## More examples

### Score only a target subset

Use `score_target` when you only care about the combined score:

```python
score = model.score_target(
    sample_time_difference=3.0,
    genetic_distance=2.0,
    target=["ad(0)", "ad(1)", "ca(0,0)"],
)

print(score)
```

### Use `Scenario` objects instead of strings

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

### Score many pairs at once

`score_target` and `pairwise_model` broadcast NumPy inputs, so you can score a whole grid or batch efficiently:

```python
import numpy as np

pairwise = model.pairwise_model(target=["ad(0)", "ca(0,0)"])

time_differences = np.array([[0.0], [2.0], [4.0]])
genetic_distances = np.array([[0.0, 1.0, 2.0, 3.0]])

scores = pairwise(time_differences, genetic_distances)
print(scores.shape)  # (3, 4)
```

### Build a toy simulated pair table

The simulation helpers are useful for generating synthetic examples and benchmarking downstream workflows:

```python
import networkx as nx

from epilink import (
    build_pairwise_case_table,
    simulate_epidemic_dates,
    simulate_genomic_sequences,
)

tree = nx.DiGraph(
    [
        ("case-0", "case-1"),
        ("case-0", "case-2"),
    ]
)

dated_tree = simulate_epidemic_dates(profile, tree, fraction_sampled=1.0)
simulated = simulate_genomic_sequences(profile, dated_tree, genome_length=500)
pair_table = build_pairwise_case_table(simulated.packed, dated_tree)

print(pair_table.head())
```

## Mutation models

- `mutation_process="deterministic"` compares the observation with expected mutation counts
- `mutation_process="stochastic"` compares the observation with Poisson mutation-count draws

The stochastic option is usually the better choice when you want mutation-count variability to be part of the score.

## Background and usage guide
- Model derivation: [docs/epilink.md](docs/epilink.md)
- Usage guide: [docs/usage_guide.ipynb](docs/usage_guide.ipynb)
- Workflow figure: [docs/epilink_schematic.pdf](docs/epilink_schematic.pdf) (source: [docs/epilink_schematic.tex](docs/epilink_schematic.tex))
- Performance guide: [docs/performance.md](docs/performance.md)

## Citation

If you use EpiLink in research, please cite the software metadata in [CITATION.cff](CITATION.cff). The underlying infectiousness model is:

1. Hart WS, Maini PK, Thompson RN. High infectiousness immediately before COVID-19 symptom onset highlights the importance of continued contact tracing. *eLife*. 2021;10:e65534. <http://dx.doi.org/10.7554/eLife.65534>

## License

MIT. See [LICENSE](LICENSE).
