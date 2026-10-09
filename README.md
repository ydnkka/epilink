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
`stochastic` members. The original, unmutated reference is always available as
`simulated.reference_sequence` (or `simulated["reference_sequence"]`), including
when `return_raw=False`. It is a one-dimensional NumPy `int8` array of length
`genome_length`, encoded as `0=A`, `1=C`, `2=G`, and `3=T`.

The decoded A/C/G/T string is available as `simulated.reference_sequence_string`
(or `simulated["reference_sequence_string"]`) and is also included in `to_dict()`.
To export node sequences and the reference together for phylogenetic inference:

```python
simulated.packed.stochastic.write_fasta("sequences.fasta")
with open("sequences.fasta", "a") as fasta:
    fasta.write(f">reference\n{simulated.reference_sequence_string}\n")
```

Use the `reference` sequence as the outgroup in your tree inference/rooting tool.
Use `simulated.packed.deterministic` to export the deterministic sequences instead.

### Infer Sampled Phylogenies

See [the small demonstration notebook](docs/phylogeny_demo.ipynb) for simulation,
inference, and side-by-side genetic and dated tree plots.

Install the optional tree-parsing dependency:

```bash
python -m pip install 'epilink[phylogeny]'
```

Install [IQ-TREE](https://iqtree.github.io/#download) version 2.0.6 or later
separately. EpiLink discovers `iqtree3`, `iqtree2`, or `iqtree` on `PATH`, or you
can provide its path with `iqtree_executable`.

Using the simulation above:

```python
from epilink import build_phylogenetic_tree

phylogeny = build_phylogenetic_tree(
    simulated,
    dated_tree,  # Epidemic graph supplying sampled flags and sampling dates
    sequence_model="stochastic",  # Or "deterministic"
    dated=True,
    output_dir="phylogeny",
    model="JC",
    threads=1,
    seed=2026,
)

raw_tree = phylogeny.raw_tree      # Biopython Tree; substitutions/site
time_tree = phylogeny.dated_tree   # Biopython Tree; days
print(phylogeny.node_dates)
print(phylogeny.output_paths["raw_tree"])
print(phylogeny.output_paths["dated_tree"])
```

The helper lives in `epilink.simulation.phylogeny`. It selects only nodes marked
`sampled=True`, requires at least three sampled cases, and works with
`return_raw=False`. Sequence rows are matched by case ID; transmission edges
are not used to infer the phylogeny.

IQ-TREE infers a maximum-likelihood genetic tree with the original reference as
the outgroup. LSD2 then dates the sampled-case tree using numeric `sample_date`
values. The genetic tree retains the reference (`phylogeny.reference_name`);
the dated tree excludes it because the reference has no sampling date. Dates
and branch lengths remain in simulation days, without assigning calendar dates.

`model="JC"` matches the simulator's symmetric nucleotide changes. Other IQ-TREE
models, including `"GTR+G"` and automatic selection with `"MFP"`, are supported.
The clock rate is estimated by default; supply `clock_rate` in
**substitutions/site/day** to fix it. Identical sampling dates require a fixed
rate. Use `dated=False` to infer only the genetic tree without sampling dates.
`timeout` optionally limits inference runtime in seconds.

Each call creates a separate `run-*` directory beneath `output_dir`, containing
sampled FASTA, label mappings, genetic Newick, dated Newick/NEXUS, node-date TSV,
sampling dates, IQ-TREE/LSD2 reports, and logs. Alignment labels are safe backend
IDs recorded in `taxon_labels.tsv`; returned/exported tree labels are the
original case IDs. Inputs are preserved, and failures raise `PhylogenyError`
with the inference log path. Dated Newick is exported from LSD2's time-scaled
NEXUS, since its own `.timetree.nwk` file uses substitution lengths.

#### Use an Aligned FASTA and Sampling Dates

`build_phylogenetic_tree_from_fasta()` accepts an existing aligned DNA FASTA,
including gaps and IUPAC ambiguity bases. Choose a reference already in it:

```python
from epilink import build_phylogenetic_tree_from_fasta

phylogeny = build_phylogenetic_tree_from_fasta(
    "aligned_samples_with_reference.fasta",
    reference_id="reference",
    dates="sampling_dates.csv",
    output_dir="phylogeny",
)
```

Or supply a separate, single-record reference FASTA aligned to the same columns:

```python
phylogeny = build_phylogenetic_tree_from_fasta(
    "aligned_samples_only.fasta",
    reference_fasta="aligned_reference.fasta",
    dates={
        "sample-1": "2026-10-01",
        "sample-2": "2026-10-04",
        "sample-3": "2026-10-09",
    },
)
```

Supply exactly one reference form. FASTA IDs are the first token of each header,
and at least three sample sequences are needed in addition to the reference.
All sequences must already share alignment columns. The default `model="MFP"`
uses IQ-TREE ModelFinder; inference options match the simulation helper.

Date files can be CSV or TSV with `case_id` and `sample_date` columns:

```csv
case_id,sample_date
sample-1,2026-10-01
sample-2,2026-10-04
sample-3,2026-10-09
```

Date mappings/files may instead use numeric days, such as
`{"sample-1": 0.0, "sample-2": 3.5, "sample-3": 8.0}`. Calendar values can also
be Python `datetime.date` objects. Use one date format per dataset; sample IDs
must match exactly (including leading zeros), with a date for every sample.
An optional reference date is ignored because the reference is an outgroup.

Both interfaces return `PhylogenyResult`. For calendar inputs, `date_origin` is
the earliest sample date; `node_dates.date` and `node_dates.sample_date` are days
relative to it. Additional `calendar_date` and `sample_calendar_date` columns
retain reconstructed and observed calendar datetimes, including ancestral dates
before the origin. The origin is also exported as `date_origin.txt`, with calendar
dates in node-date TSV and dated-tree comments. Numeric time origins are
preserved and have `date_origin=None`. Branch lengths and clock rates remain in
days and substitutions/site/day for either input format. Use `dated=False` for
a genetic tree without sampling dates. Source files are preserved.

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
