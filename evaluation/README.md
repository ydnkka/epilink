# Evaluation

This subproject contains the reproducible evaluation workflows for the `epilink`
project. It includes:

- preprocessing of raw SCoVMod transmission outputs into a tree,
- sparsification analysis for score thresholds,
- temporal stability analysis,
- synthetic robustness experiments, and
- Boston dataset clustering analysis.

The workflow is configuration-driven through `config.yaml`, and generated
artifacts are written under `results/`.

## Project layout

- `config.yaml` — central experiment and workflow configuration.
- `Snakefile` — Snakemake workflow for the end-to-end pipeline.
- `requirements.txt` — evaluation-specific dependencies, including `-e ..`
  for the parent `epilink` package.
- `src/evaluation/scovmod.py` — builds the canonical transmission tree from
  raw SCoVMod outputs.
- `src/evaluation/sparsification.py` — computes score surfaces, edge-retention
  summaries, and optimal thresholds.
- `src/evaluation/stability.py` — evaluates cumulative temporal stability as
  cases accrue over time.
- `src/evaluation/experiments.py` — runs the synthetic robustness experiments
  across configured scenarios and conditions.
- `src/evaluation/boston.py` — runs the Boston clustering analysis.
- `src/evaluation/config.py` — config loading and path resolution helpers.

## Environment setup

Use Python 3.10+ and install both the parent package and the evaluation
dependencies from the `evaluation/` directory:

```bash
python3 -m pip install -e ..
python3 -m pip install -r requirements.txt
```

The `requirements.txt` file intentionally keeps the parent package dependency
as `-e ..`.

## Inputs and outputs

### Inputs

- Raw SCoVMod inputs are read from:
  - `data/raw/scovmod/InfectedIndividuals.1.csv`
  - `data/raw/scovmod/TransmissionEvents.1.csv`
- Processed Boston inputs are read from:
  - `data/processed/boston/boston_metadata.parquet`
  - `data/processed/boston/boston_pairwise_distances.parquet`

Treat `data/raw/` and `data/processed/` as input data directories.

### Canonical tree output

The generated SCoVMod tree is the canonical tree consumed by downstream
workflows:

- `results/scovmod/scovmod_tree.gml`

This location is configured via `outputs.scovmod.tree_path` in `config.yaml`.

### Results directories

By default, all outputs are placed under `results/`:

- `results/scovmod/` — generated tree, heterogeneity summary, tree diagnostics.
- `results/sparsification/` — score surfaces, edge retention, optimal
  thresholds.
- `results/stability/` — temporal stability summaries and resolution selection.
- `results/synthetic/` — synthetic experiment results.
- `results/boston/` — Boston clustering outputs.

## Configuration

The pipeline is controlled by `config.yaml`. Important sections include:

- `paths` — locations of raw and processed input data.
- `outputs` — output root and per-workflow output paths.
- `workflows` — workflow-specific parameters for SCoVMod, sparsification,
  stability, and Boston analysis.
- `generation_baseline`, `inference_baseline`, `perturbations`, `conditions`,
  and `execution` — synthetic evaluation design.

Most changes to experiment behavior should be made in `config.yaml`, not in
code.

## Snakemake workflow

The `Snakefile` executes the analysis in this order:

1. `scovmod`
2. `sparsification`
3. `stability`
4. `experiments`
5. `boston`

This ordering ensures that the SCoVMod-generated tree is created first and used
by downstream synthetic analyses.

### Dry run

```bash
snakemake -n --cores 1
```

### Run the full workflow

```bash
snakemake --cores 1
```

### Run a specific step

```bash
snakemake --cores 1 sparsification
```

### Use a different config file

```bash
snakemake --cores 1 --config evaluation_config=path/to/config.yaml
```

## Running modules directly

If you want to run modules without Snakemake, do so from `evaluation/` with
`PYTHONPATH=src`:

```bash
PYTHONPATH=src python3 -m evaluation.scovmod
PYTHONPATH=src python3 -m evaluation.sparsification
PYTHONPATH=src python3 -m evaluation.stability
PYTHONPATH=src python3 -m evaluation.experiments
PYTHONPATH=src python3 -m evaluation.boston
```

## Recommended smoke checks

From `evaluation/`, small checks are usually enough before a full run:

```bash
PYTHONPATH=src python3 -c "from evaluation.config import load_config; load_config()"
snakemake -n --cores 1
```

For synthetic experiment changes, checking config parsing and run-spec
construction is often faster than a full evaluation:

```bash
PYTHONPATH=src python3 -c "from evaluation.config import build_run_specs, load_config; print(len(build_run_specs(load_config())))"
```
