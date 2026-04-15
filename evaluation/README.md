# Evaluation

This subproject contains the reproducible evaluation workflows for the `epilink`
project. It includes:

- preprocessing of raw SCoVMod transmission outputs into a canonical tree,
- sparsification analysis for score thresholds,
- temporal stability analysis,
- synthetic robustness experiments,
- baseline performance analysis with bootstrap statistics, and
- Boston dataset clustering analysis.

The workflow is configuration-driven through `config.yaml`, and generated
artifacts are written under `results/`.

## Project layout

### Pipeline entry-point modules

These modules contain a `main()` function and are executed by the Snakemake
workflow (or directly):

- `src/evaluation/scovmod.py` — Parses SCoVMod CSVs and builds the canonical
  GML transmission tree.
- `src/evaluation/sparsification.py` — Computes score surfaces, edge-retention
  summaries, and optimal per-model sparsification thresholds.
- `src/evaluation/stability.py` — Evaluates cumulative temporal stability as
  cases accrue over time.
- `src/evaluation/experiments.py` — Runs the synthetic robustness experiment
  grid across configured scenarios and conditions.
- `src/evaluation/baseline.py` — Runs the matched-baseline scenario and
  computes a rich statistical summary (bootstrap AP CIs, PRG, Brier score).
- `src/evaluation/figures.py` — Assembles all manuscript figures from completed
  evaluation outputs and writes PDF + TIFF exports.

### Shared library modules

These modules expose helpers imported by one or more entry-point modules and
have no `__main__` block intended for direct use:

- `src/evaluation/config.py` — Config loading, path resolution, logging setup,
  and scenario/run-spec construction.
- `src/evaluation/specs.py` — Shared constants (column names, model keys),
  parameter-expansion helpers, and score-metadata lookup.
- `src/evaluation/models.py` — EpiLink scorer construction and logistic-
  regression scoring helpers.
- `src/evaluation/evaluate.py` — Core scenario evaluation logic: simulate,
  score, compute AP / best-F1 / stability for all models.
- `src/evaluation/leiden.py` — igraph graph-construction and multi-restart
  Leiden clustering helpers.
- `src/evaluation/metrics.py` — BCubed precision/recall/F1, overlap metrics
  between consecutive partitions, and cluster-composition summaries.
- `src/evaluation/heterogeneity.py` — Negative-binomial MLE (with MoM fallback)
  and bootstrap CIs for transmission heterogeneity.
- `src/evaluation/plotting.py` — PLOS figure-dimension constants, shared
  condition/scenario/model display mappings, seaborn theme, and
  publication-grade figure export.

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

Treat `data/raw/` and `data/processed/` as read-only input directories.

### Canonical tree output

The generated SCoVMod tree is the canonical input consumed by all downstream
synthetic workflows:

- `results/scovmod/scovmod_tree.gml`

This location is configured via `outputs.scovmod.tree_path` in `config.yaml`.

### Results directories

By default, all outputs are placed under `results/`:

- `results/scovmod/` — GML tree, heterogeneity JSON, degree distributions,
  and component/tree summary parquets.
- `results/sparsification/` — Score surfaces, edge-retention summaries, and
  `optimal_thresholds.json`.
- `results/stability/` — Temporal stability parquets per model and resolution-
  selection summary.
- `results/synthetic/` — Full experiment results, raw baseline scores, and
  baseline statistical summary.
- `results/boston/` — Cluster composition, cluster sizes, and cluster summary.
- `results/figures/` — PDF and TIFF figure exports.

## Logging

All pipeline modules use Python's standard `logging` library with a shared
format:

```
YYYY-MM-DD HH:MM:SS [LEVEL] module_name: message
```

Each module call to `configure_logging()` adds:

1. A **console handler** (stderr) — active in all contexts.
2. A **file handler appending to the unified pipeline log** —
   `results/logs/pipeline.log` (configured via `outputs.logs.pipeline_log` in
   `config.yaml`).

When running via Snakemake, each module's stdout+stderr is also captured
separately to its own per-rule log file (see below). The unified pipeline log
collects a single chronological record across all modules, which is useful for
auditing the full run order and timings.

## Configuration

The pipeline is controlled by `config.yaml`. Important sections include:

- `paths` — locations of raw and processed input data.
- `outputs` — output root, per-workflow output subdirectories, and log paths.
- `workflows` — workflow-specific parameters for SCoVMod, sparsification,
  stability, and Boston analysis.
- `generation_baseline`, `inference_baseline`, `perturbations`, `conditions`,
  and `execution` — synthetic evaluation design.

Most changes to experiment behaviour should be made in `config.yaml`, not in
code.

## Snakemake workflow

The `Snakefile` executes the core analysis in this order:

1. `scovmod` — build transmission tree
2. `sparsification` — compute score surfaces and optimal thresholds
3. `baseline` — run matched-baseline evaluation and compute rich statistics
4. `stability` — evaluate cumulative temporal stability
5. `experiments` — run the full synthetic robustness grid
6. `boston` — run the empirical Boston clustering analysis

Rules 3 (`baseline`) and 4 (`stability`) are independent of each other and
both depend only on `sparsification`, so Snakemake can schedule them in
parallel when more than one core is available.

This ordering ensures the SCoVMod tree is created before any downstream
synthetic analyses depend on it.

### Dry run

```bash
PYTHONPATH=src:src/evaluation snakemake -n --cores 1
```

### Run the full workflow

```bash
PYTHONPATH=src:src/evaluation snakemake --cores 1
```

Per-rule log files are written to:

- `results/logs/scovmod.log`
- `results/logs/sparsification.log`
- `results/logs/baseline.log`
- `results/logs/stability.log`
- `results/logs/experiments.log`
- `results/logs/boston.log`

The unified log across all rules is appended to:

- `results/logs/pipeline.log`

### Run a specific step

```bash
PYTHONPATH=src:src/evaluation snakemake --cores 1 sparsification
```

### Use a different config file

```bash
PYTHONPATH=src:src/evaluation snakemake --cores 1 --config evaluation_config=path/to/config.yaml
```

## Running modules directly

Run modules without Snakemake from `evaluation/` with `PYTHONPATH=src:src/evaluation`:

```bash
PYTHONPATH=src:src/evaluation python3 -m evaluation.scovmod
PYTHONPATH=src:src/evaluation python3 -m evaluation.sparsification
PYTHONPATH=src:src/evaluation python3 -m evaluation.stability
PYTHONPATH=src:src/evaluation python3 -m evaluation.experiments
PYTHONPATH=src:src/evaluation python3 -m evaluation.boston
PYTHONPATH=src:src/evaluation python3 -m evaluation.baseline
```

Figures are generated separately (not part of the Snakemake pipeline):

```bash
PYTHONPATH=src:src/evaluation python3 -m evaluation.figures          # save PDF + TIFF
PYTHONPATH=src:src/evaluation python3 -m evaluation.figures --no-save  # preview only
```

## Recommended smoke checks

From `evaluation/`, quick import checks before a full run:

```bash
PYTHONPATH=src python3 -c "from evaluation.config import load_config; load_config()"
snakemake -n --cores 1
```

For synthetic experiment changes, checking config parsing and run-spec
construction is often faster than a full evaluation:

```bash
PYTHONPATH=src python3 -c "
from evaluation.config import build_run_specs, load_config
print(len(build_run_specs(load_config())), 'run specs')
"
```
