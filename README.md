# Evaluation of EpiLink Pairwise Compatitbility Model

Evaluation of pairwise ranking by compatibility scores, genetic distance, logistic probabilities, and graph/phylogenetic clustering.

**Start with the [operational guide](OPERATIONS.md)** for setup, configuration, execution, and troubleshooting. The [synthetic baseline protocol](evaluation/01_synthetic_baseline/README.md) defines scientific questions, targets, metrics, and evaluation. Use the [output reference](OUTPUTS.md) for column definitions and metric formulas.

## Evaluation studies

| Study                                                                         | Purpose                                                                | Main evidence                                                     |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [00 — Synthetic diagnostics](evaluation/00_synthetic_diagnostics/README.md)   | Characterize feature ambiguity and graph controls                      | Exact feature cells and oracle-graph partitions                   |
| [01 — Synthetic baseline](evaluation/01_synthetic_baseline/README.md)         | Compare methods, select operating points, evaluate on held-out data    | Pairwise/clustering accuracy, development sweeps, frozen settings |
| [02 — Synthetic perturbation](evaluation/02_synthetic_perturbation/README.md) | Test EpiLink scorers under different inference and resolution settings | Paired clustering differences from controls                       |
| [03 — Boston application](evaluation/03_boston_application/README.md)         | Examine empirical transfer and partition agreement                     | Exposure summaries and frozen partitions                          |

Diagnostics and baseline use one [shared synthetic experiment](evaluation/shared_synthetic/README.md). Complete diagnostics before baseline. Perturbation and Boston run independently after baseline.

## Implementation status

| Capability               | Status                                               |
| ------------------------ | ---------------------------------------------------- |
| SCoVMod tree preparation | `epilink-evaluate scovmod --stage prepare`           |
| Synthetic diagnostics    | `epilink-evaluate diagnostics --stage all`           |
| Synthetic baseline       | Pairwise, clustering, selection, evaluation, reports |
| Parameter sensitivity    | `epilink-evaluate perturbation` (4 clustering modes) |
| Boston input preparation | `epilink-evaluate boston --stage prepare`            |
| Boston frozen transfer   | `epilink-evaluate boston --stage all`                |
| Output cleanup           | `epilink-evaluate reset-outputs`                     |

See [perturbation and Boston execution](OPERATIONS.md#12-perturbation-and-boston-application) and [clearing outputs](OPERATIONS.md#11-clear-outputs-with-reset-outputs).

## Install and run

Python ≥3.10, from the repository root:

```bash
python -m pip install -e '.[test]'
epilink-evaluate check --config evaluation/01_synthetic_baseline/config.yaml
python evaluation/00_synthetic_diagnostics/run.py --smoke --stage all
python evaluation/01_synthetic_baseline/run.py --smoke --stage all
```

Full development:

```bash
python evaluation/00_synthetic_diagnostics/run.py --stage all
python evaluation/01_synthetic_baseline/run.py --stage develop
```

Review development reports and configure operating criteria (see [example](OPERATIONS.md#8-choose-and-freeze-operating-criteria)). Baseline selects pairwise/component cutoffs from distinct development scores and Leiden resolutions from bounded adaptive search. Reports cover M0/Mle1/Mle2 endpoints. Inspect before freezing settings:

```bash
python evaluation/01_synthetic_baseline/run.py --stage select
python evaluation/01_synthetic_baseline/run.py --stage evaluate
```

After baseline, run perturbation:

```bash
python evaluation/02_synthetic_perturbation/run.py --smoke  # or omit --smoke for full study
```

See the [perturbation guide](evaluation/02_synthetic_perturbation/README.md) for details. Each EpiLink scorer (EDD, EDS, ESD, ESS) has one `leiden/<scorer>` pipeline using all observed pairs with original scores as weights.

Apply baseline to Boston:

```bash
python evaluation/03_boston_application/run.py --stage all
python evaluation/03_boston_application/run.py --stage report
```

**Phylogenetic tools:** IQ-TREE ≥2.0.6 infers trees and performs LSD2 dating; TreeCluster partitions trees. Setting `phylogeny.executable: iqtree` discovers `iqtree3`, `iqtree2`, or `iqtree` on PATH. Use explicit paths to pin tools. `epilink-evaluate check` prints tool identity and experiment size.

## Layout

| Location                                | Role                                                            |
| --------------------------------------- | --------------------------------------------------------------- |
| `evaluation/shared_synthetic/`          | Shared configuration and immutable experiment artifacts         |
| `evaluation/00_synthetic_diagnostics/`  | Development diagnostics and entry point                         |
| `evaluation/01_synthetic_baseline/`     | Method comparison protocol and entry point                      |
| `evaluation/02_synthetic_perturbation/` | Sensitivity protocol and entry point                            |
| `evaluation/03_boston_application/`     | Empirical transfer and assessment                               |
| `evaluation/results/`                   | Manuscript figures and tables                                   |
| `src/epilink_evaluation/`               | Shared modules (input, scoring, clustering, metrics, reporting) |
| `data/raw/`, `data/sars-cov-2/`         | Source inputs and reference data                                |
| `tests/`                                | Integration tests                                               |

Each study has an `outputs/` directory with `current.json` pointing to `runs/<id>/`. Artifacts live under `artifacts/`. Smoke roots end in `_smoke`. Shared backbones and observations are in `evaluation/shared_synthetic/outputs/synthetic/`. Baseline stores models, scores, and trees in its artifact root. Perturbation has independent artifacts. Boston inputs are in its own `outputs/inputs/`. Reports are `report.md` and `report.html` in each run. Check stage coverage before interpreting results.

## Inputs and provenance

Git LFS manages tracked data. Derived trees and outputs are local ignored artifacts.

The shared config owns `inputs.tree_path`, generation, simulation, and splits. `epilink-evaluate scovmod --stage prepare` writes to `evaluation/shared_synthetic/outputs/inputs/`. Matching artifacts are reused; changed inputs or settings trigger rebuilding. Prebuilt trees without manifests are retained. See [tree regeneration](OPERATIONS.md#4-prepare-or-regenerate-the-scovmod-tree).

Boston's adapter reads `data/raw/boston/` and writes to `evaluation/03_boston_application/outputs/inputs/`. Its scoring table is censored at 0.0005 substitutions/site. IQ-TREE uses the full alignment and reference FASTA. See [input-processing notes](data/raw/boston/boston_data_processing.md).

Manifests record paths and identities at execution. The shared `heldout_access/seed_<seed>.json` ledger survives `reset-outputs`; revised analyses need fresh evaluation seeds.

The EpiLink model package: [https://github.com/ydnkka/epilink](https://github.com/ydnkka/epilink).

## Acknowledge

This project used AI-assisted development for code implementation, documentation, and utility commands.
