# EpiLink evaluation

Evaluation of EpiLink compatibility scores, genetic-distance
rankings, logistic probabilities, and graph/phylogenetic clustering.

**Start with the [operational guide](OPERATIONS.md)** for setup, configuration,
tree regeneration, stage execution, result interpretation, and troubleshooting.
The [synthetic baseline protocol](evaluation/01_synthetic_baseline/README.md) defines the
scientific questions, primary M=0 target, comparison matrix, metrics, and
held-out operating-point evaluation.
Use the [output reference](OUTPUTS.md) for column definitions, metric formulas,
artifact provenance, and worked analysis joins.

**Recorded validation (2026-09-30):** baseline and frozen-setting perturbation
smoke workflows completed, including raw and dated TreeCluster comparisons.
The [baseline checkpoint](evaluation/01_synthetic_baseline/VALIDATION.md) and
[perturbation checkpoint](evaluation/02_synthetic_perturbation/README.md#validation-checkpoint--2026-09-30)
record the tests and artifacts from those executions.

## Three evaluation studies

| Study                                                                         | Purpose                                                                                                           | Main evidence                                                              |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| [01 — Synthetic baseline](evaluation/01_synthetic_baseline/README.md)         | Compare methods against known relationships, select operating points, and evaluate them on held-out observations. | Pairwise and clustering accuracy, development sweeps, and frozen settings. |
| [02 — Synthetic perturbation](evaluation/02_synthetic_perturbation/README.md) | Test biological-parameter sensitivity and EpiLink inference mismatch using frozen operating points.               | Paired performance differences from fresh unperturbed controls.            |
| [03 — Boston application](evaluation/03_boston_application/README.md)         | Examine empirical transfer, exposure concentration/recovery, and sensitivity to clustering settings.              | Descriptive exposure summaries and graph/phylogenetic partition agreement. |

The completed baseline supplies the reference for both downstream studies.
Perturbation and Boston can run independently after baseline evaluation; the
directory numbers express the study presentation order.

## Implementation status

| Capability                    | Active implementation                                                                                           |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------- |
| SCoVMod tree preparation      | Available: `epilink-evaluate scovmod --stage prepare`. Matching prepared inputs are reused.                     |
| Synthetic baseline            | Available: pairwise comparisons, clustering sweeps, operating-point selection, held-out replay, and reports.    |
| Parameter sensitivity         | Available: `epilink-evaluate perturbation`, with paired matched/baseline-fixed scenarios and frozen settings.   |
| Boston input preparation      | Available: `epilink-evaluate boston --stage prepare`.                                                           |
| Boston frozen transfer        | Available: `epilink-evaluate boston --stage all`, including graph clustering and enabled raw/dated TreeCluster. |
| Boston clustering exploration | Available: `epilink-evaluate boston --stage explore`, with separate descriptive grid outputs.                   |
| Output cleanup                | Available: `epilink-evaluate reset-outputs`, with selective clearing by evaluation and dry-run preview.         |

See the guide for [perturbation and Boston execution](OPERATIONS.md#12-perturbation-and-boston-application) and [clearing outputs](OPERATIONS.md#11-clear-outputs-with-reset-outputs).

## Install and run

Python ≥3.10, from the repository root:

```bash
python -m pip install -e '.[test]'
epilink-evaluate check --config evaluation/01_synthetic_baseline/config.yaml
python evaluation/01_synthetic_baseline/run.py --smoke --stage all
python -m pytest

# Full development on the configured backbone:
python evaluation/01_synthetic_baseline/run.py --stage develop
```

Review the development report and configure operating criteria before selection
and evaluation; see the [worked criteria example](OPERATIONS.md#8-choose-and-freeze-operating-criteria).

```bash
python evaluation/01_synthetic_baseline/run.py --stage select
python evaluation/01_synthetic_baseline/run.py --stage evaluate
```

After completing baseline evaluation, run the perturbation smoke study:

```bash
python evaluation/02_synthetic_perturbation/run.py --smoke
# Full perturbation study:
python evaluation/02_synthetic_perturbation/run.py
```

See the [perturbation guide](evaluation/02_synthetic_perturbation/README.md) for reference-run
selection, parameter levels, and paired-result interpretation.

Apply the baseline reference to Boston, then run the optional exploratory grid:

```bash
python evaluation/03_boston_application/run.py --stage all
python evaluation/03_boston_application/run.py --stage explore
python evaluation/03_boston_application/run.py --stage report
```

Boston's `all` stage runs frozen transfer; exploration is a separate stage.

FastME is an external executable (e.g. `conda install -c bioconda fastme`).
TreeCluster and TreeTime are Python dependencies. Executables are discovered on
PATH or alongside the active Python interpreter; override their paths in the
configuration when needed. `check` prints tool identity and experiment size.
Boston tree construction additionally requires the standalone `tn93` executable;
see the [Boston guide](evaluation/03_boston_application/README.md#troubleshooting).

The existing local environment is
`/opt/homebrew/Caskroom/miniconda/base/envs/epilik_evaluation/bin/python`.
Use that interpreter consistently for installation and execution if working in
this checkout. No machine-specific path is embedded in the implementation.
Fresh-environment and Git LFS instructions are in the
[setup section](OPERATIONS.md#2-environment-and-source-inputs).

## Layout

| Location                                           | Role                                                                        |
| -------------------------------------------------- | --------------------------------------------------------------------------- |
| `evaluation/01_synthetic_baseline/`                | Method-comparison protocol, configuration, and entry point                  |
| `evaluation/02_synthetic_perturbation/`            | Frozen-reference sensitivity protocol, configuration, and entry point       |
| `evaluation/03_boston_application/`                | Empirical transfer and clustering exploration                               |
| `src/epilink_evaluation/`                          | Shared input, scoring, clustering, metrics, selection and reporting modules |
| `data/raw/`, `data/processed/`, `data/sars-cov-2/` | Preserved source inputs and reference data                                  |
| `evaluation/01_synthetic_baseline/outputs/inputs/` | Prepared transmission backbone and provenance                               |
| `evaluation/03_boston_application/outputs/inputs/` | Prepared Boston tables and provenance                                       |
| `tests/`                                           | Independent scientific correctness and integration checks                   |

Each study has its own `outputs/` directory containing an output root (`baseline`,
`perturbation`, or `boston`). Within that root, `current.json` locates the latest
initialized `runs/<id>/`, and content-addressed artifacts live under `artifacts/`.
Baseline and perturbation smoke validation use separate roots ending in `_smoke`.
Baseline and Boston prepared inputs are shared alongside their run roots, in
each study's `outputs/inputs/`. Perturbation reconstructs its frozen backbone
from baseline truth and stores it under its own `artifacts/backbones/`.
Reports are `report.md` and `report.html` inside each run. The pointer can identify
an incomplete run; check stage coverage before interpreting results.

## Inputs and provenance

Git LFS manages tracked data formats. Derived trees and study outputs are local
ignored artifacts; retain them when moving an experiment, or regenerate them.

The active baseline experiment uses the tree at `inputs.tree_path`.
`epilink-evaluate scovmod --stage prepare` uses the same preparation
as the baseline workflow, writing to `evaluation/01_synthetic_baseline/outputs/inputs/`.
Matching managed artifacts are reused; changed inputs or construction settings
trigger rebuilding. Explicit prebuilt trees without a manifest are retained.
The tree hash defines the experiment. Follow
the [tree regeneration instructions](OPERATIONS.md#4-prepare-or-regenerate-the-scovmod-tree)
to change the target component and check the actual case count.

Boston's input adapter reads `data/raw/boston/` and writes derived tables and
provenance to `evaluation/03_boston_application/outputs/inputs/`.
Use `epilink-evaluate boston --stage prepare` or the Boston `run.py --stage prepare` entry point.
Its scoring table is censored at 0.0005
substitutions/site; missing pairs remain unobserved. Tree construction separately
computes all-pair TN93 distances from the alignment. See the
[input-processing notes](data/raw/boston/boston_data_processing.md).

Saved manifests record paths at execution time. After relocating study
directories, use the updated `current.json` pointers to locate results; recorded
scientific signatures remain tied to their original execution.

READMEs under `archive/pre_reset_2026-09-30/` document the archived workflow.
The dated validation checkpoints record historical executions; their generated
report links require the corresponding local output artifacts.

The EpiLink model package is maintained separately at
<https://github.com/ydnkka/epilink>.

## Acknowledge

This project utilised AI-assisted development. AI tools were used to help with code implementation, documentation improvements, and the creation of utility commands.
