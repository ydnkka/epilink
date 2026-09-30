# EpiLink evaluation

Baseline-first evaluation of EpiLink compatibility scores, genetic-distance
rankings, logistic probabilities, and graph/phylogenetic clustering.

**Start with the [operational guide](OPERATIONS.md)** for setup, configuration,
tree regeneration, stage execution, result interpretation, and troubleshooting.
The [synthetic baseline protocol](synthetic_baseline/README.md) defines the
scientific questions, primary M=0 target, comparison matrix, metrics, and
held-out operating-point evaluation.
Use the [output reference](OUTPUTS.md) for column definitions, metric formulas,
artifact provenance, and worked analysis joins.

**Validation checkpoint (2026-09-30):** 31 tests pass and the 64-case workflow
completes through held-out replay, including raw and dated TreeCluster comparisons.
See [validation and resumption notes](synthetic_baseline/VALIDATION.md) for the
report location and next full-scale development commands.

## Implementation status and study order

| Capability | Active implementation |
| --- | --- |
| SCoVMod tree preparation | Available: `epilink-evaluate prepare-tree`. Existing trees are retained. |
| Synthetic baseline | Available: pairwise comparisons, clustering sweeps, operating-point selection, held-out replay, and reports. |
| Parameter sensitivity | Planned; no active sensitivity command. Historical code is archived. |
| Boston input preparation | Available: `epilink-evaluate prepare-boston`. |
| Empirical scoring/evaluation | Planned in the rebuilt workflow. |

The study sequence is synthetic baseline → parameter perturbations with baseline
operating rules → empirical application. See the guide for
[sensitivity and Boston support](OPERATIONS.md#11-sensitivity-analysis-and-boston-inputs).

## Install and run

Python ≥3.10, from the repository root:

```bash
python -m pip install -e '.[test]'
epilink-evaluate check --config synthetic_baseline/config.yaml
python -m synthetic_baseline.run --smoke --stage all
python -m pytest

# Full development on the configured backbone:
python -m synthetic_baseline.run --stage develop
```

Review the development report and configure operating criteria before selection
and evaluation; see the [worked criteria example](OPERATIONS.md#8-choose-and-freeze-operating-criteria).

```bash
python -m synthetic_baseline.run --stage select
python -m synthetic_baseline.run --stage evaluate
```

FastME is an external executable (e.g. `conda install -c bioconda fastme`).
TreeCluster and TreeTime are Python dependencies. Executables are discovered on
PATH or alongside the active Python interpreter; override their paths in the
configuration when needed. `check` prints tool identity and experiment size.

The existing local environment is
`/opt/homebrew/Caskroom/miniconda/base/envs/epilik_evaluation/bin/python`.
Use that interpreter consistently for installation and execution if working in
this checkout. No machine-specific path is embedded in the implementation.
Fresh-environment and Git LFS instructions are in the
[setup section](OPERATIONS.md#2-environment-and-source-inputs).

## Layout

| Location                                           | Role                                                                        |
| -------------------------------------------------- | --------------------------------------------------------------------------- |
| `synthetic_baseline/`                              | Scientific protocol, configuration, thin entry point                        |
| `src/epilink_evaluation/`                          | Shared input, scoring, clustering, metrics, selection and reporting modules |
| `data/raw/`, `data/processed/`, `data/sars-cov-2/` | Preserved source inputs and reference data                                  |
| `data/derived/`                                    | Regenerable input artifacts, including the fixed transmission backbone      |
| `tests/`                                           | Independent scientific correctness and integration checks                   |
| `tools/`                                           | Auditable one-time migration utilities                                      |
| `archive/pre_reset_2026-09-30/`                    | Verified legacy workflows, results, notebooks and local assets              |

Outputs are namespaced by experiment fingerprint under
`synthetic_baseline/outputs/baseline/runs/`; `current.json` identifies the latest
run. Content-addressed artifacts are shared under `artifacts/`. Smoke validation
uses a separate `baseline_smoke/` root. Reports are `report.md` and `report.html`
inside each run. A changed scientific configuration or implementation creates a
new run namespace, preserving earlier results.

## Inputs and preservation

The [archive manifest](archive/pre_reset_2026-09-30/manifest.json) records original
paths, checksums, tracking status and Git revision. Raw data remain in place;
Git LFS continues to manage tracked data formats. Local ignored archive assets
must accompany the checkout when it is moved.

The preserved 4,990-case reference backbone was promoted with
`python tools/promote_legacy_tree.py`. The active experiment uses the tree at
`inputs.tree_path`, which may be a regenerated backbone with a different size.
`epilink-evaluate prepare-tree` constructs a missing tree from the SCoVMod raw
inputs and retains an existing tree. Its hash defines the experiment. Follow
the [tree regeneration instructions](OPERATIONS.md#4-prepare-or-regenerate-the-scovmod-tree)
to change the target component and check the actual case count.

Boston source files and `data/boston_data_processing.py` are preserved. The new
`epilink-evaluate prepare-boston` adapter produces derived tables and provenance
in the configured output root without replacing the preserved processed inputs.
The historical TN93 pair table is distance-censored; the empirical protocol must
address candidate-universe coverage before treating it as all pairwise data.

The EpiLink model package is maintained separately at
<https://github.com/ydnkka/epilink>.
