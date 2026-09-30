# EpiLink evaluation

Baseline-first evaluation of EpiLink compatibility scores, genetic-distance
rankings, logistic probabilities, and graph/phylogenetic clustering.

**Start with the [synthetic baseline protocol](synthetic_baseline/README.md).**
It defines the scientific questions, primary M=0 target, comparison matrix,
metrics, development sweeps, and held-out operating-point evaluation.

## Study order

1. **Synthetic baseline:** pairwise comparisons → clustering sweeps → operating
   criteria → held-out evaluation on new observation realizations.
2. **Parameter perturbations:** carry baseline operating rules forward and
   measure matched and baseline-fixed inference sensitivity.
3. **Empirical application:** reuse the same scoring/clustering definitions and
   assess available epidemiological evidence.

## Install and run

Python ≥3.10, from the repository root:

```bash
python -m pip install -e '.[test]'
epilink-evaluate check --config synthetic_baseline/config.yaml
python -m synthetic_baseline.run --smoke --stage all
python -m pytest

# Full-scale development, then explicit operating-point selection/evaluation:
python -m synthetic_baseline.run --stage develop
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

The canonical backbone was promoted with `python tools/promote_legacy_tree.py`.
If no tree exists, `epilink-evaluate prepare-tree` regenerates a backbone from
the SCoVMod raw inputs with explicit ordering and provenance. The selected tree's
hash defines the experiment; historical reconstruction details are archived.

Boston source files and `data/boston_data_processing.py` are preserved. The new
`epilink-evaluate prepare-boston` adapter produces derived tables and provenance
in the configured output root without replacing the preserved processed inputs.
The historical TN93 pair table is distance-censored; the empirical protocol must
address candidate-universe coverage before treating it as all pairwise data.

The EpiLink model package is maintained separately at
<https://github.com/ydnkka/epilink>.
