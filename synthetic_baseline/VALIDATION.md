# Validation checkpoint — 2026-09-30

## Completed

- Installed the editable package and test dependencies in the documented
  `epilik_evaluation` environment.
- The installed CLI's configuration/dependency check passes: FastME, TreeTime,
  TreeCluster, and the canonical transmission backbone are available.
- **31 tests pass** with `python -m pytest -q`.
- **The 64-case smoke workflow completes** with
  `python -m synthetic_baseline.run --smoke --stage all`.

The tests cover full-tree AD/CA truth against independent graph paths,
unsampled infectors, separate introductions, tie-aware AP against scikit-learn,
all within-cluster pairs, extended BCubed against the `bcubed` package,
count-compressed logistic fitting against uncompressed fitting, threshold and
weight conventions, Leiden restarts, development-only selection, held-out replay,
cache corruption/reuse, and phylogenetic adapters. The TreeCluster executable
test skips explicitly if that tool is unavailable; it passed in this environment.

## Integration fix

The first smoke run completed graph and raw-tree comparisons but failed dated
comparisons. Biopython's Nexus reader placed TreeTime's internal node names in
the numeric confidence field when date comments were present. The adapter now
restores those names and normalizes comment brackets before Newick export,
preserving branch lengths in calendar years. A regression test checks the named
nodes, dates, and pairwise tree distances. TreeTime also now receives an explicit
random seed, recorded in its artifact signature.

## Saved smoke evidence

Successful run: `bf25d2f5f9615762cd6a`.

- [HTML report](outputs/baseline_smoke/runs/bf25d2f5f9615762cd6a/report.html)
- [Markdown report](outputs/baseline_smoke/runs/bf25d2f5f9615762cd6a/report.md)
- Run root: `outputs/baseline_smoke/runs/bf25d2f5f9615762cd6a/`
- Latest-run pointer: `outputs/baseline_smoke/current.json`

| Stage | Coverage |
| --- | --- |
| Training | Seed 71001, both logistic models |
| Development | Seed 72001, all eight scorers, 24 pairwise threshold settings |
| Development clustering | 132/132 settings complete, zero errors |
| Frozen selection | 34 operating points under the smoke `balanced_M0` criterion |
| Held-out replay | Seed 73001, eight pairwise and 26 clustering settings complete |
| Reporting | Markdown, HTML, and five development figures generated |

The partial first run remains under `e95476354ea66c432385` for diagnosis. Output
directories are ignored by Git; these report links refer to local generated
artifacts. Smoke results establish pipeline functionality. Full-baseline
scientific conclusions require the 4,990-case development and evaluation runs.

## Resume full-scale development

From the repository root, using the same environment:

```bash
conda activate epilik_evaluation
python -m synthetic_baseline.run --stage develop
```

The full configuration uses training seeds 61001/61002, development seeds
62001/62002/62003, and 2,044 operating definitions per development realization.
Artifacts and reports are written under `synthetic_baseline/outputs/baseline/`;
`current.json` identifies the run. Repeating a stage reuses validated artifacts
and resumes missing work.

Review the development precision–recall curves, clustering sweeps, comparator
completeness, and grid boundaries. Set the intended operating criteria in
`synthetic_baseline/config.yaml`, then freeze and replay:

```bash
python -m synthetic_baseline.run --stage select
python -m synthetic_baseline.run --stage evaluate
```

Full evaluation seeds 63001/63002/63003 have not been used at this checkpoint.
Parameter perturbations and the empirical application follow the completed
baseline study, as described in the protocol.
