# Synthetic diagnostics

Run development-only diagnostic controls before baseline:

```bash
python evaluation/00_synthetic_diagnostics/run.py --config evaluation/00_synthetic_diagnostics/config.yaml --stage all
```

Config references `../shared_synthetic/config.yaml` for inputs, generation, simulation, splits. Diagnostics config: output location, backbone settings, graph grid. Stages: `prepare`, `backbone`, `observations`, `graphs`, `report`, `all`. `backbone`: describes tree (no observations). Other stages prepare development observations; `prepare` stops there. `report`: reads saved files.

## Analyses and scope

- **Backbone:** offspring counts (including zeros), NB dispersion, transmission concentration, Poisson-percentile superspreading, introductions/components, generation depths, descendant counts. Direct-transmission/shared-infector pair counts for M=0 target. One artifact per backbone (independent of seeds/sampling).
- **Observations:** exact GD/GD/TD cells for both processes/endpoints; target fractions, occupancy, mixing, class-conditional overlap, feature-only error, prevalence, relationship composition. Per seed: `cells.parquet`, `summary.csv`, `prevalence.csv`, `relationships.csv`.
- **Graphs:** unit-weight oracle graphs (with isolates); summaries, components, Leiden searches (`diagnostics.leiden.resolutions`: 0.1–1, 8 initial, 20 budget). Endpoints refine using own F1. Restarts selected by algorithm objective. Evaluated on all within-cluster pairs (including nonedges).

Oracle graphs keyed by truth/endpoint/sampled-case set (independent of dates/GD/TD/process/seed). Full sampling: one graph per endpoint. Stage tables repeat per seed for aggregation; `mean`/`min`/`max`/`count` descriptive.

Report: cross-seed summaries, single-seed heatmaps (labeled), precision/recall curves. Feature ambiguity empirical (not population ceiling).

Manuscript display: `python -m evaluation.results.fig02`. Outputs: `evaluation/results/outputs/00_synthetic_diagnostics/<run-id>/`. Options: `--run-dir`, `--output-dir`. See [results index](../results/README.md).

## Backbone heterogeneity and superspreading

Describe tree or generate manuscript display:

```bash
python evaluation/00_synthetic_diagnostics/run.py --stage backbone
python -m evaluation.results.fig01
```

`all` includes this stage. Standalone `backbone` ≠ baseline prerequisite. `fig01`: `--run-dir`, `--output-dir`, `--format pdf|png|both`. Exports offspring distribution, concentration, generation profile, CSV/JSON/caption. Output: `evaluation/results/outputs/00_synthetic_diagnostics/<run-id>/fig01_backbone_characterisation`.

`Z_i` = direct offspring, `R_backbone = mean(Z_i)` (all backbone cases):

```text
cutoff = Poisson(R_backbone).ppf(0.99)
superspreading = Z_i >= cutoff
```

Inclusive `>=` boundary (Lloyd-Smith et al. 2005). No transmissions: no superspreaders flagged, ratios undefined. Analysis: direct offspring (not cluster sizes/descendants).

Single-parent forest (N cases, C introductions): mean = `(N - C) / N` (descriptive, not Re). NB fit: `Var(Z) = R + R²/k`, likelihood profiling, moments fallback. Poisson-limit/degenerate: undefined `k`.

Config: `superspreading_quantile` (0.99), `bootstrap_replicates` (0), `bootstrap_seed` (67001). Resampling: percentile intervals, finite-estimate counts. Exploratory IID intervals; observation seeds ≠ independent distributions. Smoke: truncated backbone (labeled).

## Checkpoints and completion contract

`Diagnostics(config).run(stage)` returns completion status. Run: `output_directory/runs/<fingerprint>`; `current.json` points to it. `manifest.json`: `signature`, `status`, `stage`, `experiment`, `config`, `coverage_complete`. Signatures: experiment, implementations, settings, executables. Reporting changes don't invalidate checkpoints. Failed controls: failed manifests/logs; partial reports expose errors; checkpoints reuse.

Complete coverage → `<experiment.directory>/diagnostics.json`:

```json
{
  "run_directory": "/absolute/path",
  "fingerprint": "<Diagnostics.signature>",
  "experiment": {"experiment_directory": "...", "fingerprint": "..."},
  "datasets": {"<seed>": "<artifact>"},
  "status": "complete"
}
```

Baseline validation:

```python
signature = {"experiment": exp.identity, "diagnostics": marker["fingerprint"], "datasets": marker["datasets"]}
valid_artifact(Path(marker["run_directory"]) / "completion", signature)
```

`coverage.json`: `status`, `stages` (`backbone`, `observations`, `graphs`), `datasets`, `artifacts` (dir→SHA256, files). Partial coverage: no marker. Graph controls: no external executables.

`backbone/index.json`: one artifact, empty `datasets`. Evidence/manifest in completion inventory. `backbone/summary.csv`: single-backbone summary. Figure/report readers verify signatures/checksums.

## Validation

```bash
python -m pytest -q
python evaluation/00_synthetic_diagnostics/run.py --smoke --stage all
python evaluation/01_synthetic_baseline/run.py --smoke --stage all
```

Tests: percentile boundary, zero offspring, chain/star/forest truth, fits, resampling, reuse, completion, exports. Smoke: pipeline check; full study for scientific conclusions.
