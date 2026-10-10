# Synthetic diagnostics

Run the shared synthetic experiment's development-only diagnostic controls before baseline development:

```bash
python evaluation/00_synthetic_diagnostics/run.py --config evaluation/00_synthetic_diagnostics/config.yaml --stage all
```

The configuration references `../shared_synthetic/config.yaml`. Shared inputs, generation, simulation, and splits come from that experiment; the diagnostics file contains its output location, backbone-summary settings and graph control grid. Stages are `prepare`, `backbone`, `observations`, `graphs`, `report`, and `all`. `backbone` describes the pinned transmission tree without generating observations. Other computational stages prepare development observations as dependencies; `prepare` stops at that preparation. `report` reads saved run files and can be rerun without generating observations.

## Analyses and scope

- **Backbone:** offspring counts for every case, including zeros; negative-binomial dispersion, transmission concentration, the inclusive Poisson-percentile superspreading rule, introductions/components, generation depths and descendant counts. Direct-transmission and shared-infector pair counts explain how branching contributes to the primary M=0 target. This analysis produces one artefact per backbone, independently of observation seeds and sampling fraction.
- **Observations:** exact GD and GD/TD feature cells for both distance processes and every endpoint; target fractions, occupancy, mixing, class-conditional overlap, empirical feature-only classification error, endpoint prevalence, and relationship composition. Each seed has `cells.parquet`, `summary.csv`, `prevalence.csv`, and `relationships.csv` in a content-addressed artifact.
- **Graphs:** unit-weight endpoint-oracle graphs including isolates; graph summaries, connected components, and bounded CPM Leiden searches configured by `diagnostics.leiden.resolutions`. The supplied bounds are 0.1–1 with 8 initial linear-space points and a 20-resolution budget. Each oracle endpoint refines using its own endpoint F1. Restarts are selected by the clusterer's algorithm objective. Partitions are evaluated on all within-cluster pairs, including graph nonedges, using `PartitionEvaluator` and the shared pairwise transmission truth. Explicit resolution lists remain supported for fixed diagnostic comparisons.

Oracle graph caches are keyed by truth, endpoint, and canonical sampled-case set, independently of observation dates, GD/TD, process, and seed. Full sampling thus reuses one graph per endpoint across realizations. Memberships, cluster tables, metrics, algorithm details, and source provenance are retained. Stage tables repeat the shared result once per seed for equal-seed aggregation; `mean`, `min`, `max`, and defined-value `count` are descriptive, with `n_seeds` reported separately. No confidence interval treats dependent pairs as independent observations.

The report gives cross-seed summaries, explicitly labelled single-seed GD/TD target-fraction and occupancy heatmaps, and graph partition precision/recall curves. Exact-feature ambiguity is empirical, not a population performance ceiling. Oracle graphs are not an absolute partition-performance ceiling.

The standalone feature/graph manuscript display is [`evaluation/results/fig02.py`](../results/fig02.py); from the repository root run `python -m evaluation.results.fig02` after completing diagnostics. It writes `fig02_diagnostics_figure.pdf` and `.png` under `evaluation/results/outputs/00_synthetic_diagnostics/<run-id>/`. Use `--run-dir` to pin another run or `--output-dir` to choose another destination. See the [numbered results index](../results/README.md) for all displays.

## Backbone heterogeneity and superspreading

To describe only the selected tree, or to produce its manuscript display:

```bash
python evaluation/00_synthetic_diagnostics/run.py --stage backbone
python -m evaluation.results.fig01
```

`all` includes this stage before the observation and oracle controls. A successful standalone `backbone` stage alone does not complete the baseline prerequisite. `fig01` accepts `--run-dir`, `--output-dir` and `--format pdf|png|both`, and exports the offspring distribution, cumulative transmission concentration and generation profile, together with CSV evidence, a JSON summary and a caption. Results are written under `evaluation/results/outputs/00_synthetic_diagnostics/<run-id>/` with prefix `fig01_backbone_characterisation`.

Let `Z_i` be a case's direct offspring count and `R_backbone = mean(Z_i)` over **all** backbone cases. The default rule is:

```text
cutoff = Poisson(R_backbone).ppf(0.99)
superspreading case = Z_i >= cutoff
```

This uses the homogeneous Poisson percentile reference discussed by Lloyd-Smith et al. (2005), with an inclusive `>=` boundary. The saved `poisson_percentile` and `minimum_superspreading_offspring` make it explicit. If there are no transmission edges, no case is flagged and transmission-share ratios are undefined. The analysis concerns attributable direct offspring, rather than inferred cluster sizes or all subsequent descendants.

For a single-parent forest with N cases and C introductions, the mean is `(N - C) / N`; it is therefore a descriptive backbone reference, rather than an estimate of the epidemic's effective reproduction number. The negative-binomial fit uses `Var(Z) = R + R²/k`, profiles the likelihood at the empirical mean, and retains a method-of-moments fallback. Poisson-limit and degenerate fits have undefined finite `k` and an explicit fitting status. This analysis describes the exact **selected evaluation backbone**.

`diagnostics.backbone` configures `superspreading_quantile` (default 0.99), `bootstrap_replicates` (default 0) and `bootstrap_seed` (default 67001). Optional resampling retains every replicate and reports percentile intervals with finite-estimate counts. These are exploratory IID case-resampling intervals; the observation seeds do not provide independent offspring distributions. Smoke summarises the truncated backbone and is labelled accordingly.

## Checkpoints and completion contract

`Diagnostics(config).run(stage)` returns whether the requested stage completed. The run is `output_directory/runs/<full signature fingerprint>`; `current.json` points to it. `manifest.json` records `signature`, `status`, `requested_stage`, `experiment`, `config`, and `coverage_complete`. Computational signatures contain the experiment identity, scoped producer/core algorithm implementations, settings, and relevant executable identity. Reporting source changes do not invalidate computational checkpoints. Failed controls retain failed manifests and logs; partial reports expose errors and successful checkpoints are reused on retry.

After complete coverage of backbone, observations, and graphs, `<experiment.directory>/diagnostics.json` has this schema:

```json
{
  "run_directory": "/absolute/diagnostics/runs/<fingerprint>",
  "fingerprint": "<fingerprint(Diagnostics.signature)>",
  "experiment": {
    "experiment_directory": "/absolute/shared/experiment",
    "fingerprint": "<full experiment hash>"
  },
  "datasets": { "<development seed>": "<observation directory name>" },
  "status": "complete"
}
```

Baseline can validate the `completion` artifact with:

```python
signature = {
    "experiment": experiment.identity,
    "diagnostics": marker["fingerprint"],
    "datasets": marker["datasets"],
}
valid_artifact(Path(marker["run_directory"]) / "completion", signature)
```

That artifact is made using `complete_artifact(completion, signature, ["coverage.json"])`. `coverage.json` records `status`, required `stages` (`backbone`, `observations`, `graphs`), `datasets`, aggregation scope, and `artifacts`: a mapping from absolute stage or control directory to its `manifest_sha256` and file-name-to-SHA256 `files` map. It includes the table references and all computational evidence. Partial coverage never releases a completion marker. Diagnostic graph controls run without external phylogenetic executables.

`backbone/index.json` contains exactly one artefact reference and an empty `datasets` map, rather than repeated seed records. Its evidence and stage manifest are included in the completion inventory. `backbone/summary.csv` is a single-backbone summary; detailed nodes, distributions and resampling records live in the referenced artefact. The figure/report readers check signatures and checksums and never refit the saved distribution.

## Validation

Run the scientific and workflow checks from the repository root:

```bash
python -m pytest -q
python evaluation/00_synthetic_diagnostics/run.py --smoke --stage all
python evaluation/01_synthetic_baseline/run.py --smoke --stage all
```

Tests cover the inclusive percentile boundary, zero offspring, chain/star/forest truth, finite and Poisson-limit fits, resampling reproducibility, seed/sampling-independent reuse, completion revocation/repair and manuscript exports. Smoke checks pipeline execution; scientific conclusions require the full configured study.
