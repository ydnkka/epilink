# Synthetic diagnostics

Run the shared synthetic experiment's development-only diagnostic controls before baseline development:

```bash
python evaluation/00_synthetic_diagnostics/run.py --config evaluation/00_synthetic_diagnostics/config.yaml --stage all
```

The configuration references `../shared_synthetic/config.yaml`. Shared inputs, generation, simulation, and splits come from that experiment; the diagnostics file contains only its output location and control grids. Stages are `prepare`, `observations`, `graphs`, `trees`, `report`, and `all`. Computational stages prepare development observations as dependencies. `prepare` stops at that preparation. `report` reads saved run files and can be rerun without generating observations.

## Analyses and scope

- **Observations:** exact GD and GD/TD feature cells for both distance processes and every endpoint; target fractions, occupancy, mixing, class-conditional overlap, empirical feature-only classification error, endpoint prevalence, and relationship composition. Each seed has `cells.parquet`, `summary.csv`, `prevalence.csv`, and `relationships.csv` in a content-addressed artifact.
- **Graphs:** unit-weight endpoint-oracle graphs including isolates; graph summaries, connected components, and the configured CPM Leiden resolution grid. Restarts are selected by objective value. Partitions are evaluated on all within-cluster pairs, including graph nonedges, using `PartitionEvaluator` and reference memberships reconstructed from the full transmission tree.
- **Trees:** sampled-tip known transmission topology with unit transmission-edge lengths, zero-length sampled tips, and unsampled intermediates preserved. The existing TreeCluster adapter runs each method/threshold once; its partition is evaluated at all endpoints. Forests are explicitly rejected by this control.

Oracle graph/tree caches are keyed by truth and canonical sampled-case set, independently of observation dates, GD/TD, process, and seed. Full sampling thus reuses one control across realizations. Memberships, cluster tables, metrics, algorithm details, and source provenance are retained. Stage tables repeat the shared result once per seed for equal-seed aggregation; `mean`, `min`, `max`, and defined-value `count` are descriptive, with `n_seeds` reported separately. No confidence interval treats dependent pairs as independent observations.

The report gives cross-seed summaries, explicitly labelled single-seed GD/TD target-fraction and occupancy heatmaps, graph partition precision/recall curves, and tree threshold curves. Exact-feature ambiguity is empirical, not a population performance ceiling. Oracle graphs are not an absolute partition-performance ceiling. The hop tree is not a molecular genealogy, and relationship horizon M differs from total transmission hops.

## Checkpoints and completion contract

`Diagnostics(config).run(stage)` returns whether the requested stage completed. The run is `output_directory/runs/<full signature fingerprint>`; `current.json` points to it. `manifest.json` records `signature`, `status`, `requested_stage`, `experiment`, `config`, and `coverage_complete`. Computational signatures contain the experiment identity, scoped producer/core algorithm implementations, settings, and relevant executable identity. Reporting source changes do not invalidate computational checkpoints. Failed controls retain failed manifests and logs; partial reports expose errors and successful checkpoints are reused on retry.

After complete coverage of observations, graphs, and trees when enabled, `<experiment.directory>/diagnostics.json` has this schema:

```json
{
  "run_directory": "/absolute/diagnostics/runs/<fingerprint>",
  "fingerprint": "<fingerprint(Diagnostics.signature)>",
  "experiment": {
    "experiment_directory": "/absolute/shared/experiment",
    "fingerprint": "<full experiment hash>"
  },
  "datasets": {"<development seed>": "<observation directory name>"},
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

That artifact is made using `complete_artifact(completion, signature, ["coverage.json"])`. `coverage.json` records `status`, required `stages`, `datasets`, aggregation scope, and `artifacts`: a mapping from absolute stage or control directory to its `manifest_sha256` and file-name-to-SHA256 `files` map. It includes the table references and all computational evidence. Partial coverage never releases a completion marker. Tree controls can be explicitly disabled for portable nonexternal integration checks; the report labels that omission.

## Validation

Validated on 2026-10-02 using the project's configured Python environment:

```bash
python -m pytest -q
python evaluation/00_synthetic_diagnostics/run.py --smoke --stage all
python evaluation/01_synthetic_baseline/run.py --smoke --stage all
```

All **160 tests passed**. The real-tool 64-case sequence completed feature-cell diagnostics, 9 oracle-graph partitions, and 12 transmission-hop tree partitions, then baseline development (1,541 exact pairwise candidates and 132 clustering settings) and frozen replay (23 pairwise and 47 clustering settings). TreeCluster, FastME, and TreeTime were exercised. Baseline reused the exact diagnostic development observations and linked the diagnostic report before its performance results. Endpoint-aware summaries/frontiers and development grid-adequacy diagnostics were verified for all three criteria. This validates execution; scientific conclusions require the full configured study.

The full configured diagnostics run on 2026-10-02 used 5,051 cases, 5,000-nt sequences, and development observation seeds 62001–62003. Its first timestamped simulation log was at 04:08:47 and its report file was written at 04:13:58, giving an observed log-to-report interval of about **5m 10s**. The command's exact start time was not logged, so this is an approximate runtime rather than a precise process duration. Hardware context is recorded in the [Operations runtime benchmark](../../OPERATIONS.md#observed-full-run-wall-times).
