# Synthetic EpiLink exploration

A separate, reproducible analysis of what the synthetic observations, pairwise compatibility scores, and clusters represent. The existing evaluation pipeline and historical results are not modified.

The first run is the **matched baseline, seed 12345**, with all four EpiLink combinations. A second realization, seed 54321, supplies training observations for the supervised comparisons. Both use the same transmission tree; this is not validation across independently generated epidemics.

## Start here

- [Baseline report](outputs/runs/matched/baseline/seed_12345/report.md)
- [Browser version](outputs/runs/matched/baseline/seed_12345/report.html)
- [Baseline ESS table preview](outputs/runs/matched/baseline/seed_12345/table_preview/ESS.csv)
- [Analysis table schema](outputs/analysis_table_schema.json)
- [Experiment configuration](config.yaml)

From the repository root:

```bash
python3 -m synthetic_exploration.run
python3 -m unittest discover -s synthetic_exploration/tests -v
```

An optional writable Matplotlib cache can be selected with `MPLCONFIGDIR`. The existing repository requirements provide the dependencies. To generate only the reusable tree/observation data:

```bash
python3 -m synthetic_exploration.run --stage prepare
```

## One analysis table

The user-facing dataset is **`outputs/analysis_table/`**. Read that directory as one Parquet table. Files are partitioned by condition, scenario, seed, and model so filters can avoid reading unrelated experiments. This is a single logical table, not a collection of tables that the user must join.

| Column | Meaning |
|---|---|
| `CaseID1`, `CaseID2` | Case identifiers in stable tree input order; an unordered pair |
| `AD` | 1 for ancestor–descendant relationships; otherwise 0 |
| `CA` | 1 when neither case is ancestral to the other and they share an ancestor |
| `m` | Number of intermediates between an AD pair |
| `m1` | CA intermediates from the most recent common ancestor to `CaseID1` |
| `m2` | CA intermediates from the most recent common ancestor to `CaseID2` |
| `M` | Total intermediates: `m` for AD, `m1 + m2` for CA |
| `GD` | Observed genetic distance for this row's data process |
| `TD` | Absolute sample-date difference, rounded to days, matching the existing pipeline |
| `CS` | Original summed target compatibility from this row's inference model |
| `model` | EDD, EDS, ESD, or ESS |
| `condition`, `scenario`, `seed` | Experiment identifiers |
| `data_process`, `inference_process` | Explicit deterministic/stochastic genetics axes |
| `pair_id` | Stable unordered-pair identifier within this tree |
| `IsRelated` | The original AD(0) or CA(0,0) target label |
| `relationship`, `tree_hops` | Convenient broader category and full-tree edge distance |
| `lca_index`, `lca_steps_a`, `lca_steps_b` | Common-ancestor index and edge counts to each case |

The unique row key is `(condition, scenario, seed, model, pair_id)`. Every pair appears once per model. The four baseline models therefore produce 49,790,220 rows (12,447,555 per model). This is deliberate: a single `CS` column must not silently mix inference settings. Load one model or selected columns at a time.

```python
from synthetic_exploration.table import read_analysis_table

df = read_analysis_table(
    model="ESS",
    columns=["CaseID1", "CaseID2", "AD", "CA", "m", "m1", "m2", "M", "GD", "TD", "CS"],
)
```

For streaming queries or all experiments:

```python
import pyarrow.dataset as ds

table = ds.dataset("synthetic_exploration/outputs/analysis_table", format="parquet", partitioning="hive")
scanner = table.scanner(
    columns=["CaseID1", "CaseID2", "AD", "CA", "m", "m1", "m2", "M", "GD", "TD", "CS"],
    filter=(ds.field("model") == "ESS") & (ds.field("scenario") == "baseline"),
)
for batch in scanner.to_batches():
    # Analyze one batch at a time.
    pass
```

Inactive depths are **null**, not zero. Direct transmission is `AD=1, m=0`; shared infector is `CA=1, m1=0, m2=0`. In an AD pair, the ancestor can be either case: case order does not assert transmission direction. In the pair table, CA branch order follows the two case columns. **CA(a,b) and CA(b,a) are the same relationship**: for example, CA(0,1) and CA(1,0) are combined. Relationship count summaries and score atoms use the canonical order `m1 <= m2`, meaning the shorter and longer branch rather than the first and second case. `canonical_encoding(pairs)` in `truth.py` supplies these grouping keys without modifying the original pair table. `encoding_examples.csv`, table previews, and the case-aligned `lca_steps_a/b` geometry table retain their original orientation for tracing individual branches. Across different introductions, AD and CA would both be zero and all depths null; no such pairs exist in this single-tree baseline. Self-pairs are excluded.

`M` expresses separation in total intermediates, combining the active class: `M=m` for AD and `M=m1+m2` for CA. Consequently, both `AD(0)` and `CA(0,0)` have `M=0`, while `AD(2)` and `CA(1,1)` both have `M=2`. The original branch counts retain the difference in geometry. M excludes the shared ancestor for CA, so the edge distance is `M+1` for AD and `M+2` for CA. `tree_hops` retains that edge distance. M is null for pairs from separate trees.

`M_prevalence.csv`, `score_by_M.csv`, `selection_M.csv`, and `within_cluster_M.csv` provide the background distribution, score summaries, selected-pair composition, and within-cluster composition by M and AD/CA class. The main report also shows M distributions and median/90th-percentile M.

### Distribution of M against score

[Pooled heatmap](outputs/runs/matched/baseline/seed_12345/figures/02_M_against_score.png) shows **P(M | compatibility-score band)** for all four models; separate [AD](outputs/runs/matched/baseline/seed_12345/figures/02_M_against_score_AD.png) and [CA](outputs/runs/matched/baseline/seed_12345/figures/02_M_against_score_CA.png) views use the same axes and colour scale. Each occupied column is normalized by its number of pairs. The zero-score mass has a separate column; positive bands are `(lower, upper]` with default width 0.05. Grey columns have no pairs. Solid lines show median M; dashed lines show the 10th/90th percentiles. These are descriptive distribution quantiles, not confidence intervals.

`02_scores/M_given_score_distribution.csv` contains exact pair counts and conditional fractions; `M_given_score_summary.csv` includes counts and quantiles for every band, including empty bands. PNG and SVG figures are generated from cached exact counts without resampling or rerunning the simulation:

```bash
python3 -m synthetic_exploration.score_distribution
```

Use `--run <run-directory>` for another experiment. An optional `figures.M_score_band_width` setting controls positive-score bin width.

The internal `outputs/trees/<fingerprint>/relationships.parquet` contains the seed-independent encoding once for the full tree. `encoding_examples.csv` and `encoding_counts.csv` make it inspectable. Observation realizations are cached separately, and the exported analysis table combines truth, observations, and scores. Unsampled intermediates remain in the tree.

## Four investigations

1. **`01_observations/`** — relationship and AD/CA depth prevalence; exact joint genetic/temporal feature cells; collisions between target and other pairs; empirical distribution overlap. A mixed cell establishes ambiguity for those observed inputs, not a population-wide performance ceiling.
2. **`02_scores/`** — AP and full precision–recall curves; score distributions by relationship; score-band composition; fixed score cutoffs, candidate budgets, and recall operating points; enrichment, relationship recall, transmission distances, and exact EpiLink encoding counts. All score ties are retained, including ties that exceed a requested candidate budget.
3. **`03_clusters/`** — fixed resolution sweep; memberships; every within-cluster pair, not only graph edges; direct-edge retention and fragmentation; pair-weighted and equal non-singleton-cluster-weighted composition; the true-target-edge graph diagnostic; size-preserving and size-plus-time conditional randomizations at the fixed primary resolution.
4. **`04_validation/`** — reproduction of historical baseline scores and labels; actual target observations versus the scorer's cached joint draws; marginal Wasserstein distances and binned joint total variation; genetic-only, time-only, separately trained logistic, and smoothed feature-cell lookup comparisons. The raw scorer is unchanged. Absolute-time transformations are explicitly diagnostic, and the report distinguishes them from production scoring. The report also records the limits and confirmation work remaining.

The existing overlapping-neighbourhood BCubed measure remains secondary. No resolution is selected by maximising a truth score in this exploration. The original Leiden routine uses CPM and selects restarts by generalized modularity; this exploration retains that procedure and seeds igraph explicitly. Null randomization intervals are not epidemic uncertainty intervals.

## Extending to later experiments

Copy `config.yaml` and change its selectors; paths are relative to the copied configuration. Existing scenario names and parameter definitions are taken from the main repository configuration, preserving its matched/mismatched meaning.

```yaml
conditions: [matched, mismatched]
scenarios: [baseline, substitution_rate_0.75, substitution_rate_1.25]
seeds: [12345, 12346, 12347]
models: [EDD, EDS, ESD, ESS]
```

```bash
python3 -m synthetic_exploration.run --config synthetic_exploration/my_experiment.yaml
```

| Model | Inference genetics | Observed genetics |
|---|---|---|
| EDD | Deterministic | Deterministic |
| EDS | Deterministic | Stochastic |
| ESD | Stochastic | Deterministic |
| ESS | Stochastic | Stochastic |

Matched/mismatched refers to parameter settings, independently of these two genetics axes. Identical generation parameters and seeds reuse observation caches, even across inference conditions. For mismatched runs, supervised comparisons train on the configured baseline generation parameters; matched runs use the active scenario. The training seed must differ from every evaluation seed. Perturbations and extra seeds are supported but **not run by the initial baseline command**.

To use a different tree, point a separate source configuration at it and use a distinct output directory. A row's `pair_id` and case order are specific to its tree. Multi-tree generalization and realistic introduction simulations require an explicit experimental design; the present code must not be interpreted as having validated them merely because it accepts a forest.

## Files and provenance

```text
synthetic_exploration/
  config.yaml                 # experiment selectors and fixed analysis choices
  run.py                      # orchestration
  truth.py                    # full-tree LCA and EpiLink relationship encoding
  data.py                     # observation simulation, cache, and scoring
  table.py                    # unified analysis table export/read helper
  observations.py             # investigation 1
  scores.py                   # investigation 2
  clusters.py                 # investigation 3
  validation.py               # investigation 4
  report.py                   # static scientific plots and readable report
  tests/                      # small, independent scientific correctness checks
  outputs/
    trees/<fingerprint>/      # shared relationship table
    datasets/<fingerprint>/   # dates/genomes-derived observations, cases, manifest
    analysis_table/           # one partitioned, directly readable analysis table
    runs/<condition>/<scenario>/seed_<seed>/
      manifest.json
      table_preview/
      01_observations/ ... 04_validation/
      figures/
      report.md
      report.html
```

Cache fingerprints include tree content, parameters, seed, simulator code, and package versions. The relationship cache additionally fingerprints its encoding implementation. Cached file content is checked before reuse. Run manifests record all source hashes, resolved parameters, and settings. Outputs are marked complete only after the investigations and report finish successfully.

Large regenerable Parquet tables are ignored by Git; CSV summaries, figures, reports, manifests, code, and configuration remain reviewable. Preserve the large tables separately or regenerate them with the command above.
