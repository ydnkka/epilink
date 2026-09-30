# Synthetic baseline: questions, comparisons, and operating criteria

## Purpose and study sequence

First establish what EpiLink adds at the **pairwise level**, then evaluate what
different clustering procedures recover and how their tuning parameters change
the result. Use development evidence to choose operating criteria and evaluate
frozen settings on held-out observation realizations. The next studies are
parameter-perturbation sensitivity and then empirical application.

This is the active protocol. Historical workflows and results are preserved in
[`archive/pre_reset_2026-09-30`](../archive/pre_reset_2026-09-30/ARCHIVE.md).

## 1. What are we trying to recover?

The primary target is **M=0: direct transmission AD(0) or shared infector
CA(0,0)**. Keep those two relationships separate in descriptive summaries.
Secondary endpoints are M≤1 and M≤2. For AD, M=m is the number of intermediates;
for CA, M=m1+m2 excludes the shared ancestor. Tree edge distance is M+1 for AD
and M+2 for CA. Inactive counts are null, not zero. CA(a,b) and CA(b,a) have the
same relationship interpretation; CA(0,2) and CA(1,1) remain distinguishable.

All observed unordered pairs are evaluated exactly once, excluding self-pairs.
Unsampled intermediates remain in the full-tree truth. Cross-introduction pairs,
if present in a future forest, are negative for every endpoint and reported
separately from finite M≥3 pairs. Missing truth within a connected tree is an error.

For M=0, **all M>0 pairs are false positives**. M≥3 contamination is an additional
measure of distant relationships, not the complete false-positive fraction.

## 2. Does EpiLink improve pairwise identification beyond genetic distance?

| Scorer | Inference genetics | Observed genetics | Output |
|---|---|---|---|
| EDD | Deterministic | Deterministic | Raw target compatibility |
| EDS | Deterministic | Stochastic | Raw target compatibility |
| ESD | Stochastic | Deterministic | Raw target compatibility |
| ESS | Stochastic | Stochastic | Raw target compatibility |
| GD_D / GD_S | None | Deterministic / stochastic | Genetic distance, lower is better |
| LOGIT_D / LOGIT_S | Supervised | Deterministic / stochastic | P(M=0 given GD, TD) |

Compare scorers within the same observed process, on identical cases/pairs/seeds.
Logistic regression uses an intercept and standardized GD and absolute TD, with
training-only standardization and prevalence-preserving count weights. Its
regularization is fixed in configuration. Identical feature cells are compressed
without changing their statistical weight. Neither case IDs nor truth geometry
are model inputs. M≤1/M≤2 evaluate the same M=0-trained score; a future
horizon-specific classifier must declare its distinct target.

**Evidence:** full tie-aware precision–recall curves, AP, absolute precision,
recall, F1, enrichment over prevalence, selected counts, and AD/CA/M composition.
Candidate budgets retain whole ties and report requested and achieved sizes.
Genetic ties are not broken using time or truth. Empty selections have undefined
precision and zero recall when positive pairs exist. Calibration curves, Brier
score, and log loss apply to logistic probabilities. Compatibility is not treated
as a calibrated probability, clipped to [0,1], or normalized into one.

## 3. What relationships do clusters contain?

| Approach | Input | Sweep |
|---|---|---|
| Connected components | Thresholded pair-score graph | Score or genetic threshold |
| Leiden | Same graph, declared weights | Graph threshold × resolution |
| TreeCluster, raw | FastME genetic-distance tree | Method × substitutions/site threshold |
| TreeCluster, dated | TreeTime-dated version | Method × day threshold |

All clusterers return one membership per sampled case, including isolates and
distinct TreeCluster `-1` singletons. Evaluate **all within-cluster pairs**, not
only retained graph edges. Report M=0 precision/recall/F1, broader endpoint
recovery, M≥3 contamination, direct-transmission retention, shared-infector
retention, singleton fraction, cluster-size distribution, largest-cluster fraction,
and extended BCubed against overlapping parent/child neighborhoods. Pair metrics
exclude self-pairs; extended BCubed retains its published self-pair convention.
ARI between methods, if added, would measure agreement rather than truth accuracy.

Binary Leiden graphs provide a common edge-weight convention for all scorers.
Native-weighted variants use unmodified positive EpiLink/logistic values. Genetic
Leiden uses binary edges; there is no threshold-dependent distance offset.
Connected components use inclusive thresholds, including zero-valued edges at
threshold zero. Native weighted graphs omit zero-weight edges and record this
distinction. Leiden's objective is explicit (default CPM), and restarts are
selected by that same objective, never by truth metrics. Resolution scales are
specific to their weight policy and scorer.

TreeCluster compares `max_clade`, `avg_clade`, and `single_linkage`. Raw trees
are midpoint-rooted using genetics alone; dated trees are rooted by TreeTime.
Negative genetic branches are handled by the declared `clip_zero` policy and
their count is recorded. Dating can change rooting/topology, so this comparison
measures the complete raw/dating pipelines. Threshold units and root changes are
recorded. Temporal thresholds denote tree branch distances, not simply the span
between sampling dates.

## 4. How do tuning parameters change the key metrics?

Use the declared broad grids in `config.yaml`, including empty selections,
strict settings and permissive settings. Inspect threshold curves, Leiden
threshold–resolution heatmaps, precision–recall/contamination frontiers, and
cluster-size behavior. Broaden a grid on development data when the useful region
touches its boundary; freeze the grid and selection rule before evaluation.

Observed feature overlap and partitions of a true-target graph can be useful
diagnostics. Neither an observed mixed feature cell nor a particular oracle-graph
partition establishes a universal population performance ceiling.

## 5. Which operating criteria should be carried forward?

1. Fit logistic scorers on **training** realizations only.
2. Sweep and inspect methods on **development** realizations.
3. Define a common operating criterion: e.g. balanced M=0 F1, or highest recall
   subject to precision, contamination, or workload constraints.
4. Select method-specific settings using that criterion, then freeze them.
5. Apply those settings unchanged on **held-out evaluation** realizations.

The supplied `balanced_M0` criterion is an executable starting comparison, not
an established epidemiological recommendation. Configurable constraints use
explicit `min`/`max` bounds, for example `M0_precision: {min: 0.5}`. Choose such
values from the intended use and development evidence. Bounds must hold on
every development realization. Among feasible settings, maximize the mean
objective, then prefer lower between-realization SD, then the stable setting ID.
Infeasible methods remain labeled infeasible; criteria are never silently relaxed.
TreeCluster methods are displayed separately and selected jointly within each
raw/dated observed-process pipeline. Pairwise thresholds and cluster settings
are selected independently. Frozen files contain criteria, development evidence
fingerprints, training identity and complete method definitions.

## Baseline design and provenance

The fixed backbone has 4,990 cases. Training, development and evaluation use
distinct seeds; historical seeds 12345/54321 are not held-out replicates. All
conclusions are conditional on this backbone, not independent-epidemic
generalization. Summaries use equal realization weights and report SD/range;
millions of dependent pairs are not used as independent uncertainty replicates.
Three evaluation realizations are a starting descriptive assessment, not precise
population confidence intervals. Scorer Monte Carlo and Leiden seeds are separate.

Generation and inference parameters are matched. EDD/EDS/ESD/ESS still distinguish
genetic process assumptions. Preserve the legacy EpiLink 0.1.5 convention explicitly:
`generation.genome_length=29903` controls mutation-count expectations, whereas
`simulation.sequence_length=5000` controls simulated sequence sites. Tree distances
divide observed Hamming counts by the latter. The dated clock is estimated, not
fixed to the nominal 29903-site mutation parameter. TD is absolute and rounded to
days for all scorers; TreeTime receives dates rounded by the same convention.

Input, implementation, package, parameter, seed, score, tree and membership hashes
protect artifact reuse. Stages load validated prerequisites and checkpoint their
outputs. Failed/missing TreeCluster results remain visible; a partial comparator
sweep cannot be frozen as a complete baseline.

## Run

From the repository root after `python -m pip install -e '.[test]'`:

```bash
# Dependency checks and experiment size; no simulation.
epilink-evaluate check --config synthetic_baseline/config.yaml

# Prepare training/development observations and shared truth only.
python -m synthetic_baseline.run --stage prepare

# Fit and compare scorers, then clustering, with a development report.
python -m synthetic_baseline.run --stage pairwise
python -m synthetic_baseline.run --stage clusters
# Equivalent dependency-aware development workflow:
python -m synthetic_baseline.run --stage develop

# After reviewing development evidence and configuring criteria:
python -m synthetic_baseline.run --stage select
python -m synthetic_baseline.run --stage evaluate

# Rebuild the report from saved tables.
python -m synthetic_baseline.run --stage report

# Small end-to-end validation, in a separate smoke output directory.
python -m synthetic_baseline.run --smoke --stage all
python -m pytest
```

FastME and TreeTime are used for tree construction/dating, and TreeCluster for
tree partitions. Tools are discovered on PATH or beside the active Python
interpreter; commands can also be configured explicitly. Reports list failures.
Raw/processed source inputs are preserved. `prepare-tree` regenerates a missing
backbone from raw SCoVMod data; an existing backbone is validated and retained.

## Outputs and extension points

`outputs/baseline/` contains shared `artifacts/` (truth, observations, fitted models,
scores, trees), and fingerprinted `runs/<id>/` directories. `current.json` points
to the current run. Each run contains `development/` (pairwise curves and clustering sweeps),
`selection/operating_points.json`, `evaluation/` (fixed-setting results), and
`report.md`, `report.html`, `figures/`, and a run manifest. Revising criteria after
accessing evaluation data requires fresh evaluation seeds. Smoke outputs use
`outputs/baseline_smoke/` and are labeled as pipeline validation.

Shared code is under `src/epilink_evaluation/`: inputs, truth, scorers, graphs,
phylogeny, clusterers, metrics, selection, workflows, and reporting. Scorers do
not compute evaluation metrics; clusterers do not access truth; reporting reads
saved result tables. `inputs.synthetic.analysis_table` provides an optional
joined view without duplicating stored truth for every model.

## Next studies

Once baseline operating criteria are settled, parameter perturbations compare
matched inference with baseline-fixed inference and logistic training, carrying
the baseline operating settings forward. Any retuning is a separate adaptation
analysis. Empirical application then reuses these definitions and records input
availability and external epidemiological evidence; exposure groups are not
complete transmission truth. Boston's preserved TN93 table is distance-censored
at 0.0005/site, so full-pair comparators require regenerated distances or an
explicitly restricted candidate universe.
