# Synthetic baseline: questions, comparisons, and operating criteria

## Purpose and study sequence

**Main question:** How well do EpiLink and its comparators identify close
transmission relationships, and which clustering settings provide useful
recovery–contamination trade-offs under baseline simulation conditions?

This is the **method-comparison and operating-point selection study**. It uses
simulated genetic distances and sampling times on a fixed transmission backbone,
where the true relationships between cases are known. Natural-history parameter
values are matched between generation and EpiLink inference; deterministic and
stochastic genetic-process assumptions are compared explicitly.

The study has three objectives:

1. **Measure pairwise discrimination:** compare EpiLink with genetic distance and
   logistic regression for identifying M=0 pairs (direct transmission or a shared
   infector), with broader relationships as secondary endpoints.
2. **Measure the effect of clustering:** compare connected components, Leiden,
   and raw/dated TreeCluster, including how thresholds, weights, and resolutions
   affect all within-cluster relationships and cluster sizes.
3. **Establish a reusable reference:** fit logistic models on training
   realizations, select operating points on development realizations, and measure
   their performance on held-out observation realizations.

**Evidence produced:** pairwise ranking curves, clustering parameter sweeps,
truth-based recovery and contamination metrics, and held-out results for frozen
settings. These results quantify performance conditional on the chosen backbone
and simulation design.

### How the three studies fit together

| Study                                                                | Scientific role                                                                   | Main evidence                                                                         |
| -------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Synthetic baseline** (this study)                                  | Compare methods, select settings, and evaluate them on held-out observations.     | Truth-based performance and frozen models/operating points.                           |
| [**Synthetic perturbation**](../02_synthetic_perturbation/README.md) | Test sensitivity to changed biological parameters and EpiLink parameter mismatch. | Paired performance changes with baseline-selected operating points held fixed.        |
| [**Boston application**](../03_boston_application/README.md)         | Examine empirical transfer and sensitivity to clustering settings on real data.   | Exposure composition/recovery, partition agreement, and descriptive parameter sweeps. |

The completed baseline supplies the reference for both downstream studies.
Perturbation and Boston each consume that reference directly; Boston does not
select its primary settings from perturbation results.

For setup, configuration fields, tree regeneration, stage behavior, saved
results, and troubleshooting, see the [operational guide](../../OPERATIONS.md).

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

| Scorer            | Inference genetics | Observed genetics          | Output                            |
| ----------------- | ------------------ | -------------------------- | --------------------------------- |
| EDD               | Deterministic      | Deterministic              | Raw target compatibility          |
| EDS               | Deterministic      | Stochastic                 | Raw target compatibility          |
| ESD               | Stochastic         | Deterministic              | Raw target compatibility          |
| ESS               | Stochastic         | Stochastic                 | Raw target compatibility          |
| GD_D / GD_S       | None               | Deterministic / stochastic | Genetic distance, lower is better |
| LOGIT_D / LOGIT_S | Supervised         | Deterministic / stochastic | P(M=0 given GD, TD)               |

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

| Approach             | Input                        | Sweep                                 |
| -------------------- | ---------------------------- | ------------------------------------- |
| Connected components | Thresholded pair-score graph | Score or genetic threshold            |
| Leiden               | Same graph, declared weights | Graph threshold × resolution          |
| TreeCluster, raw     | FastME genetic-distance tree | Method × substitutions/site threshold |
| TreeCluster, dated   | TreeTime-dated version       | Method × day threshold                |

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

Each experiment uses one fixed backbone from `inputs.tree_path`; reconstructed
trees can have a different size from the requested target component.
Read the tree provenance or the run's truth artifact manifest for the actual
case count; see [tree preparation](../../OPERATIONS.md#4-prepare-or-regenerate-the-scovmod-tree).
Training, development and evaluation use distinct observation seeds. All conclusions are conditional on the
chosen backbone, not independent-epidemic generalization. Summaries use equal
realization weights and report SD/range;
millions of dependent pairs are not used as independent uncertainty replicates.
Three evaluation realizations are a starting descriptive assessment, not precise
population confidence intervals. Scorer Monte Carlo, Leiden, and TreeTime seeds
are separate. `treecluster.rng_seed` controls TreeTime's stochastic choices and is
included in both the command and the tree artifact signature.

Generation and inference parameters are matched. EDD/EDS/ESD/ESS still distinguish
genetic process assumptions. The EpiLink 0.1.5 configuration uses:
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
epilink-evaluate check --config evaluation/01_synthetic_baseline/config.yaml

# Prepare or reuse the shared transmission backbone (also automatic below).
epilink-evaluate scovmod --stage prepare --config evaluation/01_synthetic_baseline/config.yaml

# Prepare training/development observations and shared truth only.
python evaluation/01_synthetic_baseline/run.py --stage prepare

# Fit and compare scorers, then clustering, with a development report.
python evaluation/01_synthetic_baseline/run.py --stage pairwise
python evaluation/01_synthetic_baseline/run.py --stage clusters
# Equivalent dependency-aware development workflow:
python evaluation/01_synthetic_baseline/run.py --stage develop

# After reviewing development evidence and configuring criteria:
python evaluation/01_synthetic_baseline/run.py --stage select
python evaluation/01_synthetic_baseline/run.py --stage evaluate

# Rebuild the report from saved tables.
python evaluation/01_synthetic_baseline/run.py --stage report

# Small end-to-end validation, in a separate smoke output directory.
python evaluation/01_synthetic_baseline/run.py --smoke --stage all
python -m pytest
```

The completed 64-case run, test coverage, and full-scale resumption steps are
recorded in [VALIDATION.md](VALIDATION.md).

FastME and TreeTime are used for tree construction/dating, and TreeCluster for
tree partitions. Tools are discovered on PATH or beside the active Python
interpreter; commands can also be configured explicitly. Reports list failures.
Raw/processed source inputs are preserved. `scovmod --stage prepare`
and the baseline workflow share the same input preparation. Managed backbones
are reused when the source hashes, construction settings, implementation and
output checksums match. Explicit prebuilt trees without a manifest are retained.

`scovmod --stage prepare` prepares only the transmission backbone and provenance.
`baseline --stage prepare` additionally prepares truth and training/development
observations. The `scovmod` command defaults to `prepare` and supports only that stage.

## Outputs and extension points

Column definitions, formulas, missing-value conventions, metadata fields, and
analysis joins are documented in the [output reference](../../OUTPUTS.md).

`outputs/inputs/` contains `transmission_tree.gml`, its
`transmission_tree.source.json` provenance, and `manifest.json`. Full and smoke
runs share this backbone; `--output` changes the run root, not the input paths.
Each run's `inputs.json` records the tree it used.

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

After baseline evaluation, the [perturbation workflow](../02_synthetic_perturbation/README.md)
replays frozen models and operating settings on paired new observations. Start
with `python evaluation/02_synthetic_perturbation/run.py --smoke`, then omit `--smoke` for all
configured parameter levels. Matched and baseline-fixed modes differ in EpiLink
inference; logistic training stays baseline-fixed in both. Any retuning or
retraining is a separate adaptation analysis.

The [Boston empirical application](../03_boston_application/README.md) applies the
same frozen definitions to observed Boston outbreak data. It tests transfer and
describes exposure composition, recovery, and method agreement. Its separate
`--stage explore` analysis varies clustering settings on those same observations
to characterize sensitivity. Complete transmission truth is unavailable, so the
Boston summaries provide descriptive evidence rather than truth-validated
operating-point selection. Boston's TN93 table is distance-censored at 0.0005/site,
so missing pairs are treated as unobserved rather than zero distance.
