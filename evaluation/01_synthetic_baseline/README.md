# Synthetic baseline: questions, comparisons, and operating criteria

## Purpose and study sequence

**Main question:** How well do EpiLink and comparators identify close transmission relationships, and which clustering settings provide useful recovery–contamination trade-offs?

**Method-comparison and operating-point selection study.** Simulated genetic distances/sampling times on fixed transmission backbone (known truth). Parameters matched between generation and EpiLink inference; deterministic/stochastic genetic processes compared.

Follows [synthetic diagnostics](../00_synthetic_diagnostics/README.md) on [shared experiment](../shared_synthetic/README.md). Requires complete diagnostics, then fits logistic models on training realizations. Evaluation observations generated after frozen settings validated.

**Objectives:**

1. **Pairwise discrimination:** EpiLink vs genetic distance vs logistic regression for M=0 pairs (direct transmission/shared infector); broader endpoints secondary.
2. **Clustering effect:** components, Leiden, raw/dated TreeCluster; thresholds, weights, resolutions affect within-cluster relationships/sizes.
3. **Reusable reference:** logistic models on training, operating points on development, performance on held-out observations.

**Evidence:** pairwise curves, clustering sweeps, truth-based recovery/contamination, held-out frozen settings. Performance conditional on backbone/simulation design.

### How the studies fit together

| Study | Role | Evidence |
|-------|------|----------|
| [**Diagnostics**](../00_synthetic_diagnostics/README.md) | Feature ambiguity, known-truth controls | GD/GD_TD cells, oracle graph partitions |
| **Baseline** (this study) | Compare methods, select settings, evaluate | Truth-based performance, frozen models/points |
| [**Perturbation**](../02_synthetic_perturbation/README.md) | Test EpiLink under inference mismatch/resolution adaptation | Paired clustering changes |
| [**Boston**](../03_boston_application/README.md) | Apply frozen settings to empirical data | Exposure concentration/recovery, partition agreement |

Baseline supplies reference for downstream studies. Perturbation/Boston consume reference directly; Boston doesn't select from perturbation.

See [operational guide](../../OPERATIONS.md) for setup, configuration, tree regeneration, stages, results, troubleshooting.

## 1. What are we trying to recover?

**Primary target: M=0** (AD(0) direct transmission or CA(0,0) shared infector). Keep separate in summaries. Secondary: M≤1, M≤2.

AD: M = intermediates. CA: M = m1+m2 (excludes shared ancestor). Tree edge distance: M+1 (AD), M+2 (CA). Inactive counts: null (not zero). CA(a,b) = CA(b,a); CA(0,2) ≠ CA(1,1).

All observed unordered pairs evaluated once (no self-pairs). Unsampled intermediates in full-tree truth. Cross-introduction pairs: negative for all endpoints, reported separately from M≥3. Missing truth in connected tree = error.

M=0: **all M>0 pairs are false positives**. M≥3 contamination: distant relationships only (not all false positives).

## 2. Does EpiLink improve pairwise identification beyond genetic distance?

| Scorer | Inference | Observed | Output |
|--------|-----------|----------|--------|
| EDD | Deterministic | Deterministic | Raw compatibility |
| EDS | Deterministic | Stochastic | Raw compatibility |
| ESD | Stochastic | Deterministic | Raw compatibility |
| ESS | Stochastic | Stochastic | Raw compatibility |
| GD_D / GD_S | None | Deterministic / stochastic | Genetic distance (lower better) |
| LOGIT_D / LOGIT_S | Supervised | Deterministic / stochastic | P(M=0 \| GD, TD) |

Compare scorers within same observed process/cases/pairs/seeds. Logistic: intercept + standardized GD/TD (training-only standardization), prevalence-preserving weights. Regularization fixed. Feature cells compressed (weight unchanged). No case IDs/truth geometry as inputs. M≤1/M≤2 evaluate M=0-trained score.

**Evidence:** tie-aware PR curves, AP, precision, recall, F1, enrichment, counts/fractions, AD/CA/M composition. Genetic ties not broken by time/truth. Empty selection: precision undefined, recall zero (if positives exist). Calibration/Brier/log loss: logistic only. Compatibility ≠ calibrated probability.

Pairwise/component selection: union of distinct development scores + empty selection. Inclusive cutoffs shared across seeds. Pairwise: cumulative PR curves. Components: incremental connectivity (newly co-clustered pairs per merge, unchanged partitions reused). Pipelines maximize equal-seed mean objective (constraints per seed). Evaluation: frozen cutoff.

## 3. What relationships do clusters contain?

| Approach | Input | Sweep |
|----------|-------|-------|
| Connected components | Thresholded pair-score graph | Score/genetic threshold |
| Leiden | Full observed graph, scorer weights | Resolution only |
| TreeCluster (raw) | IQ-TREE ML tree | Method × SNP cutoff (→ subs/site) |
| TreeCluster (dated) | IQ-TREE/LSD2-dated tree | Method × day threshold |

One membership per case (including isolates, TreeCluster `-1` singletons). Evaluate **all within-cluster pairs** (not just graph edges). Report: M=0 precision/recall/F1, endpoint recovery, M≥3 contamination, AD/CA retention, singleton fraction, cluster sizes, largest-cluster fraction. No self-pairs.

Leiden: **full graphs, no cutoff**, resolution only. EpiLink/logistic scorers (edges = original scores, including zeros). Pipelines: `leiden/EDD`, `leiden/EDS`, `leiden/ESD`, `leiden/ESS`, `leiden/LOGIT_D`, `leiden/LOGIT_S`. Genetic-distance scorers excluded (unit weights on all edges). Objective: explicit (default CPM); restarts by objective (not truth). `clustering.leiden.resolutions`:

```yaml
resolutions:
  min: 0.01, max: 1.0, initial_points: 8, budget: 28, scale: log
```

Search: both bounds, log-spaced trials, refines promising intervals (per pipeline/criterion objective, per-seed constraints). Refinement batches explore widest interval. Scorers share resolutions (paired comparisons); optima independent. Graph construction reused; trials checkpointed. Budget: distinct resolutions (all seeds/scorers). `scale: linear` available; `tolerance` (0.001) stops narrow intervals. Selection: best feasible within bounds.

TreeCluster search:

```yaml
genetic_threshold_snps: {min: 0, max: 10, initial_points: 4, budget: 11, scale: linear}
temporal_threshold_days: {min: 0, max: 105, initial_points: 8, budget: 28, scale: linear}
```

Raw: integer SNPs → subs/site (alignment length). Refinement: untested integer in interval; stops at budget/domain exhaustion (0–10 = 11 cutoffs). Dated: real days (fractional allowed). Searches share cutoffs across processes/methods. Intervals ranked by development objective; selection: best method/cutoff per raw/dated pipeline. Budget: distinct cutoffs per tree kind. Explicit lists supported. Phylogenies/partitions reused; frozen evaluation/Boston: no new cutoffs.

TreeCluster: `max_clade`, `avg_clade`, `single_linkage`. IQ-TREE: raw trees (reference as outgroup, pruned). LSD2: dated trees (no reference). Branches: subs/site (raw), days (dated); non-finite/negative rejected. Temporal cutoffs = tree branch distances.

## 4. How do tuning parameters change the key metrics?

Use clustering grids in `config.yaml`. Inspect: component cutoff curves, Leiden resolution curves, precision–recall/contamination frontiers, cluster sizes. Broaden grid if settings touch boundary; freeze before evaluation.

Reports/figures: **M0, Mle1, Mle2** (endpoint suffixes on filenames). `frontier.csv`: endpoint-labelled (`*_precision_mean`/`*_recall_mean`). Held-out tables: criterion's frozen objective; `operating_summary.csv`: all endpoints, AD0/CA00 retention, workload, cluster sizes, defined-value counts. Endpoint from objective (not criterion name).

Diagnostics: exact `GD`/`GD_TD` cells (both processes, all endpoints). Min feature-only error: `sum_cells min(n_target, n_other)` / all pairs. Applies to decisions constant within cells (not population performance/partition recovery). Mixed-cell prevalence/fractions: different denominators.

Oracle graphs: M≤h pairs, unit weights. Components/Leiden: all within-cluster pairs (including nonedges). Unsampled intermediates in truth. Controls deduplicated by truth/endpoint/sampled-case set (full-sampling ≠ independent). Oracle performance/feature ambiguity ≠ universal ceiling.

## 5. Which operating criteria should be carried forward?

1. Fit logistic scorers on **training** realizations.
2. Sweep/inspect methods on **development** realizations.
3. Define common criterion: e.g. balanced M=0 F1, or recall subject to precision/contamination/workload constraints.
4. Select method-specific settings; freeze.
5. Apply unchanged on **held-out evaluation** realizations.

Supplied criteria: `balanced_M0`, `balanced_Mle1`, `balanced_Mle2` (maximize endpoint F1). M=0 primary; broader endpoints evaluate M=0-trained scores. Starting comparisons (not epidemiological recommendations). Constraints: `min`/`max` bounds (e.g., `M0_precision: {min: 0.5}`). Choose from intended use/development evidence. Bounds must hold on every development realization.

Selection: maximize mean objective → lower SD → stable setting ID. Infeasible methods labeled infeasible (no silent relaxation). TreeCluster: displayed separately, selected jointly per raw/dated pipeline. Pairwise/cluster settings selected independently. Frozen files: criteria, evidence fingerprints, training identity, method definitions.

## Baseline design and provenance

One backbone from `inputs.tree_path`; actual case count may differ from requested component size. Read provenance/truth manifest; see [tree preparation](../../OPERATIONS.md#4-prepare-or-regenerate-the-scovmod-tree). Training/development/evaluation: distinct observation seeds. Equal-seed means/SDs/ranges: observation variation (conditional on backbone, not independent epidemics/pairs). Scorer Monte Carlo, Leiden, IQ-TREE: separate seeds.

[Shared config](../shared_synthetic/config.yaml): `inputs`, `generation`, `simulation`, `splits`; baseline derives matched inference. EDD/EDS/ESD/ESS: genetic assumptions. `generation.genome_length=29903` (mutation expectations), `simulation.sequence_length=5000` (sequence sites). IQ-TREE: branch lengths from sequences; SNP cutoffs / alignment length. LSD2: clock estimation (unless `phylogeny.clock_rate` fixed). Scorer TD: absolute, rounded to days; tree inference: original (fractional) dates.

Hashes protect artifact reuse (input, implementation, package, parameter, seed, score, tree, membership). Stages: validated prerequisites, checkpointed outputs. Failed/missing TreeCluster visible; partial sweep ≠ complete baseline.

## Run

From repository root (`python -m pip install -e '.[test]'`):

```bash
# Dependency checks (no simulation)
epilink-evaluate check --config evaluation/01_synthetic_baseline/config.yaml

# Prepare backbone (also automatic in diagnostics)
epilink-evaluate scovmod --stage prepare --config evaluation/01_synthetic_baseline/config.yaml

# Complete diagnostics first
python evaluation/00_synthetic_diagnostics/run.py --stage all

# Optional: prepare training observations
python evaluation/01_synthetic_baseline/run.py --stage prepare

# Fit scorers, clustering, development report
python evaluation/01_synthetic_baseline/run.py --stage pairwise
python evaluation/01_synthetic_baseline/run.py --stage clusters
# or:
python evaluation/01_synthetic_baseline/run.py --stage develop

# After reviewing development, configure criteria:
python evaluation/01_synthetic_baseline/run.py --stage select
python evaluation/01_synthetic_baseline/run.py --stage evaluate

# Rebuild report
python evaluation/01_synthetic_baseline/run.py --stage report

# Smoke workflow (separate roots)
python evaluation/00_synthetic_diagnostics/run.py --smoke --stage all
python evaluation/01_synthetic_baseline/run.py --smoke --stage all
```

Baseline default: `develop`. Diagnostics default: `all` (`prepare`, `backbone`, `observations`, `graphs`, `report`). `prepare` alone ≠ complete diagnostics. Complete diagnostics before baseline.

IQ-TREE: raw trees + LSD2 dating. TreeCluster: partitions. Set `phylogeny.executable`/`treecluster.executable` to override PATH discovery. Reports expose failures. `scovmod --stage prepare`: backbone + provenance (shared with diagnostics). Artifacts reused when inputs/settings/identities/checksums match; prebuilt backbones without manifests retained.

`scovmod --stage prepare`: backbone/provenance only. Diagnostics: truth/development observations. Baseline `prepare`: requires diagnostics, prepares training observations (reuses development).

## Outputs and extension points

Column definitions, formulas, missing values, metadata, joins: [output reference](../../OUTPUTS.md).

`../shared_synthetic/outputs/inputs/`: `transmission_tree.gml`, `transmission_tree.source.json`, `manifest.json`. Full/smoke share backbone; `--output` changes run root (not input paths). `inputs.json`: pinned tree + provenance.

`../shared_synthetic/outputs/synthetic/`: `artifacts/backbones/`, `artifacts/truth/`, `artifacts/observations/`, `experiments/<id>/`. Baseline's `<run>/experiment.json` pins `experiment_directory`/`fingerprint`; resolve joins from there (not baseline artifacts/latest pointer).

`outputs/baseline/`: `artifacts/` (models, scores, trees), `runs/<id>/`. `current.json`: current run. Per run: `development/` (pairwise curves, clustering sweeps), `selection/operating_points.json`, `evaluation/` (fixed-setting results), `report.md`, `report.html`, `figures/`, manifest.

Revising criteria after evaluation access: fresh evaluation seeds + diagnostics. `heldout_access/seed_<seed>.json` survives `reset-outputs` (as do all shared outputs). Smoke: `outputs/baseline_smoke/`, `../shared_synthetic/outputs/synthetic_smoke/`. Full-evaluation seeds: 63101–63103. Smoke: separate validation seeds (≠ scientific evaluation).

`pairwise/evidence/`: curves, rankings, calibration, empty-selection metrics. `development/cutoff_candidates/`: shared pairwise/component definitions + source-manifest hashes. Component sweeps: `development/seed_<seed>/clusters/component_sweeps/<scorer>/`; one membership/cluster per distinct partition; selected settings: `<setting-id>/` artifacts. `resolution_search/`: Leiden trials. `treecluster_search/{raw,dated}/`: cutoff trials, search status (SNP/day units). `settings.json`: all candidates (restored on replay).

Shared code: `src/epilink_evaluation/` (inputs, truth, scorers, graphs, phylogeny, clusterers, metrics, selection, workflows, reporting). Scorers ≠ evaluation metrics; clusterers ≠ truth; reporting reads saved tables. `inputs.synthetic.analysis_table`: optional joined view.

No `pairwise`/`thresholds` config block. Pairwise/components: shared development-score candidates. Leiden: bounded adaptive resolution search.

## Manuscript figures and tables

After `evaluate`, generate displays from saved results (no simulation/refitting):

```bash
python -m evaluation.results.fig03  # M=0 compatibility (EpiLink ED/ES)
python -m evaluation.results.fig04  # pairwise discrimination
python -m evaluation.results.fig05  # components
python -m evaluation.results.fig06  # full-graph Leiden
python -m evaluation.results.fig07  # resolution regret
python -m evaluation.results.fig08  # raw TreeCluster
python -m evaluation.results.fig09  # dated TreeCluster
python -m evaluation.results.fig10  # graph operating bars
python -m evaluation.results.fig11  # TreeCluster operating bars
python -m evaluation.results.fig12  # held-out graph/tree comparison
python -m evaluation.results.tab01  # main operating points
python -m evaluation.results.tab02  # full operating points
```

Default: baseline root's `current.json`. Output: `evaluation/results/outputs/01_synthetic_baseline/<run-id>/`. Options: `--run-dir`, `--output-dir`; figures: `--format pdf|png|both` (default `both`). See [results index](../results/README.md). Require completed evaluation; use frozen `balanced_M0` (`M0_f1`). `.tex` files: `thesistablebody`/`longtable` macros (`landscape` for wide tables).

Figure labels (not saved column names): `GD_D`/`GD_S` = **GDD**/**GDS**, `LOGIT_D`/`LOGIT_S` = **LGD**/**LGS**. Deterministic observed: **EDD, ESD, GDD, LGD**; stochastic observed: **EDS, ESS, GDS, LGS**. EpiLink code: 2nd letter = inference, 3rd = observed (D/S). EpiLink/logistic: temporal distance; GDD/GDS: genetic distance only. Compare within observed-process column.

| Output | Source/interpretation |
|--------|----------------------|
| `fig03_primary_compatibility_surfaces` | M=0 compatibility (ED/ES EpiLink inference) on GD (0–15 SNP) × TD (0–20 day) grid. Panels ≠ observed-genetics datasets. Same color scale; scores ≠ probabilities. |
| `fig04_pairwise_discrimination` | **Development** M=0 PR curves (one line/seed); diamonds = means at frozen cutoffs. Columns: four scorers per observed process. Legend AP: **held-out** means. |
| `fig05_components`, `fig06_leiden` | Development precision/recall, F1/M≥3 contamination. Components: cutoff sweep; Leiden: resolution sweep. Diamonds: frozen settings. |
| `fig07_epilink_resolution_regret` | Development shared-resolution assessment (four EpiLink pipelines). F1 loss vs best resolution (mean/IQR/worst %). Compromise: min worst loss, then mean loss. |
| `fig08_treecluster_raw`, `fig09_treecluster_dated` | **Development** figures. Lines: TreeCluster methods. Diamond: selected method+threshold per raw/dated pipeline. Raw: subs/site; dated: days. Axes: recovery/contamination. |
| `fig10_graph_cluster_operating_bars` | **Held-out** M=0 precision/recall/F1, M≥3 contamination. Rows: components/Leiden; columns: observed genetics. Whiskers: across-seed SD. |
| `fig11_treecluster_operating_bars` | **Held-out** metrics (same scale). Panels: raw/dated trees. Categories: deterministic/stochastic observed genetics. Each: selected method/threshold (SNP/days). Comparisons: separate frozen pipelines. |
| `fig12_cluster_recovery_contamination` | Compact held-out M=0 F1 vs M≥3 contamination (all pipelines). Separate observed-process panels. CSV: means + SD. |
| `tab01_operating_points.tex` | Held-out: pairwise, Leiden (EpiLink/logistic), raw/dated TreeCluster. One Leiden pipeline per scorer. |
| `tab02_operating_points_full.tex` | Supplementary: all pipelines (EpiLink, components, Leiden, TreeCluster). |

All cluster panels: **every within-cluster pair** (including nonedges). M=0: M≥3 contamination = distant false positives only. Tables: equal-realization means (sample SD), percentages (fractions), counts (selected pairs). TreeCluster raw: SNP counts (from subs/site); dated: days. Undefined: dashes. Pairwise rows: no cluster metrics. `selection/operating_points.json`, `evaluation/selection_used.json`: frozen definitions (scripts verify coverage). Between-seed SD: descriptive (not confidence interval).

Regret: `balanced_M0`, feasibility constraints, mean/SD/stable-ID tie-breaking. Proposes shared resolution (four EpiLink pipelines); baseline replays individual resolutions. New shared-resolution evaluation: fresh held-out observations.

## Next studies

After baseline evaluation, [perturbation](../02_synthetic_perturbation/README.md): EpiLink clustering only (baseline/matched inference × baseline/updated resolution). Updated resolutions: fresh development seeds; four arms: paired separate seeds. Start: `python evaluation/02_synthetic_perturbation/run.py --smoke`, then full study.

[Boston](../03_boston_application/README.md): apply frozen definitions to Boston outbreak data. Tests transfer; describes exposure composition, recovery, graph/tree partition agreement. No transmission truth; descriptive evidence (not truth-validated selection). TN93 censored at 0.0005/site; missing pairs = unobserved (not zero).
