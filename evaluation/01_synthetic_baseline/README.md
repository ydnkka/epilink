# Synthetic baseline: questions, comparisons, and operating criteria

## Purpose and study sequence

**Main question:** How well do EpiLink and its comparators identify close transmission relationships, and which clustering settings provide useful recovery–contamination trade-offs under baseline simulation conditions?

This is the **method-comparison and operating-point selection study**. It uses simulated genetic distances and sampling times on a fixed transmission backbone, where the true relationships between cases are known. Parameter values are matched between generation and EpiLink inference; deterministic and stochastic genetic-process assumptions are compared explicitly.

It follows [synthetic diagnostics](../00_synthetic_diagnostics/README.md) on the same [shared experiment](../shared_synthetic/README.md). Baseline requires complete diagnostic evidence for the exact development observation artifacts, then fits logistic models on separate training realizations. Evaluation observations are generated only after frozen operating settings are validated by `evaluate`.

The study has three objectives:

1. **Measure pairwise discrimination:** compare EpiLink with genetic distance and logistic regression for identifying M=0 pairs (direct transmission or a shared infector), with broader relationships as secondary endpoints.
2. **Measure the effect of clustering:** compare connected components, Leiden, and raw/dated TreeCluster, including how thresholds, weights, and resolutions affect all within-cluster relationships and cluster sizes.
3. **Establish a reusable reference:** fit logistic models on training realizations, select operating points on development realizations, and measure their performance on held-out observation realizations.

**Evidence produced:** pairwise ranking curves, clustering parameter sweeps, truth-based recovery and contamination metrics, and held-out results for frozen settings. These results quantify performance conditional on the chosen backbone and simulation design.

### How the studies fit together

| Study                                                                | Scientific role                                                                        | Main evidence                                                                               |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [**Synthetic diagnostics**](../00_synthetic_diagnostics/README.md)   | Characterize development feature ambiguity and known-truth controls before comparison. | Exact GD/GD_TD cells, endpoint-oracle graph partitions, and transmission-hop tree controls. |
| **Synthetic baseline** (this study)                                  | Compare methods, select settings, and evaluate them on held-out observations.          | Truth-based performance and frozen models/operating points.                                 |
| [**Synthetic perturbation**](../02_synthetic_perturbation/README.md) | Test sensitivity to changed biological parameters and EpiLink parameter mismatch.      | Paired performance changes with baseline-selected operating points held fixed.              |
| [**Boston application**](../03_boston_application/README.md)         | Examine empirical transfer and sensitivity to clustering settings on real data.        | Exposure composition/recovery, partition agreement, and descriptive parameter sweeps.       |

The completed baseline supplies the reference for both downstream studies. Perturbation and Boston each consume that reference directly; Boston does not select its primary settings from perturbation results.

For setup, configuration fields, tree regeneration, stage behavior, saved results, and troubleshooting, see the [operational guide](../../OPERATIONS.md).

## 1. What are we trying to recover?

The primary target is **M=0: direct transmission AD(0) or shared infector CA(0,0)**. Keep those two relationships separate in descriptive summaries. Secondary endpoints are M≤1 and M≤2. For AD, M=m is the number of intermediates; for CA, M=m1+m2 excludes the shared ancestor. Tree edge distance is M+1 for AD and M+2 for CA. Inactive counts are null, not zero. CA(a,b) and CA(b,a) are symmetric and so have the same relationship interpretation; CA(0,2) and CA(1,1) remain distinguishable.

All observed unordered pairs are evaluated exactly once, excluding self-pairs. Unsampled intermediates remain in the full-tree truth. Cross-introduction pairs, if present in a future forest, are negative for every endpoint and reported separately from finite M≥3 pairs. Missing truth within a connected tree is an error.

For M=0, **all M>0 pairs are false positives**. M≥3 contamination is an additional measure of distant relationships, not the complete false-positive fraction.

## 2. Does EpiLink improve pairwise identification beyond genetic distance?

| Scorer            | Inference genetics | Observed genetics          | Output                            |
| ----------------- | ------------------ | -------------------------- | --------------------------------- |
| EDD               | Deterministic      | Deterministic              | Raw target compatibility          |
| EDS               | Deterministic      | Stochastic                 | Raw target compatibility          |
| ESD               | Stochastic         | Deterministic              | Raw target compatibility          |
| ESS               | Stochastic         | Stochastic                 | Raw target compatibility          |
| GD_D / GD_S       | None               | Deterministic / stochastic | Genetic distance, lower is better |
| LOGIT_D / LOGIT_S | Supervised         | Deterministic / stochastic | P(M=0 given GD, TD)               |

Compare scorers within the same observed process, on identical cases/pairs/seeds. Logistic regression uses an intercept and standardized GD and absolute TD, with training-only standardization and prevalence-preserving count weights. Its regularization is fixed in configuration. Identical feature cells are compressed without changing their statistical weight. Neither case IDs nor truth geometry are model inputs. M≤1/M≤2 evaluate the same M=0-trained score; a future horizon-specific classifier must declare its distinct target.

**Evidence:** full tie-aware precision–recall curves, AP, absolute precision, recall, F1, enrichment over prevalence, selected counts, and AD/CA/M composition. Candidate budgets retain whole ties and report requested and achieved sizes. Genetic ties are not broken using time or truth. Empty selections have undefined precision and zero recall when positive pairs exist. Calibration curves, Brier score, and log loss apply to logistic probabilities. Compatibility is not treated as a calibrated probability, clipped to [0,1], or normalized into one.

The supplied `pairwise.threshold_mode: all_development_scores` evaluates the union of distinct score values across **development** realizations, plus an empty selection. Each candidate is one shared inclusive cutoff applied to every seed; metrics are looked up from cumulative whole-tie PR curves. Selection uses equal realization means and per-realization constraints, not pooled pairs or separately optimized per-seed thresholds. Evaluation replays the frozen numerical cutoff. `configured` mode instead uses the finite `thresholds` lists. Graph clustering always uses those finite lists, independently of the exact pairwise candidates.

## 3. What relationships do clusters contain?

| Approach             | Input                        | Sweep                                 |
| -------------------- | ---------------------------- | ------------------------------------- |
| Connected components | Thresholded pair-score graph | Score or genetic threshold            |
| Leiden               | Same graph, declared weights | Graph threshold × resolution          |
| TreeCluster, raw     | FastME genetic-distance tree | Method × substitutions/site threshold |
| TreeCluster, dated   | TreeTime-dated version       | Method × day threshold                |

All clusterers return one membership per sampled case, including isolates and distinct TreeCluster `-1` singletons. Evaluate **all within-cluster pairs**, not only retained graph edges. Report M=0 precision/recall/F1, broader endpoint recovery, M≥3 contamination, direct-transmission retention, shared-infector retention, singleton fraction, cluster-size distribution, largest-cluster fraction, and extended BCubed against overlapping parent/child neighborhoods. Pair metrics exclude self-pairs; extended BCubed retains its published self-pair convention. ARI between methods, if added, would measure agreement rather than truth accuracy.

Binary Leiden graphs provide a common edge-weight convention for all scorers. Native-weighted variants use unmodified positive EpiLink/logistic values. Genetic Leiden uses binary edges; there is no threshold-dependent distance offset. Connected components use inclusive thresholds, including zero-valued edges at threshold zero. Native weighted graphs omit zero-weight edges and record this distinction. Leiden's objective is explicit (default CPM), and restarts are selected by that same objective, never by truth metrics. Resolution scales are specific to their weight policy and scorer. `clustering.leiden.resolutions_by_weight_policy` can override the default `resolutions` list for `binary` or `native`. The supplied native grid extends below 0.1; binary and native weights need not share a resolution scale.

TreeCluster compares `max_clade`, `avg_clade`, and `single_linkage`. Raw trees are midpoint-rooted using genetics alone; dated trees are rooted by TreeTime. Negative genetic branches are handled by the declared `clip_zero` policy and their count is recorded. Dating can change rooting/topology, so this comparison measures the complete raw/dating pipelines. Threshold units and root changes are recorded. Temporal thresholds denote tree branch distances, not simply the span between sampling dates.

## 4. How do tuning parameters change the key metrics?

Use the declared broad clustering grids in `config.yaml`, including empty selections, strict settings and permissive settings. Inspect threshold curves, Leiden threshold–resolution heatmaps, precision–recall/contamination frontiers, and cluster-size behavior. Broaden a grid on development data when the useful region touches its boundary; freeze the grid and selection rule before evaluation.

Reports and figures cover **M0, Mle1 and Mle2**, with endpoint suffixes on figure filenames. `frontier.csv` is endpoint-labelled and retains the explicit `*_precision_mean`/`*_recall_mean` names. Held-out tables display each criterion's actual frozen objective; `operating_summary.csv` retains all endpoints, AD0/CA00 retention, workload, cluster-size metrics and defined-value counts. The endpoint is derived from the objective, not from the criterion's name.

`development/grid_adequacy.csv` compares each selected development setting with `grid_audit.reference` on the same observations. The reference may override `thresholds`, `leiden_resolutions`, and TreeCluster `genetic_thresholds` or `threshold_days`; unspecified fields inherit the current configuration. Keep reference clustering settings in the expanded sweep so coverage remains complete. Pairwise reference thresholds are evaluated from the saved full PR curves. `grid_neighbors.csv` reports adjacent settings with other coordinates and the TreeCluster method held fixed, including feasibility and precision/recall, contamination and cluster-size behavior.

The audit distinguishes natural zero cutoffs from arbitrary search boundaries, and reports `material_improvement`, `within_tolerance`, `not_refined`, `reference_better`, or `unassessed`. The supplied absolute objective tolerance is 0.005. It is a numerical refinement diagnostic, not a confidence interval or proof of a global optimum. Inspect boundary flags and scientific trade-offs as well as objective changes before freezing. An unchanged reference grid is labelled `not_refined`, rather than evidence of refinement stability.

The preceding diagnostics study reports exact `GD` and `GD_TD` feature cells for both genetic processes and every endpoint. The minimum empirical feature-only error count is `sum_cells min(n_target, n_other)`; its rate divides by all observed pairs. It applies to decisions constant within those cells, not population performance or partition recovery. Mixed-cell target prevalence and the fraction of all targets in mixed cells have different denominators.

Endpoint-oracle graphs connect finite M≤h pairs with unit weights; components and Leiden are assessed on all within-cluster pairs, including graph nonedges. The known transmission-hop tree retains unsampled intermediates and represents sampled ancestors as zero-length terminal tips. Each method/threshold partition of that one tree is evaluated at all endpoints. These controls are deduplicated by truth and sampled-case set across seeds/processes; full-sampling repeats are not independent controls. The tree control explicitly rejects forests with visible failure. Neither oracle partition performance nor feature ambiguity is a universal performance ceiling; the hop tree is not a molecular genealogy.

## 5. Which operating criteria should be carried forward?

1. Fit logistic scorers on **training** realizations only.
2. Sweep and inspect methods on **development** realizations.
3. Define a common operating criterion: e.g. balanced M=0 F1, or highest recall subject to precision, contamination, or workload constraints.
4. Select method-specific settings using that criterion, then freeze them.
5. Apply those settings unchanged on **held-out evaluation** realizations.

The supplied `balanced_M0`, `balanced_Mle1`, and `balanced_Mle2` criteria maximize their respective endpoint F1 values. M=0 remains primary; broader endpoints evaluate the same M=0-trained scores. These are executable starting comparisons, not an established epidemiological recommendation. Configurable constraints use explicit `min`/`max` bounds, for example `M0_precision: {min: 0.5}`. Choose such values from the intended use and development evidence. Bounds must hold on every development realization. Among feasible settings, maximize the mean objective, then prefer lower between-realization SD, then the stable setting ID. Infeasible methods remain labeled infeasible; criteria are never silently relaxed. TreeCluster methods are displayed separately and selected jointly within each raw/dated observed-process pipeline. Pairwise thresholds and cluster settings are selected independently. Frozen files contain criteria, development evidence fingerprints, training identity and complete method definitions.

## Baseline design and provenance

Each experiment uses one fixed backbone from the shared config's `inputs.tree_path`; reconstructed trees can have a different size from the requested target component. Read the tree provenance or the pinned shared truth artifact manifest for the actual case count; see [tree preparation](../../OPERATIONS.md#4-prepare-or-regenerate-the-scovmod-tree). Training, development and evaluation use distinct observation seeds. All conclusions are conditional on the chosen backbone, not independent-epidemic generalization. Summaries use equal realization weights and report SD/range; millions of dependent pairs are not used as independent uncertainty replicates. Three evaluation realizations are a starting descriptive assessment, not precise population confidence intervals. Scorer Monte Carlo, Leiden, and TreeTime seeds are separate. `treecluster.rng_seed` controls TreeTime's stochastic choices and is included in both the command and the tree artifact signature.

The [shared config](../shared_synthetic/config.yaml) owns `inputs`, `generation`, `simulation`, and `splits`; baseline's `experiment_config` references it and derives matched inference. EDD/EDS/ESD/ESS still distinguish genetic process assumptions. The EpiLink 0.1.5 configuration uses: `generation.genome_length=29903` controls mutation-count expectations, whereas `simulation.sequence_length=5000` controls simulated sequence sites. Tree distances divide observed Hamming counts by the latter. The dated clock is estimated, not fixed to the nominal 29903-site mutation parameter. TD is absolute and rounded to days for all scorers; TreeTime receives dates rounded by the same convention.

Input, implementation, package, parameter, seed, score, tree and membership hashes protect artifact reuse. Stages load validated prerequisites and checkpoint their outputs. Failed/missing TreeCluster results remain visible; a partial comparator sweep cannot be frozen as a complete baseline.

## Run

From the repository root after `python -m pip install -e '.[test]'`:

```bash
# Dependency checks and experiment size; no simulation.
epilink-evaluate check --config evaluation/01_synthetic_baseline/config.yaml

# Prepare or reuse the shared transmission backbone (also automatic in diagnostics).
epilink-evaluate scovmod --stage prepare --config evaluation/01_synthetic_baseline/config.yaml

# Complete development-only diagnostics on the shared experiment first.
python evaluation/00_synthetic_diagnostics/run.py --stage all

# Optional: prepare training observations and reuse diagnosed development data.
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

# Small end-to-end workflow; both stages use their separate smoke roots.
python evaluation/00_synthetic_diagnostics/run.py --smoke --stage all
python evaluation/01_synthetic_baseline/run.py --smoke --stage all
```

Baseline defaults to `develop`; diagnostics defaults to `all` and supports `prepare`, `observations`, `graphs`, `trees`, `all`, and `report`. Its `prepare` stage alone does not complete the required diagnostics. Retained outputs describe earlier workflow checkpoints; current evidence requires the diagnostics-first sequence above.

FastME and TreeTime are used for tree construction/dating, and TreeCluster for tree partitions. Tools are discovered on PATH or beside the active Python interpreter; commands can also be configured explicitly. Reports list failures. Raw/processed source inputs are preserved. `scovmod --stage prepare` and diagnostics share the same input preparation. Managed backbones are reused when the source hashes, construction settings, implementation and output checksums match. Explicit prebuilt trees without a manifest are retained.

`scovmod --stage prepare` prepares only the transmission backbone and provenance. Diagnostics prepares shared truth and development observations. Baseline `prepare` requires completed diagnostics and prepares training observations while reusing development data. The `scovmod` command defaults to `prepare` and supports only that stage.

## Development validation

Validated on 2026-10-02 with `python -m pytest -q` (**160 passed**) and the diagnostics→baseline `--smoke --stage all` sequence. The 64-case baseline run `f076584b64907b3f30ed` completed 1,541 development pairwise candidates, 132 clustering settings, and frozen replay of 23 pairwise / 47 clustering settings. All 102 criterion/pipeline operating summaries displayed their actual objective; all three endpoint frontiers and the reference-grid audit were verified. FastME, TreeTime and TreeCluster were exercised. No full study was run for this development validation.

### Observed full-run timing — 2026-10-02

A separate full configured run used the 5,051-case backbone, 5,000-nt sequences, eight scorers, 10,000 EpiLink Monte Carlo draws, training seeds 61001–61002, development seeds 62001–62003, and evaluation seeds 63101–63103.

| Stage | Observed elapsed |
| --- | ---: |
| `develop` | 2h 02m 10s |
| `select` | 1m 49s |
| `evaluate` | 1h 16m 33s |
| Complete `develop`–`evaluate` sequence | 3h 20m 38s |

Stage durations are measured from the first timestamped stage log to its report log; the combined time spans the first `develop` log through the final `evaluate` report log. Hardware and benchmark context are recorded in the [Operations runtime benchmark](../../OPERATIONS.md#observed-full-run-wall-times); these timings are one observed run, not a guarantee.

## Outputs and extension points

Column definitions, formulas, missing-value conventions, metadata fields, and analysis joins are documented in the [output reference](../../OUTPUTS.md).

`../shared_synthetic/outputs/inputs/` contains `transmission_tree.gml`, its `transmission_tree.source.json` provenance, and `manifest.json`. Full and smoke runs share this backbone; `--output` changes the run root, not the input paths. Each run's `inputs.json` records the pinned tree and its source provenance.

`../shared_synthetic/outputs/synthetic/` owns `artifacts/backbones/`, `artifacts/truth/`, and `artifacts/observations/`, with experiment manifests under `experiments/<id>/`. Baseline's `<run>/experiment.json` pins the shared `experiment_directory` and `fingerprint`; resolve observation/truth joins from that source, not baseline-local artifact paths or the latest shared pointer.

`outputs/baseline/` contains `artifacts/` for fitted models, scores, and inferred trees, and fingerprinted `runs/<id>/` directories. `current.json` points to the current run. Each run contains `development/` (pairwise curves and clustering sweeps), `selection/operating_points.json`, `evaluation/` (fixed-setting results), and `report.md`, `report.html`, `figures/`, and a run manifest. Revising criteria after accessing evaluation data requires fresh evaluation seeds in the shared config and diagnostics for the updated design. The shared `heldout_access/seed_<seed>.json` ledger survives `reset-outputs`, as do all shared synthetic outputs. Smoke outputs use `outputs/baseline_smoke/` and `../shared_synthetic/outputs/synthetic_smoke/`. The revised supplied design reserves fresh full-evaluation seeds 63101–63103; 63001–63003 belong to the previous completed comparison. Smoke retains its separate validation seeds and never substitutes for full scientific evaluation.

Each development `pairwise/evidence/` artifact checkpoints cumulative curves, rankings, budgets, calibration, and empty-selection metrics. Once every seed's evidence is available, `development/pairwise_candidates/` pins the candidate definitions and source-manifest hashes. `settings.json` includes these definitions alongside the static clustering grids; a fresh process restores them before replay.

Shared code is under `src/epilink_evaluation/`: inputs, truth, scorers, graphs, phylogeny, clusterers, metrics, selection, workflows, and reporting. Scorers do not compute evaluation metrics; clusterers do not access truth; reporting reads saved result tables. `inputs.synthetic.analysis_table` provides an optional joined view without duplicating stored truth for every model.

## Manuscript figures and tables

After completing `evaluate`, generate the standalone displays from saved results (no simulation or refitting):

```bash
python -m evaluation.results.fig02  # pairwise discrimination
python -m evaluation.results.fig03  # components
python -m evaluation.results.fig04  # binary Leiden
python -m evaluation.results.fig05  # native Leiden
python -m evaluation.results.fig06  # resolution regret
python -m evaluation.results.fig07  # raw TreeCluster
python -m evaluation.results.fig08  # dated TreeCluster
python -m evaluation.results.fig09  # graph operating bars
python -m evaluation.results.fig10  # TreeCluster operating bars
python -m evaluation.results.tab01  # main operating points
python -m evaluation.results.tab02  # full operating points
```

These scripts default to the baseline root's `current.json` and write to `evaluation/results/outputs/01_synthetic_baseline/<run-id>/`. Each accepts `--run-dir <run>` and `--output-dir <directory>` for a pinned run and destination; the figure scripts also accept `--format pdf|png|both` (default `both`). See the [numbered results index](../results/README.md) for the script-to-display mapping. They require a completed evaluation and use only the frozen `balanced_M0` (`M0_f1`) selection. The two `.tex` files use the project's `thesistablebody`/`longtable` macros from `src/epilink_evaluation/utils/latex_tables.py` (and `landscape` for wide tables).

Figure legends use short scorer labels without renaming the saved result columns: `GD_D`/`GD_S` are **GDD**/**GDS**, and `LOGIT_D`/`LOGIT_S` are **LGD**/**LGS**. Deterministic observed genetics include **EDD, ESD, GDD, LGD**; stochastic observed genetics include **EDS, ESS, GDS, LGS**. In each EpiLink code the second letter denotes inference genetics and the third observed genetics (D or S). All EpiLink and logistic scorers use temporal distance, whereas GDD/GDS use genetic distance alone. Compare scorers within an observed-process column.

| Output | Source and interpretation |
| --- | --- |
| `fig02_pairwise_discrimination.pdf` / `.png` | Tie-aware **development** M=0 PR curves, one line per observation seed; diamonds are the means at frozen development cutoffs. Separate columns compare four scorers within each observed genetic process. Legend AP values are equal-seed means from **held-out** `pairwise/rankings.csv`, independent of cutoffs. |
| `fig03_components`, `fig04_leiden_binary`, `fig05_leiden_native` (`.pdf` / `.png`) | One **development** trade-off figure per graph-clustering approach. Columns are observed genetic processes. Upper panels show M=0 precision versus recall; lower panels show M=0 F1 versus M≥3 contamination. Component lines trace native score/GD threshold sweeps. Leiden dots cover the threshold × CPM-resolution grid; lines trace non-dominated display envelopes, **not** a one-dimensional parameter sweep. Diamonds mark the frozen M=0 setting using its development metrics, even if it is off a display envelope. Native-weighted genetic Leiden is absent by design. |
| `fig06_epilink_resolution_regret.pdf` / `.png` | **Development-only** assessment of a single numerical CPM resolution across all four EpiLink variants × binary/native Leiden policies. The line is the average **difference from each pipeline's own best M=0 F1**, in percentage points. Shading covers the middle half (25th–75th percentiles) of the eight differences, **not** a confidence interval. The marked default minimizes the largest of the eight differences, then the mean difference. Candidate resolutions are the intersection of the saved policy grids (0.1–1.0 in the supplied run); native pipelines retain their full grid, including values below 0.1, when calculating their individual reference optima. `fig06_epilink_resolution_regret_by_pipeline.csv` gives each pipeline's threshold chosen anew on development data at every common resolution, reference optimum, F1 difference, and recovery/contamination/workload. `fig06_epilink_resolution_regret_summary.csv` records the eight-pipeline mean, IQR, worst difference, feasibility, and marked default. No per-seed thresholds or held-out metrics are used in the default recommendation. |
| `fig07_treecluster_raw`, `fig08_treecluster_dated` (`.pdf` / `.png`) | Separate **development** figures with the same panel layout. Each line follows thresholds for a TreeCluster method (`max_clade`, `avg_clade`, or `single_linkage`). The diamond marks the method **and** threshold jointly selected within that raw/dated observed-process pipeline. Raw genetic thresholds are substitutions/site in the sweep and dated thresholds are days; axes display recovery or contamination rather than threshold values. |
| `fig09_graph_cluster_operating_bars.pdf` / `.png` | **Held-out** M=0 precision, recall, F1, and M≥3 contamination at frozen settings. Rows are components, binary Leiden, and native Leiden; columns are observed genetic processes, with the available score variants along each x-axis. Whiskers are sample SD across evaluation realizations. |
| `fig10_treecluster_operating_bars.pdf` / `.png` | **Held-out** metrics on the same percentage scale, with raw and dated trees in separate panels and deterministic/stochastic observed genetics as grouped-bar categories. Each category identifies its development-selected TreeCluster method and threshold (SNP for raw, days for dated). The raw/dated and deterministic/stochastic comparisons are between separately selected **frozen operating pipelines**, not one method/threshold held constant. Other TreeCluster methods were swept on development data but not replayed as distinct held-out settings. |
| `tab01_operating_points.tex` | Main-text held-out comparison: for each observed process, the matched-genetics EpiLink scorer (EDD or ESS), GD, and logistic score as pairwise rules and binary Leiden inputs, plus raw and dated TreeCluster. This is an explicit presentation subset, not a held-out-data-driven choice of pipelines. |
| `tab02_operating_points_full.tex` | Supplementary held-out longtable of **all** frozen selected pipelines, including cross-assumption EpiLink, components, and native-weighted Leiden. |

All cluster panels evaluate **every within-cluster pair**, including graph nonedges. At M=0, M≥3 contamination counts only distant false positives, not all false positives. Tables report equal-realization means (sample SD) for the three evaluation observation seeds, with percentages for fractions and counts for selected pairs. TreeCluster raw cutoffs are displayed as SNP counts converted from the stored substitutions/site threshold using the pinned sequence length; dated cutoffs are days. Undefined ratios remain dashes. Pairwise rows have no cluster-size metrics. The run's `selection/operating_points.json` and `evaluation/selection_used.json` record the complete frozen definitions; the scripts verify summary and per-seed coverage against the latter. Between-seed SD is descriptive conditional on the fixed backbone, not a confidence interval across epidemics.

The regret calculation reuses the saved `balanced_M0` objective, per-realization feasibility constraints and mean/SD/stable-ID tie-breaking. Each graph threshold remains **one value shared across development realizations**; the same numerical resolution is imposed on all eight pipelines but a separate best threshold is chosen for each. The script independently reproduces the frozen individual development optima as a validation check. Equal numeric CPM resolutions act on differently scaled binary and native edge weights; this is a *compromise under the supplied grid*, not a universal physical scale. The existing baseline evaluation replayed individually frozen resolutions. A confirmatory assessment of a newly shared resolution would require fresh held-out observations rather than treating this development plot as an evaluation result.

## Next studies

After baseline evaluation, the [perturbation workflow](../02_synthetic_perturbation/README.md) replays frozen models and operating settings on paired new observations. Start with `python evaluation/02_synthetic_perturbation/run.py --smoke`, then omit `--smoke` for all configured parameter levels. Matched and baseline-fixed modes differ in EpiLink inference; logistic training stays baseline-fixed in both. Any retuning or retraining is a separate adaptation analysis.

The [Boston empirical application](../03_boston_application/README.md) applies the same frozen definitions to observed Boston outbreak data. It tests transfer and describes exposure composition, recovery, and graph/phylogenetic partition agreement. Complete transmission truth is unavailable, so the Boston summaries provide descriptive evidence rather than truth-validated operating-point selection. Boston's TN93 table is distance-censored at 0.0005/site, so missing pairs are treated as unobserved rather than zero distance.
