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

| Study                                                                | Scientific role                                                                        | Main evidence                                                       |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [**Synthetic diagnostics**](../00_synthetic_diagnostics/README.md)   | Characterize development feature ambiguity and known-truth controls before comparison. | Exact GD/GD_TD cells and endpoint-oracle graph partitions.          |
| **Synthetic baseline** (this study)                                  | Compare methods, select settings, and evaluate them on held-out observations.          | Truth-based performance and frozen models/operating points.         |
| [**Synthetic perturbation**](../02_synthetic_perturbation/README.md) | Test all four EpiLink scorers under inference mismatch and resolution adaptation.      | Paired clustering changes in the four inference/resolution modes.   |
| [**Boston application**](../03_boston_application/README.md)         | Apply frozen settings to empirical observations.                                       | Exposure concentration/recovery and graph/tree partition agreement. |

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

**Evidence:** full tie-aware precision–recall curves, AP, absolute precision, recall, F1, enrichment over prevalence, selected counts/fractions, and AD/CA/M composition. Genetic ties are not broken using time or truth. Empty selections have undefined precision and zero recall when positive pairs exist. Calibration curves, Brier score, and log loss apply to logistic probabilities. Compatibility is not treated as a calibrated probability, clipped to [0,1], or normalized into one.

Pairwise and component selection share the union of distinct score values across **development** realizations, plus an empty selection. Each candidate is one inclusive cutoff shared across seeds. Pairwise metrics come from cumulative whole-tie PR curves; component metrics come from an incremental connectivity sweep that counts newly co-clustered pairs at each merge and reuses unchanged partitions. Each pipeline independently maximizes the equal-seed mean objective, subject to constraints in every seed. Evaluation replays its own frozen cutoff.

## 3. What relationships do clusters contain?

| Approach             | Input                                       | Sweep                                                      |
| -------------------- | ------------------------------------------- | ---------------------------------------------------------- |
| Connected components | Thresholded pair-score graph                | Score or genetic threshold                                 |
| Leiden               | Full observed graph, scorer-derived weights | Resolution only                                            |
| TreeCluster, raw     | IQ-TREE maximum-likelihood tree             | Method × SNP-count cutoff, converted to substitutions/site |
| TreeCluster, dated   | IQ-TREE/LSD2-dated tree                     | Method × day threshold                                     |

All clusterers return one membership per sampled case, including isolates and distinct TreeCluster `-1` singletons. Evaluate **all within-cluster pairs**, not only retained graph edges. Report M=0 precision/recall/F1, broader endpoint recovery, M≥3 contamination, direct-transmission retention, shared-infector retention, singleton fraction, cluster-size distribution, and largest-cluster fraction. Pair metrics exclude self-pairs. ARI between methods, if added, would measure agreement rather than truth accuracy.

Leiden uses **full observed graphs with no cutoff** and selects **resolution only**. EpiLink/logistic edges use the original scores, including zeros; genetic-distance graphs use unit weights as unweighted controls. Each scorer has one `leiden/<scorer>` pipeline. Leiden's objective is explicit (default CPM); restarts are selected by that objective, never by truth metrics. `clustering.leiden.resolutions` supplies bounds and a search budget. Resolution scales depend on objective and scorer.

```yaml
resolutions:
  min: 0.01
  max: 1.0
  initial_points: 8
  budget: 28
  scale: log
```

The search includes both bounds, starts with logarithmically spaced trials, and refines promising intervals using every pipeline/criterion's development objective and per-seed constraints. Each refinement batch also explores a widest interval. All scorers share the evaluated resolution values, allowing paired resolution comparisons; their selected optima remain independent. Graph construction is reused within each batch, and completed trials are checkpointed. The budget counts distinct resolutions, each evaluated on all development seeds/scorers. `scale: linear` is also available; the optional `tolerance` (default 0.001) stops refinement of sufficiently narrow intervals in the chosen linear/log coordinate. Selection returns the best feasible evaluated value within the declared bounds, without claiming an exact global optimum.

TreeCluster uses the same bounded coarse-to-fine search for its two cutoff types:

```yaml
genetic_threshold_snps:
  min: 0
  max: 10
  initial_points: 4
  budget: 11
  scale: linear
temporal_threshold_days:
  min: 0
  max: 105
  initial_points: 8
  budget: 28
  scale: linear
```

Raw-tree trials are whole SNP counts, converted once to substitutions/site using the alignment length. Refinement selects an untested integer strictly inside an interval and stops when the budget or integer domain is exhausted; the supplied 0–10 range contains only eleven possible cutoffs. Dated-tree trials are nonnegative real days, passed directly to TreeCluster, including fractional-day refinements. Both searches start at their bounds and share trial cutoffs across the configured genetic processes and methods. Promising intervals are ranked using each process/method/criterion's feasible development objective; final selection independently chooses the best method/cutoff pair for each raw/dated pipeline. Each budget counts distinct cutoffs for that tree kind, not method or seed calls. Existing explicit cutoff lists remain supported for fixed comparisons. Phylogenies and completed partitions are reused, and frozen evaluation/Boston transfer do not discover new cutoff values.

TreeCluster compares `max_clade`, `avg_clade`, and `single_linkage`. IQ-TREE infers raw trees with the ancestral reference as outgroup; that tip is pruned before partitioning. LSD2 calibrates dated trees and excludes the undated reference. Raw branches are substitutions/site, dated branches are days, and non-finite/negative branches are rejected. Root splits and units are recorded. Temporal cutoffs denote tree branch distances, not simply sample-date spans.

## 4. How do tuning parameters change the key metrics?

Use the declared clustering grids in `config.yaml`. Inspect component cutoff curves, full-graph Leiden resolution curves, precision–recall/contamination frontiers, and cluster sizes. Broaden a grid on development data when useful settings touch its boundary; freeze the grid and selection rule before evaluation.

Reports and figures cover **M0, Mle1 and Mle2**, with endpoint suffixes on figure filenames. `frontier.csv` is endpoint-labelled and retains the explicit `*_precision_mean`/`*_recall_mean` names. Held-out tables display each criterion's actual frozen objective; `operating_summary.csv` retains all endpoints, AD0/CA00 retention, workload, cluster-size metrics and defined-value counts. The endpoint is derived from the objective, not from the criterion's name.

The preceding diagnostics study reports exact `GD` and `GD_TD` feature cells for both genetic processes and every endpoint. The minimum empirical feature-only error count is `sum_cells min(n_target, n_other)`; its rate divides by all observed pairs. It applies to decisions constant within those cells, not population performance or partition recovery. Mixed-cell target prevalence and the fraction of all targets in mixed cells have different denominators.

Endpoint-oracle graphs connect finite M≤h pairs with unit weights; components and Leiden are assessed on all within-cluster pairs, including graph nonedges. Unsampled intermediates remain in relationship truth. These controls are deduplicated by truth, endpoint, and sampled-case set across seeds/processes; full-sampling repeats are not independent controls. Neither oracle partition performance nor feature ambiguity is a universal performance ceiling.

## 5. Which operating criteria should be carried forward?

1. Fit logistic scorers on **training** realizations only.
2. Sweep and inspect methods on **development** realizations.
3. Define a common operating criterion: e.g. balanced M=0 F1, or highest recall subject to precision, contamination, or workload constraints.
4. Select method-specific settings using that criterion, then freeze them.
5. Apply those settings unchanged on **held-out evaluation** realizations.

The supplied `balanced_M0`, `balanced_Mle1`, and `balanced_Mle2` criteria maximize their respective endpoint F1 values. M=0 remains primary; broader endpoints evaluate the same M=0-trained scores. These are executable starting comparisons, not an established epidemiological recommendation. Configurable constraints use explicit `min`/`max` bounds, for example `M0_precision: {min: 0.5}`. Choose such values from the intended use and development evidence. Bounds must hold on every development realization. Among feasible settings, maximize the mean objective, then prefer lower between-realization SD, then the stable setting ID. Infeasible methods remain labeled infeasible; criteria are never silently relaxed. TreeCluster methods are displayed separately and selected jointly within each raw/dated observed-process pipeline. Pairwise thresholds and cluster settings are selected independently. Frozen files contain criteria, development evidence fingerprints, training identity and complete method definitions.

## Baseline design and provenance

Each experiment fixes one backbone from `inputs.tree_path`; its actual case count may differ from the requested component size. Read its provenance or pinned truth manifest; see [tree preparation](../../OPERATIONS.md#4-prepare-or-regenerate-the-scovmod-tree). Training, development and evaluation use distinct observation seeds. Equal-seed means, SDs and ranges describe observation variation conditional on this backbone, not independent epidemics or independent pairs. Scorer Monte Carlo, Leiden and IQ-TREE use separate configured seeds.

The [shared config](../shared_synthetic/config.yaml) owns `inputs`, `generation`, `simulation`, and `splits`; baseline derives matched inference. EDD/EDS/ESD/ESS distinguish genetic assumptions. `generation.genome_length=29903` controls mutation-count expectations, while `simulation.sequence_length=5000` controls simulated sequence sites. IQ-TREE infers branch lengths from exported sequences; SNP cutoffs are divided by the declared alignment length. LSD2 estimates the clock unless `phylogeny.clock_rate` is fixed explicitly. Scorer TD is absolute and rounded to days; tree inference retains the original, potentially fractional sampling dates.

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

Baseline defaults to `develop`; diagnostics defaults to `all` and supports `prepare`, `backbone`, `observations`, `graphs`, `all`, and `report`. Its `prepare` stage alone does not complete the required diagnostics. Complete the diagnostics-first sequence above before baseline comparison.

IQ-TREE builds raw trees and performs LSD2 dating; TreeCluster partitions them. Set `phylogeny.executable` and `treecluster.executable` to override discovery on PATH or beside the interpreter. Reports expose failures. Source inputs are preserved, and `scovmod --stage prepare` shares backbone preparation with diagnostics. Managed artifacts are reused only when inputs, settings, producer identities and checksums match; explicit prebuilt backbones without manifests are retained.

`scovmod --stage prepare` prepares only the transmission backbone and provenance. Diagnostics prepares shared truth and development observations. Baseline `prepare` requires completed diagnostics and prepares training observations while reusing development data. The `scovmod` command defaults to `prepare` and supports only that stage.

## Outputs and extension points

Column definitions, formulas, missing-value conventions, metadata fields, and analysis joins are documented in the [output reference](../../OUTPUTS.md).

`../shared_synthetic/outputs/inputs/` contains `transmission_tree.gml`, its `transmission_tree.source.json` provenance, and `manifest.json`. Full and smoke runs share this backbone; `--output` changes the run root, not the input paths. Each run's `inputs.json` records the pinned tree and its source provenance.

`../shared_synthetic/outputs/synthetic/` owns `artifacts/backbones/`, `artifacts/truth/`, and `artifacts/observations/`, with experiment manifests under `experiments/<id>/`. Baseline's `<run>/experiment.json` pins the shared `experiment_directory` and `fingerprint`; resolve observation/truth joins from that source, not baseline-local artifact paths or the latest shared pointer.

`outputs/baseline/` contains `artifacts/` for fitted models, scores, and inferred trees, and fingerprinted `runs/<id>/` directories. `current.json` points to the current run. Each run contains `development/` (pairwise curves and clustering sweeps), `selection/operating_points.json`, `evaluation/` (fixed-setting results), and `report.md`, `report.html`, `figures/`, and a run manifest. Revising criteria after accessing evaluation data requires fresh evaluation seeds in the shared config and diagnostics for the updated design. The shared `heldout_access/seed_<seed>.json` ledger survives `reset-outputs`, as do all shared synthetic outputs. Smoke outputs use `outputs/baseline_smoke/` and `../shared_synthetic/outputs/synthetic_smoke/`. The supplied full-evaluation seeds are 63101–63103. Smoke uses separate validation seeds and never substitutes for full scientific evaluation.

Each development `pairwise/evidence/` artifact checkpoints cumulative curves, rankings, calibration, and empty-selection metrics. Once every seed's evidence is available, `development/cutoff_candidates/` pins the shared pairwise/component definitions and source-manifest hashes. Component sweep evidence lives in `development/seed_<seed>/clusters/component_sweeps/<scorer>/`; one set of membership/cluster tables is retained per distinct partition, and selected settings receive ordinary `<setting-id>/` artifacts. `development/resolution_search/` saves adaptive Leiden trials; `development/treecluster_search/raw/` and `dated/` save the cutoff trials and search status in SNP/day input units. `settings.json` includes all evaluated candidates; a fresh process restores them before replay.

Shared code is under `src/epilink_evaluation/`: inputs, truth, scorers, graphs, phylogeny, clusterers, metrics, selection, workflows, and reporting. Scorers do not compute evaluation metrics; clusterers do not access truth; reporting reads saved result tables. `inputs.synthetic.analysis_table` provides an optional joined view without duplicating stored truth for every model.

No `pairwise` or scorer `thresholds` configuration block is needed. Pairwise and components share exact development-score candidates; full-graph Leiden uses the bounded adaptive resolution search.

## Manuscript figures and tables

After completing `evaluate`, generate the standalone displays from saved results (no simulation or refitting):

```bash
python -m evaluation.results.fig03  # M=0 compatibility surfaces (EpiLink ED / ES)
python -m evaluation.results.fig04  # pairwise discrimination
python -m evaluation.results.fig05  # components
python -m evaluation.results.fig06  # full-graph Leiden
python -m evaluation.results.fig07  # resolution regret
python -m evaluation.results.fig08  # raw TreeCluster
python -m evaluation.results.fig09  # dated TreeCluster
python -m evaluation.results.fig10  # graph operating bars
python -m evaluation.results.fig11  # TreeCluster operating bars
python -m evaluation.results.fig12  # compact held-out graph/tree comparison
python -m evaluation.results.tab01  # main operating points
python -m evaluation.results.tab02  # full operating points
```

These scripts default to the baseline root's `current.json` and write to `evaluation/results/outputs/01_synthetic_baseline/<run-id>/`. Each accepts `--run-dir <run>` and `--output-dir <directory>` for a pinned run and destination; the figure scripts also accept `--format pdf|png|both` (default `both`). See the [numbered results index](../results/README.md) for the script-to-display mapping. They require a completed evaluation and use only the frozen `balanced_M0` (`M0_f1`) selection. The two `.tex` files use the project's `thesistablebody`/`longtable` macros from `src/epilink_evaluation/utils/latex_tables.py` (and `landscape` for wide tables).

Figure legends use short scorer labels without renaming the saved result columns: `GD_D`/`GD_S` are **GDD**/**GDS**, and `LOGIT_D`/`LOGIT_S` are **LGD**/**LGS**. Deterministic observed genetics include **EDD, ESD, GDD, LGD**; stochastic observed genetics include **EDS, ESS, GDS, LGS**. In each EpiLink code the second letter denotes inference genetics and the third observed genetics (D or S). All EpiLink and logistic scorers use temporal distance, whereas GDD/GDS use genetic distance alone. Compare scorers within an observed-process column.

| Output                                                               | Source and interpretation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `fig03_primary_compatibility_surfaces.pdf` / `.png`                  | Primary M=0 (`AD(0)` or `CA(0,0)`) compatibility from deterministic (ED) and stochastic (ES) **EpiLink inference** applied to the same integer genetic-distance (0–15 SNP) and absolute sampling-time-difference (0–20 day) input grid. The panels do not represent different observed-genetics datasets. Both use the same color scale; scores are not calibrated probabilities. Parameters, scorer seed, and Monte Carlo sample count come from the pinned baseline run.                                                                                         |
| `fig04_pairwise_discrimination.pdf` / `.png`                         | Tie-aware **development** M=0 PR curves, one line per observation seed; diamonds are the means at frozen development cutoffs. Separate columns compare four scorers within each observed genetic process. Legend AP values are equal-seed means from **held-out** `pairwise/rankings.csv`, independent of cutoffs.                                                                                                                                                                                                                                                 |
| `fig05_components`, `fig06_leiden` (`.pdf` / `.png`)                 | Development precision/recall and F1/M≥3 contamination trade-offs by observed genetics. Components sweep cutoffs; Leiden sweeps resolution on full graphs. Diamonds mark frozen development settings.                                                                                                                                                                                                                                                                                                                                                               |
| `fig07_epilink_resolution_regret.pdf` / `.png`                       | Development-only shared-resolution assessment across four EpiLink pipelines. Compare F1 at each resolution with each scorer's best full-grid resolution; report mean/IQR/worst losses in percentage points. The marked compromise minimizes worst loss, then mean loss. No cutoff or held-out selection is used.                                                                                                                                                                                                                                                   |
| `fig08_treecluster_raw`, `fig09_treecluster_dated` (`.pdf` / `.png`) | Separate **development** figures with the same panel layout. Each line follows thresholds for a TreeCluster method (`max_clade`, `avg_clade`, or `single_linkage`). The diamond marks the method **and** threshold jointly selected within that raw/dated observed-process pipeline. Raw genetic thresholds are substitutions/site in the sweep and dated thresholds are days; axes display recovery or contamination rather than threshold values.                                                                                                                |
| `fig10_graph_cluster_operating_bars.pdf` / `.png`                    | Held-out M=0 precision, recall, F1 and M≥3 contamination at frozen settings. Rows are components and Leiden; columns separate observed genetics. Whiskers are across-seed sample SD.                                                                                                                                                                                                                                                                                                                                                                               |
| `fig11_treecluster_operating_bars.pdf` / `.png`                      | **Held-out** metrics on the same percentage scale, with raw and dated trees in separate panels and deterministic/stochastic observed genetics as grouped-bar categories. Each category identifies its development-selected TreeCluster method and threshold (SNP for raw, days for dated). The raw/dated and deterministic/stochastic comparisons are between separately selected **frozen operating pipelines**, not one method/threshold held constant. Other TreeCluster methods were swept on development data but not replayed as distinct held-out settings. |
| `fig12_cluster_recovery_contamination.pdf` / `.png`                  | Compact held-out M=0 F1 versus M≥3 contamination for every selected graph/tree pipeline, with separate observed-process panels. The same-named CSV retains the plotted means and across-seed sample SD.                                                                                                                                                                                                                                                                                                                                                            |
| `tab01_operating_points.tex`                                         | Held-out comparison of pairwise rules, full-graph Leiden for all scorers, and raw/dated TreeCluster. Each scorer has one Leiden pipeline.                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `tab02_operating_points_full.tex`                                    | Supplementary held-out longtable of all selected pipelines, including all four EpiLink scorers, components, full-graph Leiden and TreeCluster.                                                                                                                                                                                                                                                                                                                                                                                                                     |

All cluster panels evaluate **every within-cluster pair**, including graph nonedges. At M=0, M≥3 contamination counts only distant false positives, not all false positives. Tables report equal-realization means (sample SD) for the three evaluation observation seeds, with percentages for fractions and counts for selected pairs. TreeCluster raw cutoffs are displayed as SNP counts converted from the stored substitutions/site threshold using the pinned sequence length; dated cutoffs are days. Undefined ratios remain dashes. Pairwise rows have no cluster-size metrics. The run's `selection/operating_points.json` and `evaluation/selection_used.json` record the complete frozen definitions; the scripts verify summary and per-seed coverage against the latter. Between-seed SD is descriptive conditional on the fixed backbone, not a confidence interval across epidemics.

The regret calculation reuses `balanced_M0`, per-realization feasibility constraints and mean/SD/stable-ID tie-breaking, and reproduces individual development optima. It proposes a shared resolution across the four EpiLink pipelines; baseline evaluation still replays their individually selected resolutions. A new shared-resolution evaluation requires fresh held-out observations.

## Next studies

After baseline evaluation, the [perturbation workflow](../02_synthetic_perturbation/README.md) runs EpiLink clustering only: baseline/matched inference × baseline/updated full-graph resolution. Updated resolutions use fresh development seeds; all four arms evaluate paired separate seeds. Start with `python evaluation/02_synthetic_perturbation/run.py --smoke`, then omit `--smoke` for the full study.

The [Boston empirical application](../03_boston_application/README.md) applies the same frozen definitions to observed Boston outbreak data. It tests transfer and describes exposure composition, recovery, and graph/phylogenetic partition agreement. Complete transmission truth is unavailable, so the Boston summaries provide descriptive evidence rather than truth-validated operating-point selection. Boston's TN93 table is distance-censored at 0.0005/site, so missing pairs are treated as unobserved rather than zero distance.
