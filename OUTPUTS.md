# Output reference

Column definitions and JSON fields for the `epilink_evaluation` studies. For commands, directory layout, and which figures to inspect, use the [operational guide](OPERATIONS.md#7-find-and-interpret-results). Scientific interpretation is defined in the [diagnostics protocol](evaluation/00_synthetic_diagnostics/README.md), [baseline protocol](evaluation/01_synthetic_baseline/README.md), [perturbation guide](evaluation/02_synthetic_perturbation/README.md), and [Boston guide](evaluation/03_boston_application/README.md).

Paths below use `<root>` for the configured output directory, `<run>` for `<root>/runs/<run-id>`, and `<split>` for `development` or `evaluation`. Sections 1–9 describe synthetic-baseline outputs; perturbation reuses their metric schemas as described in section 12. Boston's empirical schemas are in section 10. Boston has descriptive exposure summaries rather than synthetic truth metrics, observation seeds, or train/development/evaluation splits. Section 13 covers the shared synthetic experiment and diagnostics. For baseline, `<shared-root>` is resolved from its run's `experiment.json`; it defaults to `evaluation/shared_synthetic/outputs/synthetic/` (or `synthetic_smoke/`).

Prepared inputs are shared outside the run roots:

| Directory                                          | Files                                                                     |
| -------------------------------------------------- | ------------------------------------------------------------------------- |
| `evaluation/shared_synthetic/outputs/inputs/`      | `transmission_tree.gml`, `transmission_tree.source.json`, `manifest.json` |
| `evaluation/03_boston_application/outputs/inputs/` | `cases.parquet`, `observed_pairs.parquet`, `manifest.json`                |

SCoVMod preparation and diagnostics use the shared input location; baseline reads the experiment's immutable backbone copy. Boston uses its own prepared inputs. Changing study `--output` leaves shared input/experiment paths unchanged; `--smoke` suffixes study and shared experiment roots, but leaves prepared input paths unchanged. Each baseline or Boston run's `inputs.json` records its input provenance.

## Contents

1. [Conventions and identifiers](#1-conventions-and-identifiers)
2. [Shared pair-metric columns](#2-shared-pair-metric-columns)
3. [Per-seed pairwise tables](#3-per-seed-pairwise-tables)
4. [Partition and cluster outputs](#4-partition-and-cluster-outputs)
5. [Aggregated and held-out results](#5-aggregated-and-held-out-results)
6. [Truth, observations, scores, and fitted models](#6-truth-observations-scores-and-fitted-models)
7. [Setting definitions and frozen decisions](#7-setting-definitions-and-frozen-decisions)
8. [Manifests, status, and provenance](#8-manifests-status-and-provenance)
9. [Phylogenetic artifacts](#9-phylogenetic-artifacts)
10. [Boston inputs and results](#10-boston-inputs-and-results)
11. [Worked joins in Python](#11-worked-joins-in-python)
12. [Perturbation study outputs](#12-perturbation-study-outputs)
13. [Shared experiments and diagnostics](#13-shared-experiments-and-diagnostics)

## 1. Conventions and identifiers

Column names are case-sensitive: `M`, `M0_AP`, `GD_D`, and `TD` retain their capitalization. Counts are nonnegative integers conceptually; CSV concatenation with missing values can represent count columns as floating point. Identifiers should be treated as strings where noted, even if they happen to contain only digits. Each table's row unit is specified below.

| Identifier     | Meaning and scope                                                                                                                        |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `split`        | `development` or `evaluation` in result tables. Training observations live in artifacts.                                                 |
| `seed`         | Observation-realization seed; it is not an algorithm seed.                                                                               |
| `score_name`   | EDD, EDS, ESD, ESS, GD_D, GD_S, LOGIT_D, or LOGIT_S.                                                                                     |
| `data_process` | Observed genetic process: `deterministic` or `stochastic`.                                                                               |
| `score_family` | `epilink`, `genetic`, or `logistic`; present in ranking summaries.                                                                       |
| `pipeline`     | Comparison family, such as `pairwise/EDD`, `components/GD_D`, `leiden/ESS`, or `treecluster/stochastic/dated`.                           |
| `setting_id`   | 20-character hash of the complete method definition. Join to this run's `settings.json`; the same definition can occur in multiple runs. |
| `criterion`    | Name of a selection rule, such as `balanced_M0`; added to held-out operating results.                                                    |
| `case_id`      | String case identifier; meaningful within the selected backbone/dataset.                                                                 |
| `pair_id`      | Zero-based unordered-pair index in a particular full-tree truth artifact. It is not a globally unique ID across trees.                   |
| `cluster_id`   | Nonnegative partition-local integer. The same number in another setting or seed does not identify the same cluster.                      |

Within one run, a per-setting result row is keyed by `(split, seed, pipeline, setting_id)`. Operating results additionally include `criterion`. Across runs, retain the run ID as part of the key.

**Missing-value conventions:**

- CSV uses empty cells for missing values; pandas normally reads these as NaN.
- Parquet retains nullable values. Inactive relationship counts are null, rather than zero: zero is a meaningful number of intermediates.
- JSON serialization converts non-finite numeric values to `null`. A missing field can also mean it does not apply to that artifact or method.
- Markdown reports display missing numbers as `undefined`. HTML may display NaN.
- Structural blanks arise when pairwise and cluster rows are combined, or when calibration metrics apply only to logistic scorers.
- JSON `threshold: null` with `empty: true` explicitly defines an empty selection.

An absent file generally means its stage has not produced it. Some intentionally empty CSVs have no header and raise `pandas.errors.EmptyDataError`; for example, `calibration.csv` is empty when no logistic scorers are included.

## 2. Shared pair-metric columns

These fields appear in pairwise threshold/curve tables and partition metrics. They all use the **observed unordered-pair universe**, excluding self-pairs. For clustering, the selected set consists of **all within-cluster pairs**, which can be larger than the retained graph-edge set.

Let:

- `U` = number of observed unordered pairs, `n_cases * (n_cases - 1) / 2`;
- `S` = number of selected pairs;
- `P_h` = number of true target pairs in the whole observed universe at horizon `h`;
- `TP_h` = number of selected pairs satisfying that target.

### Endpoint and relationship suffixes

`<endpoint>` expands to each of the following three literal prefixes:

| Prefix | Target                                                       |
| ------ | ------------------------------------------------------------ |
| `M0`   | M = 0: direct transmission AD(0) or shared infector CA(0,0). |
| `Mle1` | Finite M <= 1.                                               |
| `Mle2` | Finite M <= 2.                                               |

`<category>` expands to these ten literal suffixes, which partition the pair universe. CA branch order is interchangeable for categorization.

| Category   | Relationship                                                                      |
| ---------- | --------------------------------------------------------------------------------- |
| `AD0`      | Direct ancestor–descendant transmission; zero intermediates.                      |
| `AD1`      | Ancestor–descendant with one intermediate.                                        |
| `AD2`      | Ancestor–descendant with two intermediates.                                       |
| `ADge3`    | Ancestor–descendant with at least three intermediates.                            |
| `CA00`     | Two cases with the same infector.                                                 |
| `CA01`     | Shared ancestor with branch counts (0,1) or (1,0).                                |
| `CA02`     | Branch counts (0,2) or (2,0).                                                     |
| `CA11`     | Branch counts (1,1).                                                              |
| `CAge3`    | Shared ancestor with total M >= 3.                                                |
| `separate` | Different introduction components; M is undefined and all endpoints are negative. |

### Metric definitions

All fractions, precision, recall, and F1 values are dimensionless in `[0, 1]` when defined. Enrichment is a dimensionless ratio that can exceed one.

| Column or column pattern    | Definition                                                 | Undefined when            |
| --------------------------- | ---------------------------------------------------------- | ------------------------- |
| `selected_pairs`            | `S`, a count.                                              | Never for a valid result. |
| `selected_fraction`         | `S / U`.                                                   | `U = 0`.                  |
| `<endpoint>_precision`      | `TP_h / S`.                                                | `S = 0`.                  |
| `<endpoint>_recall`         | `TP_h / P_h`.                                              | `P_h = 0`.                |
| `<endpoint>_f1`             | `2 * TP_h / (S + P_h)`.                                    | `S + P_h = 0`.            |
| `<endpoint>_enrichment`     | `(TP_h / S) / (P_h / U)`, precision divided by prevalence. | `S = 0` or `P_h = 0`.     |
| `n_<category>`              | Count of selected pairs in that relationship category.     | Never for a valid result. |
| `fraction_<category>`       | `n_<category> / S`.                                        | `S = 0`.                  |
| `Mge3_contamination`        | `(n_ADge3 + n_CAge3) / S`.                                 | `S = 0`.                  |
| `separate_fraction`         | `n_separate / S`; also available as `fraction_separate`.   | `S = 0`.                  |
| `direct_edge_retention`     | Selected AD0 pairs divided by all observed AD0 pairs.      | No observed AD0 pairs.    |
| `shared_infector_retention` | Selected CA00 pairs divided by all observed CA00 pairs.    | No observed CA00 pairs.   |

For example, `M0_precision` uses `TP_0 = n_AD0 + n_CA00`. `Mle1` also includes `n_AD1 + n_CA01`; `Mle2` additionally includes `n_AD2 + n_CA02 + n_CA11`. All ten `n_<category>` counts sum to `selected_pairs`.

An empty selection with existing positives has precision undefined, recall zero, and F1 zero. If no positives exist but some pairs are selected, precision and F1 are zero while recall is undefined. For M=0, the false-positive fraction is `1 - M0_precision`; M>=3 contamination counts only distant relationships and excludes separate introductions. Direct-edge retention covers observed endpoints, not every transmission edge involving unsampled cases.

## 3. Per-seed pairwise tables

Directory: `<run>/<split>/seed_<seed>/pairwise/`. The common scorer identifiers are `split`, `seed`, `score_name`, `data_process`.

### `metrics.csv`

**One row per scorer threshold setting.** Columns are the four scorer identifiers, `pipeline`, `setting_id`, and every shared metric in section 2. Development always covers the union of distinct development score cutoffs for each scorer plus an empty selection. Every cutoff is applied to every development seed, retaining whole ties. Evaluation covers selected pairwise settings only; it does not discover new candidate cutoffs. Look up the actual threshold and its direction in `settings.json` and the scorer definitions; there is no `threshold` column in this file.

### `rankings.csv`

**One row per configured scorer**, in both development and evaluation. Ranking and calibration summaries use the whole pair universe, independently of the selected operating threshold.

| Column                                        | Definition                                                                                                                                                                                                               |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `split`, `seed`, `score_name`, `data_process` | Scorer identifiers.                                                                                                                                                                                                      |
| `score_family`                                | Scorer family from section 1.                                                                                                                                                                                            |
| `n_pairs`                                     | `U`, the total candidate-pair count.                                                                                                                                                                                     |
| `unique_scores`                               | Number of distinct observed scores/distances.                                                                                                                                                                            |
| `<endpoint>_AP`                               | Tie-aware average precision: sum of each recall increment times precision after admitting the entire tied group. Scores are ranked descending, genetic distances ascending. Undefined if that endpoint has no positives. |
| `<endpoint>_prevalence`                       | `P_h / U`.                                                                                                                                                                                                               |
| `brier_score`                                 | Logistic scorers only: mean `(predicted_probability - M0_indicator)²`; lower is better.                                                                                                                                  |
| `log_loss`                                    | Logistic scorers only: target/non-target cross-entropy for M0, using natural logarithms and scikit-learn's probability clipping; lower is better.                                                                        |

The six endpoint columns expand to `M0_AP`, `M0_prevalence`, `Mle1_AP`, `Mle1_prevalence`, `Mle2_AP`, and `Mle2_prevalence`. Calibration columns are blank for non-logistic scorers and may be absent entirely when none are configured.

### `precision_recall.parquet`

**One row per distinct observed value for each scorer**, saved for development. The table contains the four scorer identifiers, every shared metric, and:

| Column              | Definition                                                                                                              |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `threshold`         | Observed score or genetic distance at this point; includes the whole tie. EpiLink/logistic use `>=`, genetic uses `<=`. |
| `ties_at_threshold` | Number of pairs exactly equal to this value, added at this step.                                                        |

Rows run from strict to permissive within each scorer. There is no synthetic zero-selection row; its recall of zero is implicit in the AP calculation. This curve uses unique observed values rather than the configured operating grid. The evaluation file is an empty table because full curves are not saved there.

Development `pairwise/evidence/` checkpoints these curves and the ranking and calibration tables before candidate enumeration, together with `empty_metrics.json` and its artifact manifest. The top-level pairwise tables remain the analysis interface. `development/pairwise_candidates/definitions.json` maps IDs to pairwise definitions; its manifest binds the run and all source evidence-manifest hashes. `settings.json` combines these candidates with the independently configured clustering settings. Cached candidates are restored on resumption and before evaluation replay.

### `calibration.csv`

**One row per probability bin per logistic scorer**, with the four scorer identifiers and the following fields. Both development and evaluation use 10 equal-width bins for the M0-trained classifier.

| Column                   | Definition                                                                                         |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| `bin_lower`, `bin_upper` | Probability interval boundaries; lower inclusive, upper exclusive, except the last bin includes 1. |
| `n_pairs`                | Count in this bin.                                                                                 |
| `mean_probability`       | Mean predicted probability in this bin; undefined for an empty bin.                                |
| `observed_fraction`      | Fraction of this bin's pairs with M=0; undefined for an empty bin.                                 |

## 4. Partition and cluster outputs

Directory: `<run>/<split>/seed_<seed>/clusters/`.

### `metrics.csv` and `<setting-id>/metrics.json`

**One row/object per partition.** The CSV includes identifiers `split`, `seed`, `setting_id`, `pipeline`, `data_process`. The JSON contains metrics only. Both contain every shared metric from section 2, with all within-cluster pairs as the selected set, plus:

| Column                      | Definition                                                                                                                      |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `n_cases`                   | Number of observed cases.                                                                                                       |
| `n_clusters`                | Number of clusters, including singletons.                                                                                       |
| `n_singletons`              | Number of singleton clusters, also the number of singleton cases.                                                               |
| `singleton_fraction`        | `n_singletons / n_cases`, the fraction of cases that are singletons.                                                            |
| `largest_cluster`           | Largest cluster's case count.                                                                                                   |
| `largest_cluster_fraction`  | `largest_cluster / n_cases`.                                                                                                    |
| `within_pairs`              | Sum of `size * (size - 1) / 2` across clusters; equals `selected_pairs`.                                                        |
| `cluster_mean_M0_precision` | Unweighted mean of cluster-specific M0 precision over clusters with at least two cases; undefined for all-singleton partitions. |

Global `M0_precision` weights clusters by their pair counts. `cluster_mean_M0_precision` gives each non-singleton cluster equal weight, so the two quantities generally differ.

### `<setting-id>/memberships.parquet`

**One row per observed case**, including isolates and TreeCluster singletons:

| Column       | Definition                                                                                              |
| ------------ | ------------------------------------------------------------------------------------------------------- |
| `case_id`    | Observed case ID; join to this realization's `cases.parquet`.                                           |
| `cluster_id` | Canonical partition-local integer label. Each TreeCluster `-1` case becomes a distinct singleton label. |

### `<setting-id>/clusters.parquet`

**One row per cluster**:

| Column or pattern      | Definition                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| `cluster_id`           | Key joining this partition's membership table.                                                          |
| `n_cases`              | Cases in this cluster.                                                                                  |
| `within_pairs`         | `n_cases * (n_cases - 1) / 2`.                                                                          |
| `n_<category>`         | Within-cluster count for each of the ten categories in section 2.                                       |
| `<endpoint>_precision` | Within-cluster target pairs divided by this cluster's `within_pairs`; undefined for singleton clusters. |

This cluster-level file has counts and precision, without cluster-specific recall/F1 or category fractions. Its counts sum to the partition totals.

### `<setting-id>/algorithm.json`

| Method             | Recorded fields                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Components         | `retained_graph_edges`: number of thresholded graph edges.                                                                                                                                                                                                                                                                                                                        |
| Leiden             | `objective`: CPM/modularity; `quality`: chosen igraph objective value; `restart_qualities`: objective values from all restarts; `seed`: base algorithm seed; `retained_graph_edges`: graph edge count.                                                                                                                                                                            |
| Empty-graph Leiden | `objective`, `quality: 0.0`, `retained_graph_edges: 0`; no restarts are run, so restart fields are absent.                                                                                                                                                                                                                                                                        |
| TreeCluster        | `command`: executed argument list; `executable`: path and SHA-256; `threshold_tree_units`: actual cutoff, in substitutions/site for raw trees or calendar years for dated trees.                                                                                                                                                                                                  |
|                    | **Threshold scaling**: SNP counts from the configuration grid are converted to substitutions/site using `threshold_value = snp_count / alignment_length` (raw) or passed directly as days (dated). The `alignment_length` is set in the `simulation` config block (default: `sequence_length`). Day-unit thresholds require no conversion and are passed directly to TreeCluster. |

`quality` is the optimized clustering objective, not a truth metric. For dated TreeCluster, thresholds are in **days** (not years); raw tree thresholds are in **substitutions per site**.

## 4.2 Raw trees

IQ-TREE builds the **raw tree** from the sampled FASTA alignment using the substitution model specified in the `phylogeny` config block (default: JC). The raw tree includes the **reference tip**; for TreeCluster compatibility the reference tip is pruned producing `raw_pruned.nwk` (substitutions per site). Branch lengths are finite and non-negative validated post-inference. The raw tree captures overall genetic divergence and is the input for TreeCluster `raw` clustering.

## 4.3 Dated trees

IQ-TREE builds the **dated tree** with `--dated=True` and calendar dates from `sampling_dates.tsv`. LSD2 estimates branch lengths in **calendar days** from the earliest sample. Dated tree lengths are **days** — thresholds are passed directly (no `/365` conversion). The dated tree excludes the reference tip for TreeCluster `dated` clustering and is the input for date-dependent downstream analysis.

## 5. Aggregated and held-out results

### `<run>/<split>/metrics.csv`

Concatenates that split's available per-seed pairwise and clustering metrics. The schema is their column union; `score_name` is populated for pairwise rows, while cluster rows identify their scorer through `settings.json`. Cluster-only metrics are blank for pairwise rows. Partial runs can contain partial coverage.

### `<run>/<split>/summary.csv`

**One row per `(pipeline, setting_id)`.** Every numeric metric in the combined table, excluding `seed`, expands into these four columns:

| Suffix                         | Meaning                                                                            |
| ------------------------------ | ---------------------------------------------------------------------------------- |
| `<metric>_mean`                | Arithmetic mean over nonmissing realization values, with equal realization weight. |
| `<metric>_std`                 | Sample SD (`ddof=1`); undefined with fewer than two nonmissing values.             |
| `<metric>_min`, `<metric>_max` | Minimum and maximum nonmissing values.                                             |

`n_realizations` counts available result rows for the setting. It does not count nonmissing observations for each metric separately; missing metrics are skipped by aggregation. A partial run can therefore have fewer rows than configured seeds. All-missing metrics stay missing. String scorer/process fields are omitted; recover them from setting definitions. Counts also receive these suffixes.

### `<run>/<split>/frontier.csv`

A subset of `summary.csv`, retaining non-dominated mean precision/recall settings within each `(endpoint, pipeline)`, including exact ties. The additional `endpoint` column is `M0`, `Mle1`, or `Mle2`; **all summary column names retain their `_mean` suffixes**. One setting can appear at multiple endpoints. Settings with missing precision or recall for that endpoint are omitted. Evaluation's frontier covers only its evaluated settings; it is not another parameter sweep. Earlier reports used an M0-only schema with renamed columns; rerendering a report updates this derived table from its saved summary.

### `evaluation/operating_results.csv`

Joins evaluation metric rows to their selected `criterion` names. Columns are the union described above plus `criterion`. If two criteria chose the same setting, its results appear once for each criterion. Group with the criterion included to avoid unintended duplication.

### `evaluation/operating_summary.csv`

**One row per `(criterion, pipeline)`**, generated during reporting from the frozen mapping in `evaluation/selection_used.json`. It includes `setting_id`, `objective`, `objective_endpoint` and `n_realizations`. The endpoint is derived from the objective metric (blank for objectives without an endpoint), never guessed from the criterion name. Each group must contain one frozen setting and one row per seed.

For all three endpoints, precision, recall, F1 and enrichment receive `_mean`, `_std`, `_min`, `_max`, and `_count` suffixes. The same summaries are included for distant/separate contamination, direct-edge/shared-infector retention, selected pair counts/fractions, singleton fraction, largest-cluster fraction and number of clusters when present. `_count` is the number of defined values for that metric; `n_realizations` counts seeds even when a metric is undefined. `_std` uses `ddof=1`. Cluster-size metrics are undefined for pairwise pipelines.

`objective_mean`, `objective_std`, `objective_min`, `objective_max`, and `objective_count` alias the corresponding summaries of the actual optimized metric, including arbitrary supported objectives. The report displays the optimized endpoint's precision/recall/F1 prominently; the CSV retains every endpoint for cross-endpoint comparisons.

## 6. Truth, observations, scores, and fitted models

Baseline truth and observations live under `<shared-root>/artifacts/`; models, scores, and inferred trees live under the baseline `<root>/artifacts/`. A hash-named directory identifies a particular set of inputs and parameters; use the run's `experiment.json` and artifact manifests to establish lineage. Perturbation keeps its own truth and observation artifacts (section 12).

### `truth/<id>/nodes.parquet`

**One row per full-backbone node** (or per node of the smoke backbone):

| Column       | Definition                                            |
| ------------ | ----------------------------------------------------- |
| `node_index` | Zero-based position in the truth node order.          |
| `case_id`    | Original node identifier; GML-loaded IDs are strings. |

### `truth/<id>/relationships.parquet`

**One row per full-backbone unordered pair**, including unobserved cases:

| Column             | Definition                                                                                                           |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `pair_id`          | Position in the upper triangle of full-tree node order.                                                              |
| `node_a`, `node_b` | Full-tree node indices, `node_a < node_b`; join to `nodes.parquet`.                                                  |
| `AD`               | 1 for ancestor–descendant pairs, otherwise 0.                                                                        |
| `CA`               | 1 for same-component pairs sharing an ancestor with neither case ancestral to the other, otherwise 0.                |
| `m`                | Number of intermediates on an AD path; null outside AD.                                                              |
| `m1`, `m2`         | Intermediates on the shared-ancestor branches to `node_a`, `node_b`, excluding the shared ancestor; null outside CA. |
| `tree_hops`        | Undirected edge distance: `m + 1` for AD, `m1 + m2 + 2` for CA; null across components.                              |
| `M`                | `m` for AD, `m1 + m2` for CA; null across components.                                                                |

With `N` full-tree nodes and indices `i < j`, `pair_id = i * (2*N - i - 1) // 2 + j - i - 1`. For separate introductions, both flags are zero and all relationship counts are null. `m1` follows node order, not an assumed shorter/longer branch.

### `observations/<id>/cases.parquet`

**One row per sampled case**, in canonical sampled-case order:

| Column          | Definition                                                                       |
| --------------- | -------------------------------------------------------------------------------- |
| `case_id`       | String case identifier.                                                          |
| `node_index`    | Full-tree index linking to truth nodes.                                          |
| `sample_date`   | Simulated sampling time, in days on the simulation time axis; can be fractional. |
| `exposure_date` | Simulated infection/exposure time on the same day axis.                          |

These times are numeric simulation days. IQ-TREE/LSD2 uses the original sample dates, including fractional days; scorer TD is rounded separately. See section 9 for date origins. Every row is sampled; there is no `sampled` column.

### `observations/<id>/pairs.parquet`

**One row per sampled unordered pair**, sorted by `(a, b)`:

| Column             | Definition                                                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `pair_id`          | Full-tree pair index; may have gaps after subsampling.                                                                          |
| `a`, `b`           | Zero-based row positions in this realization's `cases.parquet`, with `a < b`. These differ from `node_index` after subsampling. |
| `TD`               | Absolute sampling-time difference rounded to whole days.                                                                        |
| `GD_deterministic` | Simulated deterministic-genome Hamming distance, in substitutions.                                                              |
| `GD_stochastic`    | Simulated stochastic-genome Hamming distance, in substitutions.                                                                 |

### `scores/<id>/scores.parquet`

**One row per observed pair**, containing `pair_id` and one column per configured scorer. All saved values are finite. The full configuration produces:

| Column               | Observed process / interpretation                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| `EDD`                | Deterministic observations; deterministic EpiLink inference.                                      |
| `EDS`                | Stochastic observations; deterministic EpiLink inference.                                         |
| `ESD`                | Deterministic observations; stochastic EpiLink inference.                                         |
| `ESS`                | Stochastic observations; stochastic EpiLink inference.                                            |
| `GD_D`, `GD_S`       | Deterministic/stochastic Hamming distances; lower is better.                                      |
| `LOGIT_D`, `LOGIT_S` | M0 probabilities from training-fitted deterministic/stochastic logistic models; higher is better. |

EpiLink columns contain raw M0 compatibility, higher is better; they are not calibrated probabilities and can exceed one. Match the score artifact to its observation artifact before joining on `pair_id`.

### `models/<id>/models.json`

Two model objects, keyed by `deterministic` and `stochastic`:

| Field                 | Definition                                                                                 |
| --------------------- | ------------------------------------------------------------------------------------------ |
| `features`            | Ordered list `["GD", "TD"]`.                                                               |
| `target`              | `M0`; the same fitted score is evaluated against broader endpoints.                        |
| `mean`, `scale`       | Two-element training-only feature standardization arrays; a constant feature uses scale 1. |
| `coef`, `intercept`   | Logistic coefficients on standardized features and scalar intercept.                       |
| `C`                   | Inverse regularization strength used in fitting.                                           |
| `training_pairs`      | Total count-weighted training pairs across realizations.                                   |
| `training_prevalence` | M0 fraction among those training pairs.                                                    |

Prediction is `sigmoid(((x - mean) / scale) @ coef + intercept)`, with `x = [GD, TD]`. The model artifact manifest records training dataset identities and seeds. Its directory uses the first 20 characters of the training fingerprint.

## 7. Setting definitions and frozen decisions

### `<run>/settings.json`

A mapping from `setting_id` to method definition. Applicable fields are:

| Field                        | Applies to / meaning                                                                                                                                            |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`                       | `pairwise`, `components`, `leiden`, or `treecluster`.                                                                                                           |
| `pipeline`, `data_process`   | Comparison group and observed genetic process.                                                                                                                  |
| `score_name`                 | Pairwise and graph methods: scorer identifier.                                                                                                                  |
| `threshold`                  | Pairwise/components: scorer cutoff; raw TreeCluster: substitutions/site; dated TreeCluster: days. Null for explicit empty selections and for full-graph Leiden. |
| `empty`                      | Pairwise/graph: whether to select no pairs regardless of values.                                                                                                |
| `graph_mode`                 | Leiden: `full`, retaining every observed pair. EpiLink/logistic edges use scores, genetic-distance edges use unit weights.                                      |
| `objective`, `resolution`    | Leiden objective and its resolution parameter.                                                                                                                  |
| `restarts`, `algorithm_seed` | Leiden restart count and base seed.                                                                                                                             |
| `tree_kind`                  | TreeCluster: `raw` or `dated`.                                                                                                                                  |
| `method`                     | TreeCluster: `max_clade`, `avg_clade`, or `single_linkage`.                                                                                                     |
| `threshold_units`            | TreeCluster input grid units: `snps` for raw trees or `days` for dated trees. Raw `threshold` is normalized to substitutions/site before execution.             |

Pipeline names are `pairwise/<score>`, `components/<score>`, `leiden/<score>`, and `treecluster/<data-process>/<raw-or-dated>`. Each scorer has one Leiden pipeline. TreeCluster's method is selected within a raw/dated pipeline.

### `selection/operating_points.json`

| Root field                    | Definition                                                                                   |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
| `run_fingerprint`             | Full SHA-256 fingerprint of the experiment signature.                                        |
| `training_fingerprint`        | Full fingerprint identifying fitted logistic models.                                         |
| `development_evidence_sha256` | File hash of `development/metrics.csv` at selection.                                         |
| `criteria`                    | Rule objects with `name`, maximized `objective`, and metric `constraints` using `min`/`max`. |
| `operating_points`            | One decision per pipeline and criterion.                                                     |

Each operating-point object contains:

| Field                        | Definition                                                                                                                         |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `pipeline`, `criterion`      | Selected comparison group and rule name.                                                                                           |
| `rule`                       | Complete rule object used for this decision.                                                                                       |
| `status`                     | `selected` or `infeasible`.                                                                                                        |
| `setting_id`                 | Chosen definition key; null when infeasible.                                                                                       |
| `development_objective_mean` | Equal-realization objective mean for the selected setting.                                                                         |
| `development_objective_sd`   | Population SD (`ddof=0`) used to break selection ties; zero with one realization. This differs from the sample SD in summary CSVs. |
| `definition`                 | Complete chosen method definition.                                                                                                 |

The final three fields are absent for infeasible decisions. Replaced decisions before held-out access are preserved as `operating_points_<fingerprint>.json`.

`evaluation/selection_used.json` copies the frozen selection used for replay. `evaluation/heldout_access.json` contains `seeds` and `selection_fingerprint`; the latter hashes the complete frozen selection document. The access record is written before scoring evaluation observations, so it can exist after a failed evaluation attempt. The durable shared ledger is `<shared-root>/heldout_access/seed_<seed>.json`, written before evaluation observation generation with `seed`, `experiment` identity, and `selection_fingerprint`. It survives study resets and enforces fresh evaluation seeds for revised analyses across replacement runs.

## 8. Manifests, status, and provenance

### Pointer and run manifest

`<root>/current.json` contains `run_directory` (absolute path) and `fingerprint` (full run fingerprint). It points to the most recently initialized run in that root, including incomplete runs. The shared experiment pointer instead contains `experiment_directory` and `fingerprint`; see section 13. Baseline's `<run>/experiment.json` pins that identity, and `<run>/diagnostics.json` records the validated completion reference.

`<run>/manifest.json` fields:

| Field             | Definition                                                                                                                        |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `status`          | `running`, `complete`, `partial`, or `failed` for the last requested computational stage.                                         |
| `requested_stage` | Stage responsible for this manifest; rebuilding a report does not change it.                                                      |
| `signature`       | `schema` version, scientific `config`, `implementation`, `tools`, truth artifact ID in `truth`, and shared `experiment` identity. |
| `git_revision`    | Git HEAD at execution; null if unavailable. Implementation hashes capture working-copy code.                                      |
| `config`          | Full resolved configuration, including selection rules and paths.                                                                 |
| `run_directory`   | Absolute path to the run.                                                                                                         |
| `error`           | Exception representation when a caught stage exception marks the run failed.                                                      |

`implementation` records relative-file SHA-256 mappings for the evaluation and EpiLink packages plus recorded dependency `versions`. Each tool identity contains `path` and `sha256`, or an `unavailable` explanation if discovery failed. The [operational guide](OPERATIONS.md#9-resume-work-and-understand-caching) explains which changes alter the run ID.

### Completed artifact manifests

Artifact directories, pairwise stages, and individual clustering settings use the common completion fields below:

| Field          | Definition                                                                                                                            |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `status`       | `complete` for a successfully written artifact.                                                                                       |
| `signature`    | Inputs and parameters defining this artifact; see lineage below.                                                                      |
| `fingerprint`  | SHA-256 of the serialized signature. Directory IDs usually use a 20-character prefix; cluster directories use the setting ID instead. |
| `files`        | Mapping of relative filename to content SHA-256 for integrity checks.                                                                 |
| `completed_at` | UTC ISO-8601 completion timestamp.                                                                                                    |

Common signature fields and artifact-specific metadata:

| Artifact               | Signature / additional metadata                                                                                                                                                                                                                                                                                                          |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Truth                  | `kind`, `source_sha256`, `nodes`, `edges`, `implementation`; root metadata `n_cases`, `n_pairs`.                                                                                                                                                                                                                                         |
| Observations           | `kind`, `truth` artifact ID, `generation`, `simulation`, `seed`, `implementation`; root metadata `truth_directory`, `n_cases`, `n_pairs`, `units`, `fasta`, `reference_fasta`, `dates_tsv`; files: `pairs.parquet`, `cases.parquet`, `sampled_deterministic.fasta`, `sampled_stochastic.fasta`, `reference.fasta`, `sampling_dates.tsv`. |
| Fitted models          | `kind`, training `datasets` IDs, `C`, `implementation`; root metadata `seeds`.                                                                                                                                                                                                                                                           |
| Scores                 | `kind`, observation `dataset` ID, full `training` fingerprint, `scorers` metadata, `inference`, `scorer_config`, `implementation`.                                                                                                                                                                                                       |
| Per-seed pairwise      | `run` fingerprint, `score_id`, `split`, `seed`, `definitions` of the evaluated settings.                                                                                                                                                                                                                                                 |
| Per-setting clustering | `run` fingerprint, `score_id`, `definition`, `split`, `seed`.                                                                                                                                                                                                                                                                            |
| Raw/dated trees        | Dataset/process/tool and tree-building parameters; detailed in section 9.                                                                                                                                                                                                                                                                |

Observation `units` has keys `GD`, `TD`, `mutation_count_genome_length`, and `simulated_sequence_length`. Scorer metadata contains `name`, `family`, `data_process`, `inference_process`, `target`, and `higher_is_better`. Genetic scorers record `target: "none"`; the evaluation endpoints are applied separately.

### Clustering status and failures

Each seed's `clusters/status.json` contains:

| Field                 | Definition                                                                                                               |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `status`              | `complete` or `partial`.                                                                                                 |
| `configured`          | Number of settings requested for this split/seed.                                                                        |
| `completed`           | Number of successful or valid cached setting results.                                                                    |
| `errors`              | Failed records with `split`, `seed`, `setting_id`, `pipeline`, `data_process`, `definition`, and exception text `error`. |
| `treecluster_enabled` | Whether phylogenetic comparisons were configured.                                                                        |

A failed setting's `manifest.json` contains `status: "failed"` and that error record instead of completion fields. File presence alone does not prove a valid cache; signature and checksums must match. Run-level `complete` describes the requested stage rather than every possible stage in the study.

### Reconstructed SCoVMod tree provenance

The prepared input directory defaults to `evaluation/shared_synthetic/outputs/inputs/`. Its `manifest.json` checks the raw file paths/hashes, `tree_seed`, `target_component_size`, resolved tree and provenance paths, and implementation hash. The `files` mapping covers the GML tree and its source JSON. Matching artifacts are reused; changed signatures or file checksums trigger reconstruction.

`<tree-stem>.source.json` contains `inputs` (input paths and SHA-256 hashes), `tree_sha256`, actual `n_cases`, requested `target_size`, reconstruction `seed`, `implementation_sha256`, and the `tie_order` description.

Each baseline run also writes `inputs.json` beside `settings.json`. It records the shared `experiment` identity, validated `diagnostics` reference, pinned `tree_path`, `tree_sha256`, `n_cases`, `tree_seed`, and `target_component_size`. Its `source` object is the backbone artifact's `source.json`: original `tree_path`, `tree_sha256`, and available preparation `provenance`. Source provenance describes the prepared backbone; shared truth records the cases used, including a smoke subset.

## 9. Phylogenetic artifacts

Each inferred raw/dated phylogeny shares one directory under `artifacts/trees/<id>/`. IQ-TREE infers maximum-likelihood sequence trees and performs LSD2 dating. TreeCluster subsequently partitions the exported trees.

### Synthetic studies

| File                          | Contents and units                                                                                                                                                                                                                                 |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `raw.nwk`                     | IQ-TREE maximum-likelihood tree rooted with the aligned reference as outgroup, then reference-pruned. Branch lengths are substitutions/site; case IDs are restored.                                                                                |
| `dated.nwk`                   | LSD2 time-calibrated tree with the reference excluded. Branch lengths are durations in days.                                                                                                                                                       |
| `node_dates.tsv`              | Tab-separated table with columns: `node`, `case_id`, `is_tip`, `date` (days from origin), `sample_date` (original numeric days).                                                                                                                   |
| `phylogeny.json`              | JSON metadata: `model` (e.g., JC), `threads`, `seed`, `clock_rate` (estimated or fixed), `date_origin`, `reference_name`, `units` (`raw: substitutions_per_site`, `dated: days`), `backend_paths` to IQ-TREE outputs.                              |
| `backend/run-*/`              | IQ-TREE/LSD2 working directory containing: `sampled.fasta` (alignment with reference), `sampling_dates.txt`, `iqtree.iqtree` (model report), `iqtree.treefile` (genetic tree), `iqtree.timetree.lsd` (LSD2 report), `iqtree.timetree.nex` (dated). |
| `backend/run-*/inference.log` | Combined IQ-TREE/LSD2 stdout/stderr; failure messages identify this log. TreeCluster partitions separately retain their stdout/stderr logs.                                                                                                        |

**Signature fields:** `kind: phylogeny-v1`, `dataset`, `process`, `fasta_sha256`, `reference_sha256`, `dates_sha256`, `n_cases`, `iqtree_model`, `iqtree_threads`, `iqtree_seed`, `clock_rate`, `iqtree_executable`, `implementation`.

**Manifest fields:** `units`, `rooting` (reference-outgroup rooting/pruning for raw trees; LSD2 clock for dated trees), `root_split`, `dated_root_split`.

**Reference handling:** The raw tree initially includes the ancestral reference as an outgroup taxon. Before TreeCluster, the reference tip is pruned so that the tree contains exactly the sampled cases. The pruned tree is saved as `raw.nwk`. The dated tree excludes the reference by design (LSD2 removes undated taxa).

**Date units:** Dated branches are durations in **days**. Synthetic node/sample dates retain the numeric simulation origin. Boston calendar dates are normalized to days from the earliest collection date; `date_origin` and calendar columns preserve that origin. TreeCluster day cutoffs are passed directly.

### Boston studies

Boston trees follow the same structure as synthetic studies, with these differences:

| File             | Boston-specific notes                                                                                                |
| ---------------- | -------------------------------------------------------------------------------------------------------------------- |
| `raw.nwk`        | Built from full Boston alignment (29,903 bp) + SARS-CoV-2 reference; IQ-TREE ModelFinder (`MFP`) for empirical data. |
| `dated.nwk`      | LSD2 dating using real collection dates (YYYY-MM-DD converted to days from earliest sample).                         |
| `node_dates.tsv` | Includes `calendar_date` and `sample_calendar_date` columns when calendar dates are provided.                        |
| `phylogeny.json` | `alignment_length: 29903`, `model: MFP`.                                                                             |

**Reference alignment compatibility:** Boston alignment is 29,903 bp (full SARS-CoV-2 genome). The reference sequence must be compatible (same strain/isolate backbone). Gap patterns in the alignment must match the reference to avoid artifactual branch lengths.

**Threshold scaling:** SNP thresholds use the same absolute SNP count range as synthetic ([0-10]), scaled to subs/site by `(snp_count × alignment_length) / 29903` = `(snp_count × 5000) / 29903` for raw trees. Dated tree thresholds are in days and passed directly.

**Signature fields:** `kind: boston-phylogeny-v1`, `alignment_sha256`, `reference_sha256`, `alignment_length`, plus IQ-TREE settings.

### TreeCluster partitions

TreeCluster reads the pruned `raw.nwk` or `dated.nwk` and writes per-setting artifacts under `clusters/` or `trees/`. Output files:

| File                     | Contents                                                                                                         |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `memberships.parquet`    | Columns: `case_id`, `cluster_id`. `cluster_id: -1` indicates unclustered singleton (normalized before analysis). |
| `clusters.parquet`       | Cluster-level summaries: `cluster_id`, `n_cases`, `within_pairs`, optional exposure/mutation counts.             |
| `metrics.json`           | Partition-level metrics: `n_cases`, `n_clusters`, `size_mean`, `size_std`, `largest_cluster`.                    |
| `algorithm.json`         | TreeCluster command, executable identity, `threshold_tree_units`.                                                |
| `treecluster.stdout.log` | TreeCluster output with `SequenceName` and `ClusterNumber` columns.                                              |
| `treecluster.stderr.log` | Captured stderr (including timeout messages).                                                                    |

**Threshold units:**

- **Raw tree:** `threshold_units: snps`, threshold value = `snp_count / alignment_length` (substitutions/site)
- **Dated tree:** `threshold_units: days`, threshold value = days (no conversion)

**Note:** Genetic thresholds preserve absolute SNP counts across studies. For synthetic (5,000 bp), 10 SNPs = 0.002 subs/site. For Boston (29,903 bp), 10 SNPs = 0.000334 subs/site.

## 10. Boston inputs and results

Here `<root>` defaults to `evaluation/03_boston_application/outputs/boston/`. The input tables live under `evaluation/03_boston_application/outputs/inputs/`, produced by `python evaluation/03_boston_application/run.py --stage prepare` or automatically by computational stages with the default input configuration. They contain empirical observations and metadata, without synthetic M truth labels.

`epilink-evaluate boston --stage prepare` uses the Boston config and the same preparation as the Boston `run.py --stage prepare` entry point. Both honor `inputs.cases_path` and `inputs.pairs_path`. Input paths are independent of the run's `--output` override.

### `cases.parquet`

**One row per matched metadata/Nextclade case**, sorted by sample date and ID. The adapter creates or renames:

| Column             | Definition                                                                                                                                                         |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `case_id`          | Source `seq_id`, converted to string; matches Nextclade `seqName`.                                                                                                 |
| `sample_date`      | Parsed source `collection_date`; a calendar timestamp.                                                                                                             |
| `Clade`            | Nextclade `clade`.                                                                                                                                                 |
| `substitutions`    | Nextclade comma-separated mutation list.                                                                                                                           |
| `QC_OverallStatus` | Nextclade `qc.overallStatus`.                                                                                                                                      |
| `Exposure`         | First matching source flag in order: CONF_A_EXPOSURE -> `Conference`, SNF_A_EXPOSURE -> `SNF`, BHCHP -> `BHCHP`, CITY_A_EXPOSURE -> `City`; otherwise `Unlabeled`. |
| `Mutation`         | First marker present in order C2416T, G105T, G28899T, G3892T, C20099T, with the adapter's descriptive label; otherwise `Minor Lineages`.                           |

All other source metadata fields pass through. In the preserved input they are:

| Source fields                                                                                                             | Interpretation                                                                       |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `virus`, `host`, `age`, `sex`, `length`                                                                                   | Source biological/demographic/sequence descriptors, retaining source missing values. |
| `gisaid_epi_isl`, `genbank_accession`, `biosample_accession`, `database`                                                  | Source accessions and database identifiers.                                          |
| `region`, `country`, `division`, `location`, `gb_raw_location`, `geocode_precision`, `geoloc_9cat`, `geoloc_cat`          | Source geographic descriptors/classifications.                                       |
| `region_exposure`, `country_exposure`, `division_exposure`                                                                | Source exposure geography.                                                           |
| `CONF_A_EXPOSURE`, `SNF_A_EXPOSURE`, `CITY_A_EXPOSURE`, `BHCHP`                                                           | Source exposure flags; `YES` drives the derived Exposure label.                      |
| `originating_lab`, `submitting_lab`, `Massachusetts_originating_lab`, `Massachusetts_submitting_lab`, `sequenced_by_2cat` | Source laboratory/sequencing descriptors.                                            |
| `date_submitted`, `authors`, `url`, `title`                                                                               | Source submission/publication metadata.                                              |

### `observed_pairs.parquet`

**One row per supplied non-self unordered pair**:

| Column               | Definition                                                                    |
| -------------------- | ----------------------------------------------------------------------------- |
| `CaseID1`, `CaseID2` | Source ID1/ID2, canonicalized by string order.                                |
| `TN93_distance`      | Source Distance, substitutions/site.                                          |
| `GD`                 | `TN93_distance * 29903`, a scaled distance that can be fractional.            |
| `TD`                 | Absolute sample-date difference in days, without the synthetic rounding step. |

The TN93 source is censored at 0.0005/site; missing pairs remain unobserved. The manifest adds `n_cases`, `n_observed_pairs`, `n_all_pairs`, and `candidate_universe`. Its signature records `kind`, input hashes, `implementation`, `distance_cutoff_per_site`, and `reference_length`.

### Boston empirical run outputs

Directory: `<root>/runs/<run-id>/`, initialized by `python evaluation/03_boston_application/run.py --stage all` or `trees`.

The frozen transfer analysis writes `settings.json`, `selection.json`, `clusters/`, `trees/`, and `assessment/` using baseline-selected operating points. `clusters/status.json` and `trees/status.json` record configured, completed, and failed partitions. `assessment/summary.csv` gives descriptive cluster-size and candidate-coverage summaries; Boston has no synthetic truth precision/recall columns.

`manifest.json` records the requested stage and its status; `inputs.json` records case/pair counts, input hashes/paths, and `trees_enabled`. A complete `trees` run does not imply graph scoring or clustering completed. The `all` stage runs frozen transfer. `report` renders saved tables.

#### Scores and settings

- `<root>/artifacts/scores/<id>/scores.parquet` has `CaseID1`, `CaseID2`, and one column per configured scorer for every observed pair. It has no synthetic `pair_id` column. A frozen `all` run records `score_id` in its manifest; graph artifact signatures also record the score ID used.
- `<run>/settings.json` contains adapted frozen definitions, including their `baseline_setting_id`, `baseline_score_name`, and `baseline_data_process`. Boston's `data_process` is `empirical`, and tree rows use `score_name: TREE`. Pairwise operating definitions are retained as reference metadata; the workflow writes scores and partitions, not Boston pairwise truth-metric tables.
- `<run>/selection.json` contains adapted baseline operating decisions and their criterion names; it is not a selection performed using Boston exposures.
- All scorers use the same empirical GD/TD observations. Synthetic D/S labels identify source rules or classifiers. Graph pipelines use names such as `leiden/ESS`; tree pipelines use `treecluster/empirical/<baseline-process>/<raw-or-dated>`.

#### Partition artifacts

`clusters/<setting-id>/` and `trees/<setting-id>/` contain `memberships.parquet`, `clusters.parquet`, `metrics.json`, `algorithm.json`, and a completion manifest. TreeCluster additionally saves its stdout/stderr logs. Each membership table has one row per case, including isolates/singletons, with `case_id` and `cluster_id`.

The per-cluster table has `cluster_id`, `n_cases`, `within_pairs`, and `n_<field>_<label>` counts for available `Exposure`, `Clade`, and `Mutation` labels. `within_pairs` counts all co-clustered unordered pairs, including pairs absent from the censored scoring table. The graph/tree `metrics.csv` tables give one row per completed setting with source identifiers and these metrics:

| Metric                  | Definition                                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| `n_cases`, `n_clusters` | Total cases and clusters, including singletons.                                                |
| `size_mean`, `size_std` | Unweighted mean and sample SD of cluster sizes; SD is set to zero for a one-cluster partition. |
| `largest_cluster`       | Number of cases in the largest cluster.                                                        |

#### Assessment tables

The following tables appear under `assessment/`. Identifiers include `setting_id`, `pipeline`, and `score_name`; source setting IDs identify the baseline settings applied to Boston.

| Table                        | Row unit and interpretation                                                                                                                                                                                                                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `summary.csv`                | One row per completed partition: cluster counts, singleton cases, largest cluster, and input coverage. `n_non_singleton_clusters` counts clusters with size **at least `assessment.min_cluster_size`**; the default is 2.                            |
| `cluster_composition.csv`    | One row per observed cluster/metadata-field/label combination: `n_evidence` and `fraction_in_cluster = n_evidence / n_cases`. Zero-count combinations are omitted.                                                                                   |
| `named_cluster_overlaps.csv` | One representative cluster per setting/focus exposure, chosen by the greatest number of exposed cases among eligible clusters; ties prefer larger clusters, then smaller cluster ID. No row is written if no eligible cluster contains the exposure. |
| `tree_agreement.csv`         | One row per completed graph/TreeCluster setting pair: `adjusted_rand` and `adjusted_mutual_information` across all cases. These compare generated partitions, not transmission truth.                                                                |
| `named_tree_overlaps.csv`    | Membership overlap of the representative exposure clusters in each graph/TreeCluster setting pair.                                                                                                                                                   |
| `cluster_overlaps.csv`       | When an external comparator is configured, nonzero overlaps of eligible focus clusters with its groups. Otherwise header-only.                                                                                                                       |
| `best_cluster_overlaps.csv`  | The external group with the most shared cases for each setting/exposure/cluster; ties prefer larger Jaccard, then group ID. Otherwise header-only.                                                                                                   |

For `named_cluster_overlaps.csv`:

| Column                                                                                                               | Meaning                                                                                                                            |
| -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `n_cases`, `n_exposure`                                                                                              | Representative cluster size and number of its cases carrying the focus-exposure label.                                             |
| `exposure_total`                                                                                                     | Number of cases with that label in the entire Boston case table.                                                                   |
| `exposure_fraction`                                                                                                  | `n_exposure / n_cases`: concentration within the representative cluster.                                                           |
| `exposure_recovery`                                                                                                  | `n_exposure / exposure_total`: fraction of the exposure group in that one cluster, not recovery summed across all clusters.        |
| `treecluster_group`, `treecluster_size`, `shared`, `model_overlap_percent`, `treecluster_overlap_percent`, `jaccard` | Optional comparison with a representative group from `assessment.treecluster_path`; blank when no external comparator is supplied. |

For any two compared membership sets A and B, `shared = |A ∩ B|` and `jaccard = |A ∩ B| / |A ∪ B|`. Overlap-percent columns divide the shared count by the respective set size and multiply by 100; exposure fractions and Jaccard are on a 0–1 scale. `named_tree_overlaps.csv` uses `model_size`, `tree_size`, and `tree_overlap_percent` for the generated TreeCluster counterpart.

Descriptive **fold enrichment** can be calculated as `exposure_fraction / (exposure_total / total_Boston_cases)`. It is not currently a saved assessment column or a significance test. Interpret it together with `exposure_recovery`. Exposure labels are mutually exclusive as defined above; these summaries do not establish transmission precision/recall.

`candidate_coverage = n_observed_pairs / n_all_pairs`, where `n_all_pairs = total_Boston_cases * (total_Boston_cases - 1) / 2`. It describes the censored pair table and is repeated on tree rows for context. Boston sequence trees use the full alignment directly.

#### Boston tree artifacts

Boston tree artifacts contain `raw.nwk`, `dated.nwk`, `node_dates.tsv`, `phylogeny.json`, `sampling_dates.csv` with collection dates, and IQ-TREE/LSD2 outputs under `backend/run-*/`. They use aligned samples plus a same-length reference FASTA. Raw branches are substitutions/site; dated branches are days. `trees/inputs.json` pins paths and hashes. Run `inputs.json` additionally records aligned reference path/hash and alignment length; the run signature includes phylogeny settings and both sequence inputs.

For raw TreeCluster transfer, the selected source SNP count is recovered using the baseline alignment length and converted to substitutions/site using the Boston length. Adapted raw definitions retain `baseline_threshold`, `threshold_snps` and `baseline_setting_id`; dated cutoffs remain unchanged in days. Execution uses the same TreeCluster settings whose tool identity is recorded. Optional Boston overrides affect only executable/timeout.

## 11. Worked joins in Python

Run these examples from the repository root in the installed environment. They read existing artifacts and require completed development results. The examples start with the small smoke output; change `root` to use a full experiment.

### Attach method definitions to summary rows

```python
import json
from pathlib import Path

import pandas as pd

root = Path("evaluation/01_synthetic_baseline/outputs/baseline_smoke")
run = Path(json.loads((root / "current.json").read_text())["run_directory"])
definitions = pd.DataFrame.from_dict(
    json.loads((run / "settings.json").read_text()), orient="index"
).rename_axis("setting_id").reset_index()
summary = pd.read_csv(
    run / "development/summary.csv", dtype={"setting_id": str}
)
annotated = summary.merge(
    definitions, on=["pipeline", "setting_id"], validate="many_to_one"
)
print(annotated.reindex(columns=[
    "pipeline", "setting_id", "threshold", "resolution",
    "M0_precision_mean", "M0_recall_mean", "M0_f1_mean", "n_realizations",
]).to_string(index=False))
```

### Follow lineage before joining pairs to truth and scores

Continue with `root` and `run` above. A seed alone is insufficient to choose an observation artifact: multiple configurations can reuse the same seed. Follow the selected run's manifests instead. Models/scores are baseline-local; observations and truth belong to its pinned shared experiment, which may differ from the latest shared `current.json`.

```python
from epilink_evaluation.inputs.synthetic import (
    analysis_table, load_observations, load_truth,
)

run_manifest = json.loads((run / "manifest.json").read_text())
experiment = json.loads((run / "experiment.json").read_text())
experiment_dir = Path(experiment["experiment_directory"])
experiment_manifest = json.loads((experiment_dir / "manifest.json").read_text())
assert experiment_manifest["fingerprint"] == experiment["fingerprint"]
shared_root = experiment_dir.parent.parent
seed = run_manifest["config"]["splits"]["development"][0]
seed_dir = run / "development" / f"seed_{seed}"
pairwise = json.loads((seed_dir / "pairwise/manifest.json").read_text())
score_dir = root / "artifacts/scores" / pairwise["signature"]["score_id"]
score_manifest = json.loads((score_dir / "manifest.json").read_text())
observation_dir = (
    shared_root / "artifacts/observations" / score_manifest["signature"]["dataset"]
)
observation_link = json.loads(
    (experiment_dir / "observations" / f"seed_{seed}.json").read_text()
)
assert observation_link["dataset"] == observation_dir.name
observation_manifest = json.loads(
    (observation_dir / "manifest.json").read_text()
)
assert observation_manifest["signature"]["truth"] == experiment_manifest["signature"]["truth"]
truth_dir = shared_root / "artifacts/truth" / observation_manifest["signature"]["truth"]
observations, cases = load_observations(observation_dir)
truth = load_truth(truth_dir, observations.pair_id)
scores = pd.read_parquet(score_dir / "scores.parquet")
joined = analysis_table(observations, cases, truth, scores)
print(joined.columns.tolist())
```

`analysis_table` joins on `pair_id` with one-to-one validation and adds `CaseID1`, `CaseID2` using sampled-case positions. The joined table includes observation, truth, and configured score columns. These helpers load pair tables into memory; full runs can have millions of rows.

### Attach one partition to sampled cases

Continue from the lineage example to use its matching `cases` table:

```python
partitions = pd.read_csv(
    seed_dir / "clusters/metrics.csv", dtype={"setting_id": str}
)
setting_id = partitions.iloc[0]["setting_id"]  # A completed setting in this seed.
membership_path = seed_dir / "clusters" / setting_id / "memberships.parquet"
memberships = pd.read_parquet(membership_path)
case_clusters = cases.merge(memberships, on="case_id", validate="one_to_one")
print(case_clusters.groupby("cluster_id").size().rename("n_cases"))
```

The same `cluster_id` can join this partition's `clusters.parquet`. Keep the run, split, seed, and setting identity when combining multiple partitions.

## 12. Perturbation study outputs

The [perturbation runner](evaluation/02_synthetic_perturbation/README.md) has a separate `<root>` under `evaluation/02_synthetic_perturbation/outputs/perturbation/` or `perturbation_smoke/`. Its `current.json` and `runs/<id>/` naming follow the baseline convention. In this section, `<run>` is a perturbation run.

### Identity, scenarios, and coverage

| File             | Fields / purpose                                                                                                                                                                                                                          |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `manifest.json`  | Study `status`, `requested_stage: all`, schema-2 `config`, full `signature`, `git_revision`, study `n_cases`, `run_directory`, and optional caught `error`.                                                                               |
| `reference.json` | Source `run_directory`, `run_fingerprint`, original `selection_fingerprint`, `truth_fingerprint`, reference `n_cases`, and `baseline_implementation`. No fitted models are needed.                                                        |
| `selection.json` | Baseline selection metadata filtered to the requested EpiLink pipelines and criterion.                                                                                                                                                    |
| `settings.json`  | Baseline-selected full-graph EpiLink Leiden definitions. `graph_mode: full`, `threshold: null`, `empty: false`; only resolution is selected.                                                                                              |
| `scenarios.json` | List of `name`, `parameter`, absolute `value`, `baseline_value`, `multiplier` (null for absolute levels), and complete scenario `generation` parameters. The unperturbed scenario is named `baseline` and its parameter metadata is null. |
| `coverage.csv`   | One row per scenario/mode with `inference_mode`, `clustering_mode`, `status`, `completed`, `expected`, `error`. Expected count is requested EpiLink scorers × evaluation seeds.                                                           |

Study signatures include effective configuration, resolved reference identity, scenarios, implementation and truth ID. Each `scenarios/<scenario>/<mode>/manifest.json` records replay status, configuration/signature and an optional error. Failed/not-run arms contribute no result rows. Complete status requires all four arms in every scenario.

### Absolute clustering results

`results.csv` contains evaluation partition metrics from section 4, joined to the configured baseline criterion. The study produces no pairwise rankings or ambiguity tables. Added metadata:

| Column            | Definition                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------------ |
| `scenario`        | Resolved scenario name from `scenarios.json`, including the `baseline` control.            |
| `mode`            | One of the four `baseline_inference_*_clustering` / `matched_inference_*_clustering` arms. |
| `inference_mode`  | `baseline` or `matched`.                                                                   |
| `clustering_mode` | `baseline` or `updated` resolution.                                                        |
| `parameter`       | Changed natural-history field, such as `incubation.mean`; blank for controls.              |
| `value`           | Absolute perturbed value in the parameter's original units.                                |
| `baseline_value`  | Original natural-history parameter value, not a performance metric.                        |
| `multiplier`      | Requested relative multiplier, or blank for absolute levels and controls.                  |

Top-level rows use `split: evaluation` and fresh evaluation seeds; development evidence remains inside each updated arm. `pipeline` is `leiden/<EpiLink-scorer>`, with EDD/EDS/ESD/ESS in the supplied study. `setting_id` identifies the actual resolution/definition used for that scenario and arm. Complete default coverage is 624 rows, or 48 in smoke mode.

### Paired deltas

`results_deltas.csv` contains perturbed rows only and adds:

| Column pattern        | Meaning                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| `baseline_<metric>`   | Unperturbed control's metric on the same seed, mode, and comparison identity.                     |
| `delta_<metric>`      | `metric - baseline_<metric>`, in the original metric's units.                                     |
| `control_available`   | Whether a matching control row exists; true does not guarantee every metric is defined.           |
| `baseline_setting_id` | Control's actual resolution/definition ID, potentially different from the perturbed `setting_id`. |

Controls match on `(mode, seed, criterion, pipeline)`, **excluding setting ID** because updated resolution may change between scenario and control. Parameter metadata/seeds are not differenced. Missing controls survive the left join with missing metrics/deltas. Negative F1 changes indicate reduced recovery; positive contamination changes indicate more distant within-cluster pairs. Controls are fresh paired observations, not the reference's old held-out observations.

### Summary tables

| File                        | Groups and values                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| `results_summary.csv`       | Absolute metrics by `(scenario, mode, criterion, pipeline, setting_id)`. Includes controls. |
| `results_delta_summary.csv` | Delta metrics grouped like operating results, excluding control scenarios.                  |

Each metric receives `_mean`, `_std`, `_min`, `_max`, and `_count` suffixes. Means give each nonmissing realization equal weight; SD is sample SD (`ddof=1`), undefined with fewer than two valid values. `_count` is the number of nonmissing values for that specific metric. `n_realizations` counts group rows; delta summaries additionally report `n_controls`, the number with a matching control row. For example, `delta_M0_f1_count` can be smaller than `n_controls` when F1 is undefined. Join `scenario` to `scenarios.json` for absolute parameter values.

### Detailed artifacts and reports

`scenarios/<scenario>/<mode>/selection.json` records actual operating points, inference/clustering modes, development seeds (empty for baseline-resolution arms), and development-evidence checksum (null for baseline arms). `settings.json` stores replayed definitions. `evaluation/heldout_access.json` pins selection fingerprint and evaluation seeds before observations are released.

Updated arms have `development/metrics.csv`, summaries, and per-setting development artifacts. All arms have `evaluation/seed_<seed>/clusters/<setting-id>/` memberships, cluster tables, metrics, algorithm metadata and checksummed manifests, using section 4's schemas. Full graphs retain every observed edge including zero weights. All arms share scenario/seed observations; arms with the same inference share scores.

`artifacts/backbones/<id>/transmission_tree.gml` stores the frozen reference topology or smoke prefix. Truth, observations and training-free scores live under the study's `artifacts/truth/`, `artifacts/observations/` and `artifacts/scores/`; score signatures have `training: null`. No models or phylogenetic artifacts are generated.

`report.md` and `report.html` show coverage, parameter levels, actual resolutions, absolute clustering performance and paired changes. Smoke reports are labeled pipeline validation. Standalone `fig12`, `fig14`, `fig15`, `fig23` and `tab03` consume these clustering-only tables; see the study guide for display interpretation.

## 13. Shared experiments and diagnostics

### Shared experiment identity and access

The shared config owns `inputs`, `generation`, `simulation`, and `splits`. Diagnostics and baseline reference it through `experiment_config`; baseline derives matched inference from generation. Default shared root: `evaluation/shared_synthetic/outputs/synthetic/`, or `synthetic_smoke/`.

| Path relative to shared root                                 | Contract                                                                                                                                                                                                       |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `current.json`                                               | Latest prepared `experiment_directory` (absolute) and full `fingerprint`. Preparation alone does not imply diagnostics completion.                                                                             |
| `experiments/<id>/experiment.json`                           | Resolved data design (`inputs`, `generation`, `simulation`, `splits`), schema version, and shared `output_directory`; `inputs.tree_path` pins the backbone copy.                                               |
| `experiments/<id>/manifest.json`                             | Completion manifest with signature `kind`, original `specification`, generation `producer`, `backbone` and `truth` artifact IDs; checksums cover `experiment.json`. IDs use 20-character fingerprint prefixes. |
| `artifacts/backbones/<id>/`                                  | Pinned `transmission_tree.gml`, `source.json` (original tree path/hash and available preparation provenance), and completion manifest. Signature includes source SHA-256, nodes, and edges.                    |
| `artifacts/truth/<id>/`, `artifacts/observations/<id>/`      | Shared pair/case schemas from section 6. Diagnostics prepares development observations; baseline prepares training and, after frozen release, evaluation observations.                                         |
| `experiments/<id>/observations/seed_<seed>.json`             | `seed`, `role` (`train`, `development`, `evaluation`), `dataset` directory ID, and observation `fingerprint`. Created when that dataset is prepared/released.                                                  |
| `experiments/<id>/diagnostics.json`                          | Completed diagnostics `run_directory`, diagnostics `fingerprint`, shared `experiment` identity, exact development `datasets` mapping (seed string → artifact ID), and `status: complete`.                      |
| `heldout_access/seed_<seed>.json`                            | `seed`, shared `experiment` identity, and `selection_fingerprint`; records access before evaluation observations are generated.                                                                                |
| `validation_access/<selection-fingerprint>/seed_<seed>.json` | The same access fields for smoke validation. Smoke observations may be reused across changed comparison implementations; they are pipeline checks rather than held-out scientific evidence.                    |

Baseline pins the shared identity in its own `<run>/experiment.json` and validates the diagnostics marker against checksummed completion evidence. Follow that pinned identity for joins, rather than the shared latest pointer. `reset-outputs` preserves the entire shared output area, including full/smoke held-out ledgers. Previously accessed evaluation seeds cannot be reassigned to training/development or used with revised frozen selection. Retained outputs from earlier versions are historical; current evidence requires the diagnostics-first workflow.

### Diagnostics layout and completion

Here `<diagnostics-root>` is `evaluation/00_synthetic_diagnostics/outputs/diagnostics/` (or `diagnostics_smoke/`), and `<diagnostics-run>` is its `runs/<full-signature-fingerprint>/` directory.

| Path                                                      | Contents                                                                                                                                                                                                                                |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `<diagnostics-root>/current.json`                         | `run_directory`, diagnostics `fingerprint`, and shared `experiment` identity.                                                                                                                                                           |
| `<diagnostics-run>/manifest.json`                         | `signature`, `status`, `requested_stage`, `experiment`, resolved `config`, `run_directory`, `coverage_complete`, and optional `error`.                                                                                                  |
| `<diagnostics-run>/<stage>/index.json`                    | `status`, `records`, `errors`, and seed-to-dataset map. `backbone` has one record and an empty dataset map; other stages use exact development datasets. Records link absolute `artifact` paths; graph records also link `source`. |
| `<diagnostics-root>/artifacts/<kind>/<full-fingerprint>/` | Diagnostic feature tables or known-truth controls with completion manifests; failed artifacts retain `status: failed` and `error`. Kinds are described below.                                                                           |
| `<diagnostics-run>/completion/`                           | Checksummed `coverage.json` and manifest, released only when all required stages and their evidence validate.                                                                                                                           |
| `<diagnostics-run>/report.md`, `report.html`, `figures/`  | Saved-table reports with stage coverage, visible errors, descriptive summaries, and figures.                                                                                                                                            |

Completion signature fields are `experiment`, `diagnostics` (run fingerprint), and `datasets`. `coverage.json` has `status`, required `stages` (`backbone`, `observations`, `graphs`), `datasets`, `aggregation`, and `artifacts`: absolute directory → `manifest_sha256` plus the file-name-to-SHA256 `files` inventory. Baseline validates this inventory as well as the marker. A complete requested `prepare` or individual stage is insufficient without full required coverage. Failed graph controls prevent completion. Report-only execution reads saved files without generating observations.

### Fixed transmission-backbone characterisation

`<diagnostics-run>/backbone/index.json` has `scope: "one fixed backbone"`, an empty `datasets` map, and exactly one record containing the pinned `backbone` ID and an absolute `artifact` path. `backbone/summary.csv` contains one row of scalar summaries; it is not replicated by observation seed. The stage and its evidence participate in the diagnostics completion inventory.

`<diagnostics-root>/artifacts/backbone/<id>/` is keyed by the immutable backbone ID, scoped computational implementation and resolved `diagnostics.backbone` settings. Sampling fractions, genetic processes and observation seeds do not independently repeat this artefact. It contains:

| File                | Row unit and contents                                                                                                                                                                                                                                                                                 |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nodes.parquet`     | One row per full-backbone case: `node_index`, string `case_id`, string `root_case_id`, `depth` in transmission hops, direct `offspring`, all-generation `descendant_count`, and Boolean `is_superspreader`. Node indices align with this backbone's truth nodes, including unobserved cases.          |
| `offspring.csv`     | One row per integer `offspring` from zero through the larger of the maximum observed count and Poisson cutoff. `n_cases`, `case_fraction`, `poisson_probability`, `negative_binomial_probability`, and `qualifies_superspreading`. Model probabilities are not renormalised to the displayed support. |
| `concentration.csv` | Ranks 0 through N after sorting offspring descending: `rank`, `case_fraction`, `cumulative_transmissions`, `transmission_fraction`. Includes all zero-offspring cases and retains whole cases when reaching 80%.                                                                                      |
| `generations.csv`   | One row per root-relative `depth`: `n_cases` and `case_fraction`, pooling introductions at equal hop depth.                                                                                                                                                                                           |
| `components.csv`    | One row per `root_case_id`: `n_cases`, `n_transmissions`, `max_depth`.                                                                                                                                                                                                                                |
| `bootstrap.csv`     | One row per optional case-resampling `replicate`: `mean_offspring`, nullable `dispersion_k`, `fit_method`. Disabled resampling produces a header-only table.                                                                                                                                          |
| `summary.json`      | Scalar structure, concentration, fitting and superspreading summaries, plus `bootstrap` metadata. Undefined numeric values are JSON null.                                                                                                                                                             |
| `provenance.json`   | Backbone identity, full-truth case order, offspring definition, inclusive superspreading rule and single-backbone replication scope.                                                                                                                                                                  |

`summary.json` uses counts for `n_cases`, `n_transmissions`, `max_offspring`, `n_zero_offspring`, `n_roots`, `n_components`, `largest_component_cases`, `n_superspreaders`, `superspreader_transmissions`, `n_cases_for_80_percent`, and `top20_n_cases`. Fractions use these denominators:

- `zero_offspring_fraction`, `superspreader_fraction`, `largest_component_fraction`, `fraction_for_80_percent`, `top20_case_fraction`: all backbone cases.
- `superspreader_transmission_fraction`, `top20_transmission_fraction`: all direct transmission edges. The top-20% summary includes `ceil(0.2 * N)` whole cases and records the actual selected case fraction.
- `M0_prevalence`: all unordered backbone pairs. `n_direct_transmission_pairs` is the edge count; `n_shared_infector_pairs = sum_i offspring_i * (offspring_i - 1) / 2`; `n_M0_pairs` is their sum.

`mean_offspring = n_transmissions / n_cases`; `offspring_variance` uses `ddof=0`, `offspring_sample_variance` uses `ddof=1`, and `variance_to_mean` uses the population variance. `mean_depth` and `max_depth` are transmission-hop depths. In a single-parent forest the mean offspring is `(N - C) / N`, where C is the number of introductions. This is a descriptive reference for the selected backbone.

`dispersion_k` describes the negative-binomial fit with mean fixed at the empirical mean and variance `R + R²/k`. `fit_method` is `mle`, `moments_fallback`, `poisson_limit`, `degenerate`, or `insufficient_cases`; `fit_notes` exposes fallback/boundary details. There is no finite `k` at the Poisson limit. Its model probabilities use the Poisson limiting distribution; degenerate or insufficient fits leave the NB probabilities undefined.

The saved superspreading definition is **inclusive**: `offspring >= poisson_percentile`, where `poisson_percentile = Poisson(mean_offspring).ppf(superspreading_quantile)` and the default quantile is 0.99. `superspreading_reference` is `backbone_mean_offspring`; `superspreading_operator` is `>=`; `minimum_superspreading_offspring` equals that percentile. With no transmissions, no case is flagged, the minimum qualifying count and transmission-share/80%-concentration summaries are undefined.

The `bootstrap` object records `requested`, `completed`, `seed`, `interpretation`, `mean_offspring_kept`, `dispersion_k_kept`, and their `*_interval95` arrays. Intervals are 2.5th/97.5th percentiles of finite case-resampling estimates; dispersion intervals condition on finite fits and retain their contributing count. These optional exploratory IID intervals are disabled by default. Neither seed replication nor the fixed tree supplies independent epidemic replicates.

`fig24_backbone_characterisation` exports the three plotted distributions, component/resampling tables, scalar summary CSV, source-pinned JSON summary, and Markdown caption under the manuscript result directory. Full and smoke backbone scopes are labelled from the saved run configuration.

### Exact observation feature cells

`artifacts/observations/<id>/` under the **diagnostics root** contains diagnostic tables, distinct from the shared root's raw observation artifacts. The stage index records each source `dataset` ID and its diagnostic `artifact` path.

`cells.parquet` has **one row per occupied exact feature cell, process, endpoint, and seed**. No additional rounding or binning is applied to saved GD/TD values.

| Column                                       | Definition                                                                                                |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `seed`, `process`, `feature_set`, `endpoint` | Development seed; `deterministic`/`stochastic`; `GD`/`GD_TD`; `M0`/`Mle1`/`Mle2`.                         |
| `GD`, `TD`                                   | Exact saved coordinates, substitutions and days. `TD` is null for GD-only cells.                          |
| `n_pairs`, `n_target`, `n_other`             | Cell occupancy, endpoint-positive count, and endpoint-negative count; the last two sum to occupancy.      |
| `target_fraction`, `mixed`                   | `n_target / n_pairs`; boolean indicating both classes occur in the cell.                                  |
| `n_<category>`                               | All ten exhaustive relationship counts from section 2, including `n_separate`. Sum equals cell occupancy. |

`summary.csv` has **one row per seed/process/feature_set/endpoint**. It contains `n_pairs`, `n_target`, `n_other` for the whole sampled pair universe, plus `n_cells`, `mixed_cells`, `n_pairs_in_mixed_cells`, `n_target_in_mixed_cells`, and `n_other_in_mixed_cells`. Its ratios use these distinct denominators:

| Column                                        | Definition                                                                                                                |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `mixed_cell_fraction`                         | Mixed cells / all occupied cells.                                                                                         |
| `pair_fraction_in_mixed_cells`                | Pairs in mixed cells / all observed pairs.                                                                                |
| `target_fraction_in_mixed_cells`              | Targets in mixed cells / all targets.                                                                                     |
| `target_prevalence_in_mixed_cells`            | Targets in mixed cells / all pairs in mixed cells.                                                                        |
| `non_target_fraction_in_mixed_cells`          | Non-targets in mixed cells / all non-targets.                                                                             |
| `class_conditional_overlap`                   | Sum over cells of `min(n_target_cell / total_targets, n_other_cell / total_others)`; undefined if either class is absent. |
| `minimum_feature_only_misclassifications`     | Exact count `sum_cells min(n_target_cell, n_other_cell)`.                                                                 |
| `minimum_feature_only_misclassification_rate` | That minimum count / all observed pairs.                                                                                  |

Zero-denominator ratios are undefined. The minimum error applies empirically to target/non-target decisions constant within exact feature cells. It is neither a population performance ceiling nor a bound on partition recovery.

`prevalence.csv` has one row per seed/endpoint: `n_pairs`, `n_target`, `n_other`, `target_prevalence = n_target / n_pairs`. `relationships.csv` has one row per seed/relationship: `n_pairs` is **that category's count**, and `pair_fraction` divides it by all observed pairs. These truth summaries have no process/feature-set replication.

Run-level `observations/` concatenates the per-seed `summary.csv`, `prevalence.csv`, and `relationships.csv`, and writes corresponding `*_aggregate.csv` tables. Numeric fields receive `_mean`, `_min`, `_max`, and `_count` (defined-value count); `n_seeds` records contributing distinct seeds. Means give seeds equal weight.

### Endpoint-oracle graph controls

For each sampled-case set, one graph per horizon h=0,1,2 connects precisely finite M≤h pairs. Vertices include isolates and edges have unit weight. The graph is not necessarily a union of cliques.

- `artifacts/graphs/<id>/cases.parquet`: `case_id`, `node_index`, ordered lexically by string case ID. `edges.parquet`: `a`, `b` (positions in this cases table), `weight: 1`. `provenance.json` records truth/sample identity, endpoint, horizon, edge rule, weight, pair universe, and case order.
- `summary.json`: `n_cases`, `n_target_edges`, `n_components`, `n_isolates`, `largest_component`, `largest_component_fraction` (denominator `n_cases`), and `n_wedges = sum_vertices degree * (degree - 1) / 2`.
- `artifacts/graph_partitions/<id>/`: `memberships.parquet`, `clusters.parquet`, `metrics.json` use section 4 schemas; `algorithm.json` records components or Leiden settings/details; `provenance.json` links the source graph. Leiden restarts are selected by algorithm objective, not truth metrics.

`graphs/metrics.csv` has one row per `(seed, endpoint, algorithm, resolution)`; components has null resolution. `endpoint` identifies the graph's edge rule; each row includes metrics for **all** endpoints and all within-cluster pairs, including graph nonedges. `graphs/summary.csv` aggregates across seeds with the same `_mean`, `_min`, `_max`, `_count`, `n_seeds` convention. `graph_summary.csv` has one row per seed/endpoint with graph-structure fields; `graph_summary_aggregate.csv` aggregates those by endpoint.

Graph controls are keyed by truth and the canonical sampled-case set, independently of seed, GD/TD, and genetic process. Full sampling reuses one graph per horizon across seeds/processes. The stage tables repeat shared results by seed for descriptive equal-seed summaries, not independent control replicates. Neither these oracle partitions nor dependent pairs establish a universal performance ceiling or pair-based confidence interval.

### Diagnostic figures

- `feature_cells_deterministic.png`, `feature_cells_stochastic.png`: exact GD_TD target fraction and occupancy for the lowest completed development seed, explicitly labeled as a single-realization example; occupancy uses log color.
- `backbone_characterisation.png`: offspring frequencies with saved negative-binomial/Poisson fits and the inclusive superspreading cutoff, transmission concentration, and cases by root-relative generation; one full or smoke backbone, without seed replication.
- `oracle_graph_precision_recall.png`: equal-seed within-pair precision/recall for components and Leiden, with Leiden resolution labels, by graph endpoint.

Figures live under the diagnostics run's `figures/` and appear as evidence becomes available. Reports expose partial coverage and failed controls alongside completed tables; inspect coverage before interpreting a summary.
