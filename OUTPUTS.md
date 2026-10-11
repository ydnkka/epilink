# Output reference

Column definitions and JSON fields for `epilink_evaluation` studies. For commands and layout, see [operational guide](OPERATIONS.md#7-find-and-interpret-results). Scientific interpretation: [diagnostics](evaluation/00_synthetic_diagnostics/README.md), [baseline](evaluation/01_synthetic_baseline/README.md), [perturbation](evaluation/02_synthetic_perturbation/README.md), [Boston](evaluation/03_boston_application/README.md).

Paths: `<root>` = output directory, `<run>` = `<root>/runs/<run-id>`, `<split>` = `development`/`evaluation`. Sections 1–9: synthetic-baseline outputs. Section 10: Boston. Section 12: perturbation (reuses baseline schemas). Section 13: shared experiment/diagnostics. `<shared-root>` from `experiment.json`; defaults to `evaluation/shared_synthetic/outputs/synthetic/`.

Prepared inputs:

| Directory | Files |
|-----------|-------|
| `evaluation/shared_synthetic/outputs/inputs/` | `transmission_tree.gml`, `transmission_tree.source.json`, `manifest.json` |
| `evaluation/03_boston_application/outputs/inputs/` | `cases.parquet`, `observed_pairs.parquet`, `manifest.json` |

SCoVMod/diagnostics use shared inputs; baseline reads experiment's backbone copy. Boston uses its own inputs. `--output` doesn't change shared paths; `--smoke` suffixes roots, not input paths. `inputs.json` records provenance.

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

Column names are case-sensitive (`M`, `M0_AP`, `GD_D`, `TD`). Counts are nonnegative integers; CSVs may show floats for missing values. Identifiers are strings. Row units specified below.

| Identifier | Meaning |
|------------|---------|
| `split` | `development` or `evaluation` |
| `seed` | Observation-realization seed (not algorithm seed) |
| `score_name` | EDD, EDS, ESD, ESS, GD_D, GD_S, LOGIT_D, LOGIT_S |
| `data_process` | `deterministic` or `stochastic` |
| `score_family` | `epilink`, `genetic`, `logistic` |
| `pipeline` | Comparison family (e.g., `pairwise/EDD`, `components/GD_D`) |
| `setting_id` | 20-char hash of method definition; join to `settings.json` |
| `criterion` | Selection rule name (e.g., `balanced_M0`) |
| `case_id` | String case identifier |
| `pair_id` | Zero-based unordered-pair index (not globally unique) |
| `cluster_id` | Partition-local integer |

Per-setting key: `(split, seed, pipeline, setting_id)`. Operating results add `criterion`. Across runs, include run ID.

**Missing-value conventions:**

- CSV: empty cells (pandas reads as NaN)
- Parquet: nullable values; inactive relationship counts are null (zero is meaningful)
- JSON: non-finite values → `null`
- Reports: missing = `undefined`; HTML may show NaN
- `threshold: null` with `empty: true` = empty selection

Absent files = stage not produced. Some CSVs intentionally empty (e.g., `calibration.csv` without logistic scorers).

## 2. Shared pair-metric columns

Fields in pairwise threshold/curve tables and partition metrics. Use **observed unordered-pair universe** (no self-pairs). Clustering selected set = **all within-cluster pairs** (can exceed graph-edge set).

Let: `U` = observed pairs (`n_cases * (n_cases - 1) / 2`), `S` = selected pairs, `P_h` = target pairs at horizon `h`, `TP_h` = selected target pairs.

### Endpoint and relationship suffixes

Endpoints:

| Prefix | Target |
|--------|--------|
| `M0` | M = 0: AD(0) or CA(0,0) |
| `Mle1` | M <= 1 |
| `Mle2` | M <= 2 |

Categories (10 suffixes):

| Category | Relationship |
|----------|--------------|
| `AD0` | Direct transmission (0 intermediates) |
| `AD1` | AD with 1 intermediate |
| `AD2` | AD with 2 intermediates |
| `ADge3` | AD with >= 3 intermediates |
| `CA00` | Same infector |
| `CA01` | Shared ancestor (0,1) or (1,0) |
| `CA02` | Shared ancestor (0,2) or (2,0) |
| `CA11` | Shared ancestor (1,1) |
| `CAge3` | Shared ancestor M >= 3 |
| `separate` | Different components (M undefined) |

### Metric definitions

Fractions, precision, recall, F1 ∈ `[0, 1]` when defined. Enrichment can exceed 1.

| Column | Definition | Undefined when |
|--------|------------|----------------|
| `selected_pairs` | `S` (count) | Never |
| `selected_fraction` | `S / U` | `U = 0` |
| `<endpoint>_precision` | `TP_h / S` | `S = 0` |
| `<endpoint>_recall` | `TP_h / P_h` | `P_h = 0` |
| `<endpoint>_f1` | `2 * TP_h / (S + P_h)` | `S + P_h = 0` |
| `<endpoint>_enrichment` | `(TP_h / S) / (P_h / U)` | `S = 0` or `P_h = 0` |
| `n_<category>` | Selected pairs in category | Never |
| `fraction_<category>` | `n_<category> / S` | `S = 0` |
| `Mge3_contamination` | `(n_ADge3 + n_CAge3) / S` | `S = 0` |
| `separate_fraction` | `n_separate / S` | `S = 0` |
| `direct_edge_retention` | Selected AD0 / all AD0 | No AD0 pairs |
| `shared_infector_retention` | Selected CA00 / all CA00 | No CA00 pairs |

`M0_precision`: `TP_0 = n_AD0 + n_CA00`. `Mle1`: adds `n_AD1 + n_CA01`. `Mle2`: adds `n_AD2 + n_CA02 + n_CA11`. Ten `n_<category>` sum to `selected_pairs`.

Empty selection with positives: precision undefined, recall/F1 = 0. No positives but pairs selected: precision/F1 = 0, recall undefined. M=0 false-positive fraction: `1 - M0_precision`. M>=3 contamination excludes separate introductions.

## 3. Per-seed pairwise tables

Directory: `<run>/<split>/seed_<seed>/pairwise/`. Identifiers: `split`, `seed`, `score_name`, `data_process`.

### `metrics.csv`

**One row per scorer threshold.** Columns: four identifiers, `pipeline`, `setting_id`, section 2 metrics. Development: union of distinct score cutoffs + empty selection per scorer. Cutoffs apply to all seeds, retaining ties. Evaluation: selected settings only. No `threshold` column; look up in `settings.json`.

### `rankings.csv`

**One row per scorer** (development and evaluation). Uses whole pair universe.

| Column | Definition |
|--------|------------|
| `split`, `seed`, `score_name`, `data_process` | Identifiers |
| `score_family` | `epilink`, `genetic`, `logistic` |
| `n_pairs` | `U` (total pairs) |
| `unique_scores` | Distinct scores/distances |
| `<endpoint>_AP` | Tie-aware average precision (undefined if no positives) |
| `<endpoint>_prevalence` | `P_h / U` |
| `brier_score` | Logistic only: mean `(pred - M0)²`; lower better |
| `log_loss` | Logistic only: cross-entropy; lower better |

Endpoints: `M0_AP`, `M0_prevalence`, `Mle1_AP`, `Mle1_prevalence`, `Mle2_AP`, `Mle2_prevalence`. Calibration blank for non-logistic scorers.

### `precision_recall.parquet`

**One row per distinct observed value** (development only). Contains four identifiers, all shared metrics, plus:

| Column | Definition |
|--------|------------|
| `threshold` | Score/distance at this point. EpiLink/logistic: `>=`, genetic: `<=` |
| `ties_at_threshold` | Pairs at this value |

Strict to permissive per scorer. No zero-selection row (recall = 0 implicit in AP). Uses observed values, not operating grid. Evaluation file empty.

Development `pairwise/evidence/` checkpoints curves, rankings, calibration, `empty_metrics.json`, manifest. `development/cutoff_candidates/definitions.json` maps IDs to pairwise/component definitions. Pipelines select own optimum. Cached candidates restored on resumption.

### Shared cutoff and adaptive-resolution search evidence

`development/seed_<seed>/clusters/component_sweeps/<scorer>/metrics.parquet`: shared cutoff metrics, `setting_id`, `partition_id`, `retained_graph_edges`. `algorithm.json`: candidate/partition counts. `partitions/<partition-id>/`: `memberships.parquet`, `clusters.parquet`, `metrics.json`. Manifest checksums states. Edges added in ties; unchanged partitions reuse counts. Selected settings get `clusters/<setting-id>/` artifacts.

`development/resolution_search/`: `definitions.json`, `search.json` (status, settings, `evaluated_resolutions`, criteria, seeds), manifest. Budget = common resolutions across scorers.

`development/treecluster_search/{raw,dated}/`: `definitions.json`, `search.json` (status, `tree_kind`, settings, `evaluated_cutoffs`, `units`, methods, criteria, seeds), manifest. Raw: integer SNPs; dated: fractional days. `threshold_input` in SNP/day units. Refinement resumes failed trials. `settings.json` combines all candidates.

### `calibration.csv`

**One row per bin per logistic scorer.** 10 equal-width bins.

| Column | Definition |
|--------|------------|
| `bin_lower`, `bin_upper` | Boundaries (last bin includes 1) |
| `n_pairs` | Count |
| `mean_probability` | Mean predicted prob |
| `observed_fraction` | Fraction with M=0 |

## 4. Partition and cluster outputs

Directory: `<run>/<split>/seed_<seed>/clusters/`.

### `metrics.csv` and `<setting-id>/metrics.json`

**One row/object per partition.** CSV has identifiers (`split`, `seed`, `setting_id`, `pipeline`, `data_process`); JSON has metrics only. Section 2 metrics (all within-cluster pairs) plus:

| Column | Definition |
|--------|------------|
| `n_cases` | Observed cases |
| `n_clusters` | Clusters (including singletons) |
| `n_singletons` | Singleton clusters/cases |
| `singleton_fraction` | `n_singletons / n_cases` |
| `largest_cluster` | Largest cluster size |
| `largest_cluster_fraction` | `largest_cluster / n_cases` |
| `within_pairs` | Sum of `size*(size-1)/2`; = `selected_pairs` |
| `cluster_mean_M0_precision` | Unweighted mean (non-singleton clusters); undefined if all-singleton |

Global `M0_precision` weights by pair counts; `cluster_mean_M0_precision` gives equal weight to non-singleton clusters.

### `<setting-id>/memberships.parquet`

**One row per case** (including isolates):

| Column | Definition |
|--------|------------|
| `case_id` | Case ID; join to `cases.parquet` |
| `cluster_id` | Partition-local integer; TreeCluster `-1` → distinct singletons |

### `<setting-id>/clusters.parquet`

**One row per cluster**:

| Column | Definition |
|--------|------------|
| `cluster_id` | Join key to membership |
| `n_cases` | Cases in cluster |
| `within_pairs` | `n_cases * (n_cases - 1) / 2` |
| `n_<category>` | Within-cluster count (10 categories) |
| `<endpoint>_precision` | Target pairs / `within_pairs`; undefined for singletons |

Counts sum to partition totals. No cluster-specific recall/F1.

### `<setting-id>/algorithm.json`

| Method | Fields |
|--------|--------|
| Components | `retained_graph_edges` |
| Leiden | `objective`, `quality`, `restart_qualities`, `seed`, `retained_graph_edges` |
| Empty-graph Leiden | `objective`, `quality: 0.0`, `retained_graph_edges: 0` |
| TreeCluster | `command`, `executable`, `threshold_tree_units` (subs/site for raw, days for dated) |

`quality` = clustering objective (not truth metric). Raw thresholds: subs/site. Dated thresholds: days.

## 4.2 Raw trees

IQ-TREE builds **raw tree** from FASTA (model from `phylogeny` config, default: JC). Includes reference tip; pruned to `raw_pruned.nwk` for TreeCluster. Branch lengths: subs/site. Input for `raw` clustering.

## 4.3 Dated trees

IQ-TREE builds **dated tree** with `--dated=True` and `sampling_dates.tsv`. LSD2 estimates branch lengths in **days** from earliest sample. Excludes reference tip. Input for `dated` clustering. Thresholds in days (no conversion).

## 5. Aggregated and held-out results

### `<run>/<split>/metrics.csv`

Concatenates per-seed pairwise and clustering metrics. Column union; `score_name` for pairwise, cluster rows use `settings.json` for scorer. Cluster-only metrics blank for pairwise. Partial runs may have partial coverage.

### `<run>/<split>/summary.csv`

**One row per `(pipeline, setting_id)`.** Numeric metrics expand to:

| Suffix | Meaning |
|--------|---------|
| `<metric>_mean` | Mean (equal realization weight) |
| `<metric>_std` | Sample SD (`ddof=1`); undefined if < 2 values |
| `<metric>_min`, `<metric>_max` | Min/max |

`n_realizations` = available rows. Missing metrics skipped. String fields omitted; recover from `settings.json`. Counts receive suffixes.

### `<run>/<split>/frontier.csv`

Non-dominated mean precision/recall per `(endpoint, pipeline)`, including ties. `endpoint` = `M0`, `Mle1`, `Mle2`. Column names retain `_mean` suffixes. One setting can appear at multiple endpoints. Omits missing precision/recall. Evaluation frontier covers evaluated settings only.

### `evaluation/operating_results.csv`

Evaluation metrics joined to `criterion` names. Same setting with two criteria → two rows. Group by criterion.

### `evaluation/operating_summary.csv`

**One row per `(criterion, pipeline)`.** From `selection_used.json`: `setting_id`, `objective`, `objective_endpoint`, `n_realizations`. Endpoint derived from objective (blank if none). One frozen setting, one row per seed.

Endpoints: precision/recall/F1/enrichment with `_mean`, `_std`, `_min`, `_max`, `_count`. Also: contamination, retention, workload, cluster sizes. `_count` = defined values; `n_realizations` = seeds. `_std` uses `ddof=1`. Cluster metrics undefined for pairwise.

`objective_*` = actual optimized metric summaries. Report shows optimized endpoint; CSV retains all endpoints.

## 6. Truth, observations, scores, and fitted models

Truth/observations: `<shared-root>/artifacts/`. Models/scores/trees: baseline `<root>/artifacts/`. Hash-named directories; use `experiment.json` and manifests for lineage. Perturbation has separate artifacts (section 12).

### `truth/<id>/nodes.parquet`

**One row per backbone node:**

| Column | Definition |
|--------|------------|
| `node_index` | Zero-based position |
| `case_id` | Node identifier (string) |

### `truth/<id>/relationships.parquet`

**One row per unordered pair** (including unobserved):

| Column | Definition |
|--------|------------|
| `pair_id` | Upper triangle position |
| `node_a`, `node_b` | Node indices (`node_a < node_b`) |
| `AD` | 1 = ancestor-descendant |
| `CA` | 1 = common ancestor (neither ancestral) |
| `m` | Intermediates on AD path; null outside AD |
| `m1`, `m2` | Intermediates on CA branches; null outside CA |
| `tree_hops` | `m+1` (AD), `m1+m2+2` (CA); null across components |
| `M` | `m` (AD), `m1+m2` (CA); null across components |

`pair_id = i * (2*N - i - 1) // 2 + j - i - 1` for `i < j`. Separate introductions: flags = 0, counts null.

### `observations/<id>/cases.parquet`

**One row per sampled case:**

| Column | Definition |
|--------|------------|
| `case_id` | String identifier |
| `node_index` | Full-tree index |
| `sample_date` | Sampling time (days, can be fractional) |
| `exposure_date` | Infection time (days) |

IQ-TREE/LSD2 uses original dates; scorer TD rounded. All rows sampled.

### `observations/<id>/pairs.parquet`

**One row per sampled pair**, sorted by `(a, b)`:

| Column | Definition |
|--------|------------|
| `pair_id` | Full-tree pair index (gaps after subsampling) |
| `a`, `b` | Row positions in `cases.parquet` (`a < b`) |
| `TD` | Sampling-time difference (whole days) |
| `GD_deterministic` | Deterministic Hamming distance |
| `GD_stochastic` | Stochastic Hamming distance |

### `scores/<id>/scores.parquet`

**One row per observed pair** (`pair_id` + scorer columns). All values finite.

| Column | Interpretation |
|--------|----------------|
| `EDD` | Deterministic obs; deterministic inference |
| `EDS` | Stochastic obs; deterministic inference |
| `ESD` | Deterministic obs; stochastic inference |
| `ESS` | Stochastic obs; stochastic inference |
| `GD_D`, `GD_S` | Hamming distances; lower better |
| `LOGIT_D`, `LOGIT_S` | M0 probabilities; higher better |

EpiLink: raw compatibility (can exceed 1). Match score artifact to observation artifact.

### `models/<id>/models.json`

Two objects (`deterministic`, `stochastic`):

| Field | Definition |
|-------|------------|
| `features` | `["GD", "TD"]` |
| `target` | `M0` |
| `mean`, `scale` | Standardization arrays |
| `coef`, `intercept` | Logistic coefficients |
| `C` | Inverse regularization |
| `training_pairs` | Training pairs |
| `training_prevalence` | M0 fraction |

Prediction: `sigmoid(((x - mean) / scale) @ coef + intercept)`. Manifest records training identities/seeds.

## 7. Setting definitions and frozen decisions

### `<run>/settings.json`

`setting_id` → method definition:

| Field | Meaning |
|-------|---------|
| `kind` | `pairwise`, `components`, `leiden`, `treecluster` |
| `pipeline`, `data_process` | Comparison group, genetic process |
| `score_name` | Scorer identifier (pairwise/graph) |
| `threshold` | Cutoff (pairwise/components), subs/site (raw TreeCluster), days (dated). Null for empty/Leiden |
| `empty` | Select no pairs |
| `graph_mode` | Leiden: `full` (all pairs). EpiLink/logistic: scores; genetic: unit weights |
| `objective`, `resolution` | Leiden objective/resolution |
| `restarts`, `algorithm_seed` | Leiden restarts/seed |
| `tree_kind` | TreeCluster: `raw` or `dated` |
| `method` | `max_clade`, `avg_clade`, `single_linkage` |
| `threshold_units` | `snps` (raw) or `days` (dated) |

Pipelines: `pairwise/<score>`, `components/<score>`, `leiden/<score>`, `treecluster/<process>/<kind>`. EpiLink/logistic have Leiden; genetic-distance excluded. TreeCluster method selected within pipeline.

### `selection/operating_points.json`

| Root field | Definition |
|------------|------------|
| `run_fingerprint` | Experiment SHA-256 |
| `training_fingerprint` | Logistic models SHA-256 |
| `development_evidence_sha256` | `development/metrics.csv` hash |
| `criteria` | Rules with `name`, `objective`, `constraints` |
| `operating_points` | One decision per pipeline/criterion |

| Field | Definition |
|-------|------------|
| `pipeline`, `criterion` | Comparison group, rule name |
| `rule` | Rule object |
| `status` | `selected` or `infeasible` |
| `setting_id` | Chosen key; null if infeasible |
| `development_objective_mean` | Equal-realization mean |
| `development_objective_sd` | Population SD (`ddof=0`); tie-breaker |
| `definition` | Method definition |

Last three fields absent for infeasible. Replaced decisions saved as `operating_points_<fingerprint>.json`.

`evaluation/selection_used.json`: frozen selection. `evaluation/heldout_access.json`: `seeds`, `selection_fingerprint`. Written before evaluation scoring. Shared ledger: `<shared-root>/heldout_access/seed_<seed>.json` (seed, experiment, selection). Survives resets; enforces fresh seeds.

## 8. Manifests, status, and provenance

### Pointer and run manifest

`<root>/current.json`: `run_directory` (absolute), `fingerprint`. Points to most recent run (including incomplete). Shared experiment: `experiment_directory`, `fingerprint`. `<run>/experiment.json` pins identity; `<run>/diagnostics.json` records completion.

`<run>/manifest.json`:

| Field | Definition |
|-------|------------|
| `status` | `running`, `complete`, `partial`, `failed` |
| `requested_stage` | Stage for this manifest |
| `signature` | `schema`, `config`, `implementation`, `tools`, `truth`, `experiment` |
| `git_revision` | Git HEAD (null if unavailable) |
| `config` | Resolved configuration |
| `run_directory` | Absolute path |
| `error` | Exception (if failed) |

`implementation`: SHA-256 mappings + `versions`. Tool identity: `path`, `sha256`, or `unavailable`. See [operational guide](OPERATIONS.md#9-resume-work-and-understand-caching) for run ID changes.

### Completed artifact manifests

Common fields:

| Field | Definition |
|-------|------------|
| `status` | `complete` |
| `signature` | Inputs/parameters |
| `fingerprint` | SHA-256 of signature (20-char prefix; cluster dirs use setting ID) |
| `files` | Filename → SHA-256 |
| `completed_at` | UTC timestamp |

Artifact signatures:

| Artifact | Signature |
|----------|-----------|
| Truth | `kind`, `source_sha256`, `nodes`, `edges`, `implementation`; `n_cases`, `n_pairs` |
| Observations | `kind`, `truth` ID, `generation`, `simulation`, `seed`, `implementation`; `truth_directory`, `n_cases`, `n_pairs`, `units`, FASTA, dates |
| Fitted models | `kind`, training `datasets`, `C`, `implementation`; `seeds` |
| Scores | `kind`, observation `dataset`, training fingerprint, scorers, inference, config |
| Per-seed pairwise | `run`, `score_id`, `split`, `seed`, `definitions` |
| Per-setting clustering | `run`, `score_id`, `definition`, `split`, `seed` |
| Trees | Dataset/process/tool/parameters (section 9) |

Observation `units`: `GD`, `TD`, `mutation_count_genome_length`, `simulated_sequence_length`. Scorer metadata: `name`, `family`, `data_process`, `inference_process`, `target`, `higher_is_better`. Genetic: `target: "none"`.

### Clustering status and failures

`clusters/status.json`:

| Field | Definition |
|-------|------------|
| `status` | `complete` or `partial` |
| `configured` | Settings requested |
| `completed` | Successful/cached results |
| `errors` | Failed records (split, seed, setting_id, pipeline, error) |
| `treecluster_enabled` | Phylogenetic comparisons enabled |

Failed setting `manifest.json`: `status: "failed"` + error. File presence ≠ valid cache; check signature/checksums. Run-level `complete` = requested stage.

### SCoVMod tree provenance

Prepared input: `evaluation/shared_synthetic/outputs/inputs/`. `manifest.json`: raw paths/hashes, `tree_seed`, `target_component_size`, tree/provenance paths, implementation. `files`: GML tree + source JSON. Matching artifacts reuse; changed signatures trigger reconstruction.

`<tree-stem>.source.json`: `inputs` (paths/hashes), `tree_sha256`, `n_cases`, `target_size`, `seed`, `implementation_sha256`, `tie_order`.

Baseline `inputs.json`: `experiment`, `diagnostics`, `tree_path`, `tree_sha256`, `n_cases`, `tree_seed`, `target_component_size`, `source` (backbone `source.json`).

## 9. Phylogenetic artifacts

Trees under `artifacts/trees/<id>/`. IQ-TREE infers ML trees; LSD2 dates; TreeCluster partitions.

### Synthetic studies

| File | Contents |
|------|----------|
| `raw.nwk` | IQ-TREE ML tree (reference-pruned). Branch lengths: subs/site |
| `dated.nwk` | LSD2 dated tree (no reference). Branch lengths: days |
| `node_dates.tsv` | `node`, `case_id`, `is_tip`, `date` (days), `sample_date` |
| `phylogeny.json` | `model`, `threads`, `seed`, `clock_rate`, `date_origin`, `reference_name`, `units`, `backend_paths` |
| `backend/run-*/` | IQ-TREE/LSD2 outputs: FASTA, dates, model report, trees, LSD2 report |
| `backend/run-*/inference.log` | IQ-TREE/LSD2 stdout/stderr |

**Signature:** `kind: phylogeny-v1`, `dataset`, `process`, FASTA/reference/dates hashes, `n_cases`, IQ-TREE settings, implementation.

**Manifest:** `units`, `rooting`, `root_split`, `dated_root_split`.

**Reference:** Raw tree includes reference as outgroup, pruned before TreeCluster. Dated tree excludes reference.

**Date units:** Days. Synthetic: numeric simulation origin. Boston: calendar dates normalized to days from earliest.

### Boston studies

| File | Notes |
|------|-------|
| `raw.nwk` | Full alignment (29,903 bp) + reference; IQ-TREE MFP |
| `dated.nwk` | LSD2 with real collection dates |
| `node_dates.tsv` | Includes `calendar_date` columns |
| `phylogeny.json` | `alignment_length: 29903`, `model: MFP` |

**Alignment:** 29,903 bp. Reference must be compatible; gap patterns must match.

**Threshold scaling:** SNP counts scaled to subs/site: `(snp_count × 5000) / 29903`. Days passed directly.

**Signature:** `kind: boston-phylogeny-v1`, alignment/reference hashes, `alignment_length`.

### TreeCluster partitions

Artifacts under `clusters/` or `trees/`:

| File | Contents |
|------|----------|
| `memberships.parquet` | `case_id`, `cluster_id` (-1 = singleton) |
| `clusters.parquet` | `cluster_id`, `n_cases`, `within_pairs`, exposure/mutation counts |
| `metrics.json` | `n_cases`, `n_clusters`, `size_mean`, `size_std`, `largest_cluster` |
| `algorithm.json` | Command, executable, `threshold_tree_units` |
| `treecluster.stdout.log` | TreeCluster output |
| `treecluster.stderr.log` | Stderr (including timeouts) |

**Threshold units:**

- Raw: `snps` → `snp_count / alignment_length` (subs/site)
- Dated: `days` (no conversion)

10 SNPs: synthetic (5,000 bp) = 0.002 subs/site; Boston (29,903 bp) = 0.000334 subs/site.

## 10. Boston inputs and results

`<root>`: `evaluation/03_boston_application/outputs/boston/`. Inputs: `outputs/inputs/` (stage `prepare`). Empirical data; no synthetic truth.

`epilink-evaluate boston --stage prepare`: uses Boston config, `inputs.cases_path`, `inputs.pairs_path`. Input paths independent of `--output`.

### `cases.parquet`

**One row per metadata/Nextclade case:**

| Column | Definition |
|--------|------------|
| `case_id` | Source `seq_id` (string); matches Nextclade `seqName` |
| `sample_date` | `collection_date` (calendar) |
| `Clade` | Nextclade clade |
| `substitutions` | Mutation list |
| `QC_OverallStatus` | QC status |
| `Exposure` | Derived: Conference, SNF, BHCHP, City, Unlabeled |
| `Mutation` | Derived: C2416T, G105T, G28899T, G3892T, C20099T, Minor Lineages |

Source fields pass through: demographics, accessions, geography, exposure flags, labs, submission metadata.

### `observed_pairs.parquet`

**One row per non-self unordered pair:**

| Column | Definition |
|--------|------------|
| `CaseID1`, `CaseID2` | Source IDs (string order) |
| `TN93_distance` | Substitutions/site |
| `GD` | `TN93_distance * 29903` |
| `TD` | Sample-date difference (days, no rounding) |

TN93 censored at 0.0005/site; missing pairs unobserved. Manifest: `n_cases`, `n_observed_pairs`, `n_all_pairs`, `candidate_universe`. Signature: `kind`, input hashes, `distance_cutoff_per_site`, `reference_length`.

### Boston empirical run outputs

Directory: `<root>/runs/<run-id>/` (stage `all` or `trees`).

Frozen transfer: `settings.json`, `selection.json`, `clusters/`, `trees/`, `assessment/` (baseline operating points). `clusters/status.json`, `trees/status.json`: configured/completed/failed. `assessment/summary.csv`: cluster-size/coverage summaries (no truth metrics).

`manifest.json`: stage/status. `inputs.json`: case/pair counts, hashes, `trees_enabled`. `trees` complete ≠ graph scoring done. `all`: frozen transfer. `report`: render saved tables.

#### Scores and settings

- `artifacts/scores/<id>/scores.parquet`: `CaseID1`, `CaseID2`, scorer columns (no `pair_id`). Frozen `all` records `score_id`.
- `settings.json`: adapted frozen definitions (`baseline_setting_id`, `baseline_score_name`, `baseline_data_process`). `data_process: empirical`, tree: `score_name: TREE`.
- `selection.json`: adapted baseline decisions (not Boston-based selection).
- Scorers use empirical GD/TD. Pipelines: `leiden/ESS`, `treecluster/empirical/<process>/<kind>`.

#### Partition artifacts

`clusters/<setting-id>/`, `trees/<setting-id>/`: `memberships.parquet`, `clusters.parquet`, `metrics.json`, `algorithm.json`, manifest. TreeCluster: stdout/stderr logs. Memberships: one row per case (`case_id`, `cluster_id`).

Per-cluster: `cluster_id`, `n_cases`, `within_pairs`, `n_<field>_<label>` counts (Exposure, Clade, Mutation). `within_pairs`: all co-clustered pairs.

`metrics.csv` (one row per setting):

| Metric | Definition |
|--------|------------|
| `n_cases`, `n_clusters` | Total cases/clusters (including singletons) |
| `size_mean`, `size_std` | Mean/SD of cluster sizes (SD=0 for one cluster) |
| `largest_cluster` | Largest cluster size |

#### Assessment tables

Under `assessment/`:

| Table | Row unit |
|-------|----------|
| `summary.csv` | Per partition: cluster counts, singletons, largest, coverage |
| `cluster_composition.csv` | Per cluster/field/label: `n_evidence`, `fraction_in_cluster` |
| `named_cluster_overlaps.csv` | Per setting/exposure: representative cluster overlap |
| `tree_agreement.csv` | Per graph/TreeCluster pair: ARI, AMI |
| `named_tree_overlaps.csv` | Representative exposure cluster overlaps |
| `cluster_overlaps.csv` | External comparator overlaps (if configured) |
| `best_cluster_overlaps.csv` | Best external group per cluster (if configured) |

`named_cluster_overlaps.csv`:

| Column | Meaning |
|--------|---------|
| `n_cases`, `n_exposure` | Cluster size, exposed cases |
| `exposure_total` | Total exposed in Boston |
| `exposure_fraction` | `n_exposure / n_cases` (concentration) |
| `exposure_recovery` | `n_exposure / exposure_total` (one cluster only) |
| `treecluster_*`, `jaccard` | External comparator (if supplied) |

`shared = |A ∩ B|`, `jaccard = |A ∩ B| / |A ∪ B|`. Overlap % = `shared / size * 100`.

**Fold enrichment:** `exposure_fraction / (exposure_total / total_cases)` (not saved; interpret with `exposure_recovery`).

`candidate_coverage = n_observed_pairs / n_all_pairs`. Trees use full alignment.

#### Boston tree artifacts

`raw.nwk`, `dated.nwk`, `node_dates.tsv`, `phylogeny.json`, `sampling_dates.csv`, `backend/run-*/`. Aligned samples + reference. Raw: subs/site; dated: days. `trees/inputs.json`: paths/hashes. Run `inputs.json`: reference path/hash, alignment length.

Raw TreeCluster transfer: SNP count from baseline → subs/site using Boston length. Adapted definitions: `baseline_threshold`, `threshold_snps`, `baseline_setting_id`. Dated cutoffs unchanged.

## 11. Worked joins in Python

Examples from repository root. Require completed development. Start with smoke output; change `root` for full experiment.

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
summary = pd.read_csv(run / "development/summary.csv", dtype={"setting_id": str})
annotated = summary.merge(definitions, on=["pipeline", "setting_id"], validate="many_to_one")
```

### Follow lineage before joining pairs to truth and scores

Seed alone insufficient; follow manifests. Models/scores baseline-local; observations/truth in pinned shared experiment.

```python
from epilink_evaluation.inputs.synthetic import analysis_table, load_observations, load_truth

run_manifest = json.loads((run / "manifest.json").read_text())
experiment = json.loads((run / "experiment.json").read_text())
experiment_dir = Path(experiment["experiment_directory"])
shared_root = experiment_dir.parent.parent
seed = run_manifest["config"]["splits"]["development"][0]
seed_dir = run / "development" / f"seed_{seed}"
pairwise = json.loads((seed_dir / "pairwise/manifest.json").read_text())
score_dir = root / "artifacts/scores" / pairwise["signature"]["score_id"]
score_manifest = json.loads((score_dir / "manifest.json").read_text())
observation_dir = shared_root / "artifacts/observations" / score_manifest["signature"]["dataset"]
observation_manifest = json.loads((observation_dir / "manifest.json").read_text())
truth_dir = shared_root / "artifacts/truth" / observation_manifest["signature"]["truth"]
observations, cases = load_observations(observation_dir)
truth = load_truth(truth_dir, observations.pair_id)
scores = pd.read_parquet(score_dir / "scores.parquet")
joined = analysis_table(observations, cases, truth, scores)
```

`analysis_table` joins on `pair_id` (one-to-one), adds `CaseID1`, `CaseID2`. Full runs: millions of rows.

### Attach one partition to sampled cases

```python
partitions = pd.read_csv(seed_dir / "clusters/metrics.csv", dtype={"setting_id": str})
setting_id = partitions.iloc[0]["setting_id"]
membership_path = seed_dir / "clusters" / setting_id / "memberships.parquet"
memberships = pd.read_parquet(membership_path)
case_clusters = cases.merge(memberships, on="case_id", validate="one_to_one")
```

Same `cluster_id` joins `clusters.parquet`. Keep run/split/seed/setting identity.

## 12. Perturbation study outputs

`<root>`: `evaluation/02_synthetic_perturbation/outputs/perturbation/` or `perturbation_smoke/`. Follows baseline `current.json`/`runs/<id>/` convention.

### Identity, scenarios, and coverage

| File | Purpose |
|------|---------|
| `manifest.json` | `status`, `requested_stage`, `config`, `signature`, `n_cases`, `run_directory` |
| `reference.json` | Source run, fingerprints, `n_cases`, `baseline_implementation` |
| `selection.json` | Baseline selection (EpiLink pipelines/criterion) |
| `settings.json` | Frozen Leiden definitions (`graph_mode: full`, `threshold: null`) |
| `scenarios.json` | `name`, `parameter`, `value`, `baseline_value`, `multiplier`, `generation` |
| `coverage.csv` | Per scenario/mode: `inference_mode`, `clustering_mode`, `status`, `completed`, `expected` |

Study signature: configuration, reference, scenarios, implementation, truth. `scenarios/<scenario>/<mode>/manifest.json`: status, config, error. Failed/not-run arms: no results. Complete: all four arms per scenario.

### Absolute clustering results

`results.csv`: section 4 partition metrics + baseline criterion. Metadata:

| Column | Definition |
|--------|------------|
| `scenario` | Scenario name (`baseline` = control) |
| `mode` | One of four inference/clustering arms |
| `inference_mode` | `baseline` or `matched` |
| `clustering_mode` | `baseline` or `updated` |
| `parameter` | Changed field (blank for controls) |
| `value` | Perturbed value |
| `baseline_value` | Original value |
| `multiplier` | Relative multiplier (blank for absolute/controls) |

`split: evaluation`, fresh seeds. `pipeline`: `leiden/<scorer>` (EDD/EDS/ESD/ESS). `setting_id`: actual resolution. Coverage: 624 rows (full), 48 (smoke).

### Paired deltas

`results_deltas.csv`: perturbed rows only +

| Column | Meaning |
|--------|---------|
| `baseline_<metric>` | Control metric (same seed, mode, pipeline) |
| `delta_<metric>` | `metric - baseline_<metric>` |
| `control_available` | Matching control exists |
| `baseline_setting_id` | Control's setting ID |

Controls match on `(mode, seed, criterion, pipeline)`, not setting ID. Missing controls: left join with missing metrics. Negative ΔF1 = reduced recovery; positive Δcontamination = more distant pairs. Fresh controls (not reference's old observations).

### Summary tables

| File | Content |
|------|---------|
| `results_summary.csv` | Absolute metrics by `(scenario, mode, criterion, pipeline, setting_id)` |
| `results_delta_summary.csv` | Delta metrics (no controls) |

Suffixes: `_mean`, `_std`, `_min`, `_max`, `_count`. Mean: equal weight per nonmissing realization. SD: `ddof=1`. `_count`: nonmissing values. `n_realizations`: group rows; delta: also `n_controls`.

### Detailed artifacts and reports

`scenarios/<scenario>/<mode>/selection.json`: operating points, modes, seeds, evidence checksum. `settings.json`: replayed definitions. `evaluation/heldout_access.json`: selection fingerprint, seeds.

Updated arms: `development/metrics.csv`, summaries, per-setting artifacts. All arms: `evaluation/seed_<seed>/clusters/<setting-id>/` (memberships, clusters, metrics, algorithm, manifest). Full graphs: all edges. Shared observations/scores per scenario/inference.

`artifacts/backbones/<id>/`: frozen tree. Truth/observations/scores: study's `artifacts/`. Scores: `training: null`. No models/phylogeny.

`report.md`/`report.html`: coverage, levels, resolutions, performance, deltas. Smoke: pipeline validation.

## 13. Shared experiments and diagnostics

### Shared experiment identity and access

Shared config: `inputs`, `generation`, `simulation`, `splits`. Diagnostics/baseline reference via `experiment_config`. Root: `evaluation/shared_synthetic/outputs/synthetic/` or `synthetic_smoke/`.

| Path | Contract |
|------|----------|
| `current.json` | Latest `experiment_directory`, `fingerprint` |
| `experiments/<id>/experiment.json` | Data design, schema, `output_directory`, `inputs.tree_path` |
| `experiments/<id>/manifest.json` | `kind`, `specification`, `producer`, `backbone`, `truth` IDs |
| `artifacts/backbones/<id>/` | `transmission_tree.gml`, `source.json`, manifest |
| `artifacts/truth/<id>/`, `observations/<id>/` | Section 6 schemas |
| `experiments/<id>/observations/seed_<seed>.json` | `seed`, `role`, `dataset`, `fingerprint` |
| `experiments/<id>/diagnostics.json` | Completed diagnostics reference |
| `heldout_access/seed_<seed>.json` | `seed`, `experiment`, `selection_fingerprint` |
| `validation_access/<fingerprint>/seed_<seed>.json` | Smoke validation access |

Baseline pins shared identity in `<run>/experiment.json`; validates diagnostics. Follow pinned identity (not latest pointer). `reset-outputs` preserves shared area. Accessed seeds cannot be reassigned. Complete diagnostics before baseline.

### Diagnostics layout and completion

`<diagnostics-root>`: `evaluation/00_synthetic_diagnostics/outputs/diagnostics/` or `diagnostics_smoke/`. `<diagnostics-run>`: `runs/<fingerprint>/`.

| Path | Contents |
|------|----------|
| `<diagnostics-root>/current.json` | `run_directory`, `fingerprint`, `experiment` |
| `<diagnostics-run>/manifest.json` | `signature`, `status`, `stage`, `config`, `coverage_complete` |
| `<diagnostics-run>/<stage>/index.json` | `status`, `records`, `errors`, seed→dataset map |
| `<diagnostics-root>/artifacts/<kind>/<id>/` | Diagnostic tables/controls with manifests |
| `<diagnostics-run>/completion/` | `coverage.json`, manifest (all stages validated) |
| `<diagnostics-run>/report.md`, `report.html`, `figures/` | Reports with coverage, errors, summaries |

Completion: `experiment`, `diagnostics`, `datasets`. `coverage.json`: `status`, `stages` (`backbone`, `observations`, `graphs`), `datasets`, `artifacts` (dir→SHA256, files). Baseline validates inventory. Complete `prepare` or stage ≠ full coverage. Failed graphs prevent completion.

### Fixed transmission-backbone characterisation

`<diagnostics-run>/backbone/index.json`: `scope: "one fixed backbone"`, one record (backbone ID, artifact path). `backbone/summary.csv`: one row (not per seed).

`<diagnostics-root>/artifacts/backbone/<id>/`:

| File | Contents |
|------|----------|
| `nodes.parquet` | `node_index`, `case_id`, `root_case_id`, `depth`, `offspring`, `descendant_count`, `is_superspreader` |
| `offspring.csv` | Per offspring count: `n_cases`, `case_fraction`, Poisson/NB probabilities, `qualifies_superspreading` |
| `concentration.csv` | Rank, `case_fraction`, `cumulative_transmissions`, `transmission_fraction` |
| `generations.csv` | Per depth: `n_cases`, `case_fraction` |
| `components.csv` | Per `root_case_id`: `n_cases`, `n_transmissions`, `max_depth` |
| `bootstrap.csv` | Per replicate: `mean_offspring`, `dispersion_k`, `fit_method` |
| `summary.json` | Scalar summaries + bootstrap metadata |
| `provenance.json` | Backbone identity, superspreading rule |

`summary.json` counts: `n_cases`, `n_transmissions`, `max_offspring`, `n_zero_offspring`, `n_roots`, `n_components`, `n_superspreaders`, etc.

Fractions denominators:
- Offspring/component fractions: all backbone cases
- Transmission fractions: all direct edges
- `M0_prevalence`: all unordered pairs

`mean_offspring = n_transmissions / n_cases`. Variance: `ddof=0` (population), `ddof=1` (sample). Mean offspring = `(N - C) / N` (C = introductions).

`dispersion_k`: NB fit (variance `R + R²/k`). `fit_method`: `mle`, `moments_fallback`, `poisson_limit`, `degenerate`, `insufficient_cases`.

Superspreading: **inclusive** (`offspring >= poisson_percentile`, default quantile 0.99). No transmissions: no superspreaders flagged.

Bootstrap: `requested`, `completed`, `seed`, `interpretation`, `*_kept`, `*_interval95`. Intervals: 2.5th/97.5th percentiles. Disabled by default.

`fig01_backbone_characterisation`: distributions, tables, summary CSV/JSON, caption.

### Exact observation feature cells

`artifacts/observations/<id>/` (diagnostics root): diagnostic tables (distinct from shared raw observations).

`cells.parquet`: **one row per occupied cell/process/endpoint/seed**. No rounding/binning.

| Column | Definition |
|--------|------------|
| `seed`, `process`, `feature_set`, `endpoint` | Seed; `deterministic`/`stochastic`; `GD`/`GD_TD`; `M0`/`Mle1`/`Mle2` |
| `GD`, `TD` | Exact values (subs, days); `TD` null for GD-only |
| `n_pairs`, `n_target`, `n_other` | Occupancy, positives, negatives |
| `target_fraction`, `mixed` | `n_target / n_pairs`; both classes present |
| `n_<category>` | Ten relationship counts (sum = occupancy) |

`summary.csv`: **one row per seed/process/feature_set/endpoint**. `n_pairs`, `n_target`, `n_other` (universe) + `n_cells`, `mixed_cells`, pairs/targets/others in mixed cells.

| Column | Definition |
|--------|------------|
| `mixed_cell_fraction` | Mixed / occupied cells |
| `pair_fraction_in_mixed_cells` | Pairs in mixed / all pairs |
| `target_fraction_in_mixed_cells` | Targets in mixed / all targets |
| `target_prevalence_in_mixed_cells` | Targets in mixed / pairs in mixed |
| `non_target_fraction_in_mixed_cells` | Non-targets in mixed / all non-targets |
| `class_conditional_overlap` | Sum `min(n_target/total_targets, n_other/total_others)` |
| `minimum_feature_only_misclassifications` | Sum `min(n_target_cell, n_other_cell)` |
| `minimum_feature_only_misclassification_rate` | Min errors / all pairs |

Zero denominators: undefined. Minimum error: empirical (not population ceiling).

`prevalence.csv`: per seed/endpoint (`n_pairs`, `n_target`, `n_other`, `target_prevalence`). `relationships.csv`: per seed/relationship (`n_pairs`, `pair_fraction`). No process/feature-set replication.

Run-level `observations/`: concatenates per-seed tables + `*_aggregate.csv` (`_mean`, `_min`, `_max`, `_count`, `n_seeds`).

### Endpoint-oracle graph controls

One graph per horizon (h=0,1,2) per sampled-case set. Edges: M≤h pairs, unit weight. Vertices include isolates.

- `artifacts/graphs/<id>/cases.parquet`: `case_id`, `node_index` (lexical order). `edges.parquet`: `a`, `b` (positions), `weight: 1`. `provenance.json`: truth, sample, endpoint, horizon, rule.
- `summary.json`: `n_cases`, `n_target_edges`, `n_components`, `n_isolates`, `largest_component`, `n_wedges`.
- `artifacts/graph_partitions/<id>/`: `memberships.parquet`, `clusters.parquet`, `metrics.json` (section 4), `algorithm.json`, `provenance.json`.

`graphs/metrics.csv`: per `(seed, endpoint, algorithm, resolution)`. Components: null resolution. Metrics for **all** endpoints/pairs (including nonedges). `graphs/summary.csv`: aggregates (`_mean`, `_min`, `_max`, `_count`, `n_seeds`). `graph_summary.csv`: per seed/endpoint (structure). `graph_summary_aggregate.csv`: by endpoint.

Graphs keyed by truth + sampled-case set (independent of seed/GD/TD/process). Full sampling: one graph per horizon (reused across seeds/processes). Stage tables repeat for equal-seed summaries (not independent replicates).

### Diagnostic figures

- `feature_cells_deterministic.png`, `feature_cells_stochastic.png`: GD_TD target fraction/occupancy (lowest seed, single realization)
- `backbone_characterisation.png`: offspring fits, superspreading cutoff, concentration, generations
- `oracle_graph_precision_recall.png`: precision/recall (components/Leiden) by endpoint

Figures in `figures/`. Reports show coverage/errors. Inspect coverage before interpreting.
