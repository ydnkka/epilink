# Boston empirical application

Empirical clustering analysis of SARS-CoV-2 sequences from Boston outbreaks (March-May 2020), using frozen operating settings from the synthetic baseline evaluation.

## Overview

This workflow applies EpiLink compatibility scores, genetic distances, and logistic probabilities to observed Boston TN93 distances. All scoring rules and graph clustering thresholds are frozen from a completed synthetic baseline—nothing is refitted or reselected. The workflow:

Do synthetic-selected operating points transfer to real Boston data?

1. **Prepares inputs** from raw Boston metadata, Nextclade results, and TN93 distances
2. **Scores observed pairs** using eight frozen inference rules (EDD, EDS, ESD, ESS, GD_S, GD_D, LOGIT_S, LOGIT_D)
3. **Clusters graphs** at frozen thresholds using components and Leiden algorithms
4. **Builds phylogenetic trees** from the full Boston alignment (independent of the distance-censored TN93 table)
5. **Runs TreeCluster** on raw and dated trees at frozen thresholds
6. **Optionally explores graph and TreeCluster threshold/resolution grids** as descriptive sensitivity analysis
7. **Compares partitions** with exposure metadata and new TreeCluster results

## Quick start

```bash
# From the repository root, with the environment activated:
python -m pip install -e '.[test]'

# Prepare Boston input tables (cases.parquet, observed_pairs.parquet)
python -m boston_application.run --stage prepare

# Build trees from the Boston alignment and run TreeCluster
python -m boston_application.run --stage trees

# Run the full empirical analysis (scoring, clustering, trees, assessment)
python -m boston_application.run --stage all

# Run descriptive threshold/resolution sweeps without changing the frozen result
python -m boston_application.run --stage explore

# Re-render the report from saved results
python -m boston_application.run --stage report
```

The equivalent installed CLI command is `epilink-evaluate boston --config boston_application/config.yaml --stage all`.

## Configuration

Edit [`config.yaml`](config.yaml) to change:

| Field | Meaning |
| --- | --- |
| `baseline_run` | Completed synthetic baseline run directory or `current.json` pointer. |
| `output_directory` | Boston output root (separate from baseline outputs). |
| `inputs.data_root` | Root containing `raw/boston/` source files. |
| `scorers` | Subset of EDD, EDS, ESD, ESS, GD_S, GD_D, LOGIT_S, LOGIT_D. Aliases ES→ESS and ED→EDS are accepted but cannot be combined. |
| `assessment.treecluster_path` | Optional external TreeCluster partition for comparison (TSV with `SequenceName` and `ClusterNumber` columns; `-1` denotes singletons). Default is `null`. |
| `assessment.focus_exposures` | Exposure labels for named-cluster summaries (default: Conference, SNF). |
| `assessment.min_cluster_size` | Minimum cluster size for focus-cluster analysis (default: 2). |
| `trees.enabled` | Whether to build raw/dated trees and run TreeCluster (default: true). |
| `trees.alignment_path` | Boston FASTA alignment for tree building (required when `trees.enabled` is true). |
| `trees.tn93_executable` | Optional path to `tn93` for all-pair distances (default: `tn93` on PATH). |
| `exploration` | Optional descriptive grid for `--stage explore`. It can set scorer subsets, graph thresholds/resolutions, and TreeCluster methods/thresholds. |

The baseline's EpiLink inference parameters, Monte Carlo settings, clustering thresholds, Leiden resolutions/objectives/restarts, and TreeCluster methods/thresholds are all frozen from the reference.

## Inputs

Source files under `data/raw/boston/`:

| File | Role |
| --- | --- |
| `MGH_DPH_98percent_772samples_metadata.csv` | Case metadata with collection dates and exposure flags. |
| `MGH_DPH_98percent_772samples_nextclade.tsv` | Nextclade clade assignments and substitutions. |
| `MGH_DPH_98percent_772samples_tn93_distances.csv` | Pairwise TN93 distances (censored at 0.0005/site). |
| `MGH_DPH_98percent_772samples_aligned.fasta` | Aligned sequences for tree building (uncensored). |

The `prepare` stage derives `cases.parquet` and `observed_pairs.parquet` with provenance. The TN93 pair table is **distance-censored**: missing pairs are unobserved, not zero distance. The candidate universe is explicit in the manifest.

## Outputs

Default root: `boston_application/outputs/boston/`

```text
boston/
  current.json                          # Latest run pointer
  boston_inputs/                        # Prepared input tables and manifest
  artifacts/
    scores/<id>/                        # Pairwise compatibility scores
    trees/<id>/                         # Raw and dated tree artifacts
  runs/<fingerprint>/
    manifest.json                       # Run status, config, implementation hashes
    reference.json                      # Baseline run identity
    selection.json                      # Frozen operating points used
    settings.json                       # Method definitions by setting ID
    inputs.json                         # Input paths and checksums
    scores/                             # Score artifacts and provenance
    clusters/                           # Graph clustering results by setting
      metrics.csv, status.json
    trees/                              # TreeCluster results by setting
      metrics.csv, status.json, inputs.json
    assessment/                         # Empirical evidence summaries
      summary.csv                       # Partition sizes and coverage
      cluster_composition.csv           # Exposure/clade/mutation counts per cluster
      named_cluster_overlaps.csv        # Named exposure group summaries
      best_cluster_overlaps.csv         # Best overlap per focus cluster
      tree_agreement.csv                # ARI/AMI between graph and tree partitions
      named_tree_overlaps.csv           # Named exposure overlaps with tree clusters
    exploration/                        # Optional --stage explore sensitivity sweep
      settings.json, setting_metadata.csv
      clusters/, trees/, assessment/
    figures/                            # Cluster size and overlap plots
    report.md, report.html              # Human-readable summary
```

## Interpretation

- **Synthetic D/S labels** (e.g., EDS, GD_S, LOGIT_D) identify the source operating rules or fitted classifiers from the baseline, not separate Boston measurements. All scorers use the same observed Boston GD and TD vectors.
- **LOGIT_S/LOGIT_D** reuse the baseline's fitted logistic classifiers without retraining.
- **Raw trees** are built from uncensored TN93 all-pair distances (FastME, midpoint rooted). **Dated trees** use TreeTime with real collection dates.
- **TreeCluster partitions** on raw trees often collapse into few clusters; dated trees typically yield finer structure due to temporal signal.
- **Exploration outputs** are descriptive stability checks across thresholds/resolutions. They are not Boston-selected operating points because complete transmission truth is unavailable.
- **Exposure labels and TreeCluster agreement** are descriptive external evidence, not transmission truth. The Boston dataset lacks complete transmission links for precision/recall evaluation.
- **Candidate coverage** is less than 1.0 because the TN93 source is distance-censored; missing pairs are not imputed.

## Method Scope

The workflow:
- Uses **all eight baseline-frozen scorers** with their selected operating points
- Builds trees from the Boston alignment
- Adds an optional **exploratory threshold/resolution sweep** reported separately from the frozen transfer analysis
- Reports **partition agreement metrics** (Adjusted Rand, Adjusted Mutual Information) between graph and tree clusters

## Troubleshooting

| Symptom | Check / Next action |
| --- | --- |
| `Executable not found: tn93` | Install TN93 (`conda install -c bioconda tn93`) or set `trees.tn93_executable` in config. |
| `Reference has no selected ... TreeCluster settings` | The baseline must have completed `--stage evaluate` with TreeCluster enabled and selected settings. |
| `Boston scorers must be a unique nonempty subset` | Use valid scorer names; ES/ESS and ED/EDS are aliases and cannot be combined. |
| `Scientific implementation/dependencies differ from baseline` | The Boston adapter (`inputs/boston.py`) is excluded from baseline checks. Other scientific module changes require a fresh baseline. |
| `FastME exited 1; invalid distance matrix` | Ensure the PHYLIP distance matrix uses fixed-point format (not scientific notation). |

## Scientific notes

The Boston empirical application demonstrates how frozen operating settings from a synthetic baseline transfer to real outbreak data. It does **not** claim transmission truth recovery, as complete epidemiological links are unavailable. The analysis is descriptive: it reports cluster sizes, exposure composition, and method agreement, without asserting correctness.

For sensitivity to natural-history parameters, use the **perturbation workflow** on synthetic data. Adaptation (retraining or retuning) under changed parameters is a separate analysis not performed here.

## See also

- [Operational guide](../OPERATIONS.md) for setup, configuration, and troubleshooting
- [Synthetic baseline protocol](../synthetic_baseline/README.md) for metric definitions and operating criteria
- [Perturbation study](../synthetic_perturbation/README.md) for parameter sensitivity with frozen settings
- [Output reference](../OUTPUTS.md) for column definitions and worked joins
