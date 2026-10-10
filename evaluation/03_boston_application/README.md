# Boston empirical application

Empirical clustering analysis of SARS-CoV-2 sequences from Boston outbreaks (March-May 2020), using frozen operating settings from the synthetic baseline evaluation.

## Purpose and study sequence

**Main question:** Do methods selected on synthetic data identify clusters concentrated for the Conference and SNF exposures in Boston, how much of each exposure group do they recover, and how do their partitions compare with TreeCluster partitions?

This is the **empirical transfer and descriptive application study**. It applies EpiLink compatibility scores, genetic distances, and baseline-fitted logistic models to observed Boston genetic distances and sampling dates. It also compares graph partitions with TreeCluster partitions built from the Boston alignment.

The study has two objectives:

1. **Assess transfer of frozen settings:** apply the synthetic baseline's selected graph and TreeCluster settings directly to empirical observations.
2. **Characterize epidemiological coherence:** describe exposure concentration, the fraction of each focus-exposure group captured, cluster sizes and singletons, and agreement between graph and phylogenetic partitions.

These questions are addressed by the frozen-transfer analysis:

| Analysis                                    | Settings                                                     | Evidence                                                                    |
| ------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------- |
| **Frozen transfer** (`--stage all`) | Baseline-selected operating points applied directly.         | Empirical partitions and focus-exposure summaries at prespecified settings. |

**Evidence produced:** cluster memberships, exposure composition and recovery, cluster-size summaries, and partition agreement. Exposure concentration and recovery should be interpreted together: a small pure cluster can capture few exposed cases, while a very large cluster can capture many with little concentration. Complete transmission truth is unavailable, so these summaries describe external epidemiological evidence rather than transmission accuracy.

**Role in the study sequence:** the [synthetic baseline](../01_synthetic_baseline/README.md) compares methods against known truth and supplies the primary operating points. The [perturbation study](../02_synthetic_perturbation/README.md) examines biological parameter changes and inference mismatch in simulation. Boston uses the baseline reference directly for real-data application, reporting exposure concentration, recovery, and agreement between graph and phylogenetic partitions.

## Workflow overview

1. **Prepares inputs** from raw Boston metadata, Nextclade results, and TN93 distances
2. **Scores observed pairs** using eight frozen inference rules (EDD, EDS, ESD, ESS, GD_S, GD_D, LOGIT_S, LOGIT_D)
3. **Clusters graphs** at frozen thresholds using components and Leiden algorithms
4. **Builds phylogenetic trees** from the full Boston alignment (independent of the distance-censored TN93 table)
5. **Runs TreeCluster** on raw and dated trees at frozen thresholds
6. **Compares partitions** with exposure metadata and TreeCluster results

## Quick start

```bash
# From the repository root, with the environment activated:
python -m pip install -e '.[test]'

# Prepare Boston input tables (cases.parquet, observed_pairs.parquet)
python evaluation/03_boston_application/run.py --stage prepare
# Equivalent installed command:
epilink-evaluate boston --stage prepare --config evaluation/03_boston_application/config.yaml

# Build trees from the Boston alignment and run TreeCluster
python evaluation/03_boston_application/run.py --stage trees

# Run the full empirical analysis (scoring, clustering, trees, assessment)
python evaluation/03_boston_application/run.py --stage all

# Re-render the report from saved results
python evaluation/03_boston_application/run.py --stage report
```

The equivalent installed CLI command is `epilink-evaluate boston --config evaluation/03_boston_application/config.yaml --stage all`.

`--stage all` runs frozen scoring, graph clustering, enabled TreeCluster, and assessment. `--stage trees` only builds/reuses trees and applies the frozen TreeCluster rules. Boston does not support `--smoke`. Input preparation can run before baseline evaluation; computational stages need the baseline's frozen selection and held-out evaluation provenance.

## Configuration

Edit [`config.yaml`](config.yaml) to change:

| Field                                        | Meaning                                                                                                                                                          |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `baseline_run`                             | Completed synthetic baseline run directory or`current.json` pointer.                                                                                           |
| `output_directory`                         | Boston output root (separate from baseline outputs).                                                                                                             |
| `inputs.data_root`                         | Root containing`raw/boston/` source files.                                                                                                                     |
| `inputs.cases_path`, `inputs.pairs_path` | Prepared tables, defaulting to`outputs/inputs/` beside the config. Paths are independent of the run output root.                                               |
| `scorers`                                  | Subset of EDD, EDS, ESD, ESS, GD_S, GD_D, LOGIT_S, LOGIT_D. Aliases ES→ESS and ED→EDS are accepted but cannot be combined.                                     |
| `assessment.treecluster_path`              | Optional external TreeCluster partition for comparison (TSV with`SequenceName` and `ClusterNumber` columns; `-1` denotes singletons). Default is `null`. |
| `assessment.focus_exposures`               | Exposure labels for named-cluster summaries (default: Conference, SNF).                                                                                          |
| `assessment.min_cluster_size`              | Minimum cluster size for focus-cluster analysis (default: 2).                                                                                                    |
| `trees.enabled`                            | Whether to build raw/dated trees and run TreeCluster (default: true).                                                                                            |
| `trees.alignment_path`                     | Boston FASTA alignment for tree building (required when`trees.enabled` is true).                                                                               |
| `trees.tn93_executable`                    | Optional path to`tn93` for all-pair distances (default: `tn93` on PATH).                                                                                     |

For the frozen-transfer analysis, EpiLink inference parameters, Monte Carlo settings, fitted logistic models, graph thresholds, Leiden settings, and TreeCluster methods/thresholds come from the baseline reference.

## Inputs

Source files under `data/raw/boston/`:

| File                                                | Role                                                    |
| --------------------------------------------------- | ------------------------------------------------------- |
| `MGH_DPH_98percent_772samples_metadata.csv`       | Case metadata with collection dates and exposure flags. |
| `MGH_DPH_98percent_772samples_nextclade.tsv`      | Nextclade clade assignments and substitutions.          |
| `MGH_DPH_98percent_772samples_tn93_distances.csv` | Pairwise TN93 distances (censored at 0.0005/site).      |
| `MGH_DPH_98percent_772samples_aligned.fasta`      | Aligned sequences for tree building (uncensored).       |

The `prepare` stage (`epilink-evaluate boston --stage prepare`) derives `cases.parquet` and `observed_pairs.parquet` with provenance under `evaluation/03_boston_application/outputs/inputs/`. Computational stages automatically use the same preparation and reuse matching artifacts. `--output` changes the run root, not these shared input paths. The TN93 pair table is **distance-censored**: missing pairs are unobserved, not zero distance. The candidate universe is explicit in the manifest.

## Outputs

- Run root: `evaluation/03_boston_application/outputs/boston/`
- Shared inputs: `evaluation/03_boston_application/outputs/inputs/`

The shared input directory contains `cases.parquet`, `observed_pairs.parquet`, and `manifest.json`. The run root contains:

```text
boston/
  current.json                          # Latest run pointer
  artifacts/
    scores/<id>/                        # Pairwise compatibility scores
    trees/<id>/                         # Raw and dated tree artifacts
  runs/<fingerprint>/
    manifest.json                       # Run status, config, implementation hashes
    reference.json                      # Baseline run identity
    selection.json                      # Frozen operating points used
    settings.json                       # Method definitions by setting ID
    inputs.json                         # Input paths and checksums
    clusters/                           # Graph clustering results by setting
      metrics.csv, status.json
    trees/                              # TreeCluster results by setting
      metrics.csv, status.json, inputs.json
    assessment/                         # Empirical evidence summaries
      summary.csv                       # Partition sizes and coverage
      cluster_composition.csv           # Exposure/clade/mutation counts per cluster
      named_cluster_overlaps.csv        # Named exposure group summaries
      best_cluster_overlaps.csv         # Optional external-comparator overlaps
      tree_agreement.csv                # ARI/AMI between graph and tree partitions
      named_tree_overlaps.csv           # Named exposure overlaps with tree clusters
    figures/                            # Cluster size and overlap plots
    report.md, report.html              # Human-readable summary
```

## Manuscript displays from frozen results

Generate the main exposure-composition table and two figures, then supplementary figures covering every frozen partition, from saved results alone:

```bash
python -m evaluation.results.tab04  # frozen exposures
python -m evaluation.results.fig18  # exposure trade-offs
python -m evaluation.results.fig19  # partition context
python -m evaluation.results.fig20  # all-exposures supplement
python -m evaluation.results.fig21  # all-agreement supplement
```

Scripts default to `outputs/boston/current.json` and write into `evaluation/results/outputs/03_boston_application/<run-id>/`. Each accepts `--run-dir <run>` and `--output-dir <directory>`; the figure scripts accept `--format pdf|png|both` (default `both`). The two supplementary scripts also accept `--criterion balanced_M0|balanced_Mle1|balanced_Mle2|all` (default `all`). See the [numbered results index](../results/README.md) for the display mapping. They validate the pinned reference, setting IDs, graph/tree completion and assessment coverage without changing any producer artifacts. The current saved run has complete `all` report and graph/tree checkpoints but a later stale `running`/`explore` manifest entry; the scripts warn when consuming those independently checked results. Pin a finalized run for manuscript provenance.

The **main-text focus** is the original baseline-frozen `balanced_M0` settings for ESD native Leiden, LGD native Leiden, GDD binary Leiden, and the deterministic-source raw and dated TreeCluster definitions. D/S in a scorer or TreeCluster setting names the *synthetic source rule*, not an alternative Boston distance measurement. The tree comparator uses uncensored distances rebuilt from the alignment; graph inputs use the distance-censored TN93 pair table. No exposure-based threshold or resolution selection is performed.

| Output | Manuscript role |
| --- | --- |
| `tab04_boston_frozen_exposures.tex` | Main table: for Conference and SNF, exposed counts/denominators, representative-cluster size, concentration (`n_exposure / n_cases`), recovery (`n_exposure / exposure_total`), number of clusters, singleton cases and largest cluster. Generated with the shared `utils/latex_tables.py` manuscript table environment. |
| `fig18_boston_exposure_tradeoffs.pdf` / `.png` | Main figure: exposure recovery versus concentration in **one representative eligible cluster** per frozen pipeline, with cluster-size-dependent point areas and reference lines for exposure prevalence among all Boston cases. A large impure group can recover many cases without strong exposure concentration. |
| `fig19_boston_partition_context.pdf` / `.png` | Main figure: singleton and largest-cluster shares of all cases, alongside ARI (cell colour/text) and AMI (cell text) for the three focused graph partitions versus frozen raw and dated TreeCluster partitions. Agreement measures partition similarity, **not** epidemiological truth. |
| `fig20_boston_all_exposures_<criterion>.pdf` / `.png` | Supplement for each frozen criterion: concentration and recovery of Conference/SNF in every selected graph and TreeCluster partition. No eligible exposure cluster is shown as undefined, not zero. |
| `fig21_boston_all_agreement_<criterion>.pdf` / `.png` | Supplement for each frozen criterion: all selected graph-versus-tree ARI and AMI combinations. D/S on tree axes distinguishes source cutoffs applied to the same empirical raw/dated trees. |

In the named-exposure assessment a representative cluster is the eligible cluster containing the **largest number of labelled exposure cases** for that frozen setting; exposure recovery is the fraction captured by **that one cluster**, not the total distributed among clusters. Its choice uses exposure metadata for *description*, not for tuning any model. `cluster_composition.csv` provides the complete exposure/clade/mutation composition for more detailed supplementary analysis, and `named_tree_overlaps.csv` supplies exposure-specific graph/tree membership overlap. With no external TreeCluster input configured, external `best_cluster_overlaps.csv` is empty. The candidate-pair coverage denominator applies to censored graph scoring, whereas tree building uses all-pair alignment distances. These are descriptive empirical results, not transmission precision/recall or independent outbreak replicates.

## Interpretation

- **Synthetic D/S labels** (e.g., EDS, GD_S, LOGIT_D) identify the source operating rules or fitted classifiers from the baseline, not separate Boston measurements. All scorers use the same observed Boston GD and TD vectors.
- **LOGIT_S/LOGIT_D** reuse the baseline's fitted logistic classifiers without retraining.
- **Raw trees** are built from uncensored TN93 all-pair distances (FastME, midpoint rooted). **Dated trees** use TreeTime with real collection dates.
- **TreeCluster thresholds** use substitutions/site for raw trees and days of tree branch distance for dated trees. Cluster sizes depend on the method and cutoff; neither tree type guarantees finer or more accurate partitions.
- **Exposure labels and TreeCluster agreement** are descriptive external evidence, not transmission truth. The Boston dataset lacks complete transmission links for precision/recall evaluation.
- **Candidate coverage** is less than 1.0 because the TN93 source is distance-censored; missing pairs are not imputed.

## Troubleshooting

| Symptom                                                         | Check / Next action                                                                                                                   |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `Executable not found: tn93`                                  | Install TN93 (`conda install -c bioconda tn93`) or set `trees.tn93_executable` in config.                                         |
| `Reference has no selected ... TreeCluster settings`          | The baseline must have completed`--stage evaluate` with TreeCluster enabled and selected settings.                                  |
| `Boston scorers must be a unique nonempty subset`             | Use valid scorer names; ES/ESS and ED/EDS are aliases and cannot be combined.                                                         |
| `Scientific implementation/dependencies differ from baseline` | The Boston adapter (`inputs/boston.py`) is excluded from baseline checks. Other scientific module changes require a fresh baseline. |
| `FastME exited 1; invalid distance matrix`                    | Ensure the PHYLIP distance matrix uses fixed-point format (not scientific notation).                                                  |

## Scientific notes

The Boston empirical application demonstrates how frozen operating settings from a synthetic baseline transfer to real outbreak data. It does **not** claim transmission truth recovery, as complete epidemiological links are unavailable. The analysis is descriptive: it reports cluster sizes, exposure composition, and method agreement, without asserting correctness.

For sensitivity to natural-history parameters, use the **perturbation workflow** on synthetic data. It crosses baseline/matched inference with baseline/updated full-graph Leiden resolution using separate development observations. Boston applies the baseline full-graph resolutions directly; every observed pair is retained, and binary-edge EpiLink Leiden is excluded.

## See also

- [Operational guide](../../OPERATIONS.md) for setup, configuration, and troubleshooting
- [Synthetic baseline protocol](../01_synthetic_baseline/README.md) for metric definitions and operating criteria
- [Perturbation study](../02_synthetic_perturbation/README.md) for parameter sensitivity with frozen settings
- [Output reference](../../OUTPUTS.md#10-boston-inputs-and-results) for column definitions and worked joins
