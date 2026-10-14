# Boston empirical application

Empirical clustering analysis of SARS-CoV-2 sequences from Boston outbreaks (March-May 2020), using frozen baseline settings.

## Purpose and study sequence

**Main question:** Do methods selected on synthetic data identify clusters concentrated for Conference/SNF exposures in Boston, how much of each exposure group is recovered, and how do partitions compare with TreeCluster?

**Empirical transfer and descriptive application study.** Applies EpiLink scores, genetic distances, baseline-fitted logistic models to Boston genetic distances/sampling dates. Compares graph partitions with TreeCluster partitions (Boston alignment).

**Objectives:**

1. **Assess transfer:** apply baseline's selected graph/TreeCluster settings to empirical observations.
2. **Characterize coherence:** describe exposure concentration, focus-exposure fraction captured, cluster sizes/singletons, graph/tree partition agreement.

Frozen-transfer analysis:

| Analysis                            | Settings                           | Evidence                                       |
| ----------------------------------- | ---------------------------------- | ---------------------------------------------- |
| **Frozen transfer** (`--stage all`) | Baseline-selected operating points | Empirical partitions, focus-exposure summaries |

**Evidence:** cluster memberships, exposure composition/recovery, cluster-size summaries, partition agreement. Concentration + recovery interpreted together: small pure cluster = few exposed cases; large cluster = many with little concentration. No transmission truth; descriptive epidemiological evidence (not transmission accuracy).

**Role:** [baseline](../01_synthetic_baseline/README.md) compares methods (known truth), supplies operating points. [perturbation](../02_synthetic_perturbation/README.md): biological parameter changes, inference mismatch (simulation). Boston: baseline reference for real-data application (exposure concentration, recovery, graph/tree agreement).

## Workflow overview

1. **Prepares inputs** from Boston metadata, Nextclade, TN93 distances
2. **Scores observed pairs** (eight frozen rules: EDD, EDS, ESD, ESS, GD_S, GD_D, LOGIT_S, LOGIT_D)
3. **Clusters graphs** (frozen component cutoffs, Leiden resolutions)
4. **Builds phylogenetic trees** (Boston alignment + reference, IQ-TREE/LSD2)
5. **Runs TreeCluster** (raw/dated trees, frozen thresholds)
6. **Compares partitions** (exposure metadata, TreeCluster results)

## Quick start

```bash
python -m pip install -e '.[test]'

# Prepare inputs (cases.parquet, observed_pairs.parquet)
python evaluation/03_boston_application/run.py --stage prepare
# or:
epilink-evaluate boston --stage prepare --config evaluation/03_boston_application/config.yaml

# Build trees + TreeCluster
python evaluation/03_boston_application/run.py --stage trees

# Full analysis (scoring, clustering, trees, assessment)
python evaluation/03_boston_application/run.py --stage all

# Re-render report
python evaluation/03_boston_application/run.py --stage report
```

CLI: `epilink-evaluate boston --config evaluation/03_boston_application/config.yaml --stage all`.

`--stage all`: frozen scoring, graph clustering, TreeCluster, assessment. `--stage trees`: trees + frozen TreeCluster rules. No `--smoke`. Input preparation: before baseline evaluation OK; computational stages: need baseline frozen selection + held-out provenance.

## Configuration

Edit [`config.yaml`](config.yaml):

| Field                                           | Meaning                                                                                             |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `baseline_run`                                  | Completed baseline run or `current.json` pointer                                                    |
| `output_directory`                              | Boston output root (separate from baseline)                                                         |
| `inputs.data_root`                              | Root for `raw/boston/` source files                                                                 |
| `inputs.cases_path`, `inputs.pairs_path`        | Prepared tables (default `outputs/inputs/`); independent of run output root                         |
| `scorers`                                       | Subset of EDD, EDS, ESD, ESS, GD_S, GD_D, LOGIT_S, LOGIT_D. ES→ESS, ED→EDS aliases (cannot combine) |
| `assessment.treecluster_path`                   | Optional external TreeCluster partition (TSV: `SequenceName`, `ClusterNumber`; `-1` = singletons)   |
| `assessment.focus_exposures`                    | Exposure labels (default: Conference, SNF)                                                          |
| `assessment.min_cluster_size`                   | Min cluster size for focus analysis (default: 2)                                                    |
| `trees.enabled`                                 | Infer raw/dated trees + TreeCluster (supplied: true; loader: false)                                 |
| `trees.alignment_path`                          | Boston FASTA (required if `trees.enabled`)                                                          |
| `trees.reference_path`                          | Aligned reference FASTA (required if `trees.enabled`)                                               |
| `phylogeny.executable`                          | IQ-TREE name/path (`iqtree` discovers `iqtree3`/`iqtree2`/`iqtree`)                                 |
| `phylogeny.model`, `threads`, `seed`            | Model (MFP), threads, IQ-TREE seed                                                                  |
| `phylogeny.clock_rate`                          | Fixed rate (subs/site/day); null = LSD2 estimation                                                  |
| `phylogeny.timeout`                             | IQ-TREE/LSD2 limit (seconds)                                                                        |
| `treecluster.executable`, `treecluster.timeout` | Optional runtime overrides (methods/cutoffs frozen from baseline)                                   |

EpiLink parameters, Monte Carlo, logistic models, component cutoffs, Leiden resolutions, TreeCluster methods/cutoffs: from baseline. Leiden graphs: all observed pairs retained. Raw TreeCluster: SNP counts from synthetic alignment length → Boston alignment length. Dated rules: days (unchanged). Unit conversion; source setting ID retained.

## Inputs

Source files: `data/raw/boston/`:

| File                                              | Role                                              |
| ------------------------------------------------- | ------------------------------------------------- |
| `MGH_DPH_98percent_772samples_metadata.csv`       | Case metadata (collection dates, exposure flags)  |
| `MGH_DPH_98percent_772samples_nextclade.tsv`      | Nextclade clades, substitutions                   |
| `MGH_DPH_98percent_772samples_tn93_distances.csv` | Pairwise TN93 distances (censored at 0.0005/site) |
| `MGH_DPH_98percent_772samples_aligned.fasta`      | Aligned sequences (tree building, uncensored)     |

`prepare` stage (`epilink-evaluate boston --stage prepare`): derives `cases.parquet`, `observed_pairs.parquet` + provenance (`evaluation/03_boston_application/outputs/inputs/`). Computational stages: same preparation, reuse matching artifacts. `--output`: changes run root (not shared input paths). TN93: **distance-censored** (missing pairs = unobserved, not zero). Candidate universe: explicit in manifest.

## Outputs

- Run root: `evaluation/03_boston_application/outputs/boston/`
- Shared inputs: `evaluation/03_boston_application/outputs/inputs/`

Shared inputs: `cases.parquet`, `observed_pairs.parquet`, `manifest.json`. Run root:

```text
boston/
  current.json                          # Latest run pointer
  artifacts/
    scores/<id>/                        # Pairwise compatibility scores
    trees/<id>/                         # Raw/dated tree artifacts
  runs/<fingerprint>/
    manifest.json                       # Status, config, implementation hashes
    reference.json                      # Baseline run identity
    selection.json                      # Frozen operating points
    settings.json                       # Method definitions (setting ID)
    inputs.json                         # Input paths/checksums
    clusters/                           # Graph clustering (metrics.csv, status.json)
    trees/                              # TreeCluster (metrics.csv, status.json, inputs.json)
    assessment/                         # Empirical summaries
      summary.csv                       # Partition sizes, coverage
      cluster_composition.csv           # Exposure/clade/mutation counts
      named_cluster_overlaps.csv        # Named exposure summaries
      best_cluster_overlaps.csv         # External-comparator overlaps
      tree_agreement.csv                # ARI/AMI (graph vs tree)
      named_tree_overlaps.csv           # Named exposure overlaps (tree clusters)
    figures/                            # Cluster size/overlap plots
    report.md, report.html              # Summary
```

## Manuscript displays from frozen results

Generate from saved results:

```bash
python -m evaluation.results.tab04  # frozen exposures
python -m evaluation.results.tab05  # every cluster containing Conference or SNF sequences
python -m evaluation.results.fig17  # exposure trade-offs
python -m evaluation.results.fig18  # partition context
python -m evaluation.results.fig19  # all-exposures supplement
python -m evaluation.results.fig20  # all-agreement supplement
```

Default: `outputs/boston/current.json`. Output: `evaluation/results/outputs/03_boston_application/<run-id>/`. Options: `--run-dir`, `--output-dir`; figures: `--format pdf|png|both`; supplements: `--criterion balanced_M0|balanced_Mle1|balanced_Mle2|all`. Validate: reference, setting IDs, graph/tree completion, assessment coverage.

**Focused comparisons (`tab04`):** `balanced_M0` full-graph Leiden (ESD, LGD), genetic-distance connected components (GDD), and deterministic-source raw/dated TreeCluster. Full-graph Leiden uses EpiLink/logistic similarity weights; genetic-distance scorers are component comparators, not Leiden pipelines. D/S = _synthetic source rule_; Boston: one empirical GD/TD table. Sequence trees: full alignment (IQ-TREE/LSD2), independent of censored pair table. Exposure metadata: describes partitions (no graph/tree setting selection).

`fig17` includes every selected `balanced_M0` cluster method, including both D/S source variants. As in `fig12`, marker shapes identify the clustering approach and colours identify the scorer family; D/S variants share family colours and tree markers are neutral. Hollow markers denote deterministic source rules and filled markers denote stochastic source rules. Four-column legends below the plots show both fill styles for every method. The plotted rows, frozen setting IDs, and source rules are saved to `fig17_boston_exposure_tradeoffs.csv`.

| Output                                           | Manuscript role                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tab04_boston_frozen_exposures.tex`              | Main table: frozen resolutions/cutoffs, Conference/SNF exposed counts/denominators, representative-cluster size, concentration (`n_exposure / n_cases`), recovery (`n_exposure / exposure_total`), clusters, singletons, largest cluster. `utils/latex_tables.py` environment.                                                    |
| `tab05_boston_exposure_clusters_<criterion>.tex` | Supplementary table and CSV: every cluster containing at least one Conference or SNF sequence, for all selected methods. Each row gives the method, selected setting, cluster ID and size, exposure group and counts, concentration, and recovery. Includes singletons.                                                           |
| `fig17_boston_exposure_tradeoffs`                | Main figure: recovery vs concentration for every selected M=0 clusterer (**one representative eligible cluster** per pipeline), separate method-marker/scorer-colour legends, cluster-size-dependent point areas, reference lines (exposure prevalence), and plotted-data CSV. Large impure group: many cases, low concentration. |
| `fig18_boston_partition_context`                 | Main figure: all selected M=0 rules in separate Leiden, connected-component, and TreeCluster grid cells, comparing singleton-case and largest-cluster shares of all Boston cases. Plotted counts and percentages are also saved as CSV.                                                                                           |
| `fig19_boston_all_exposures_<criterion>`         | Supplement (per criterion): concentration/recovery (Conference/SNF) in all frozen component, similarity-weighted Leiden, and TreeCluster partitions. Single-line labels include their rules; Leiden resolutions use two decimals. No eligible cluster = undefined (not zero).                                                     |
| `fig20_boston_all_agreement_<criterion>`         | Supplement (per criterion): all graph-vs-tree ARI/AMI combinations. Tree axes identify type, method, cutoff, and D/S source (same empirical raw/dated trees).                                                                                                                                                                     |

Representative cluster: eligible cluster with **largest number of labelled exposure cases** per frozen setting; recovery = **that one cluster**. Exposure metadata: description (not model tuning). `cluster_composition.csv`: complete exposure/clade/mutation composition. `named_tree_overlaps.csv`: graph/tree overlaps. No external comparator: `best_cluster_overlaps.csv` empty. Candidate-pair coverage: censored graph scoring; IQ-TREE: full alignment. Descriptive empirical results (not transmission accuracy/outbreak replicates).

`tab05` lists only clusters containing at least one Conference or SNF sequence, including clusters of one sequence. Each row describes one cluster and either Conference or SNF; a cluster containing both groups has two rows. Cluster size includes all sequences in the cluster, including other exposure groups and sequences without a known exposure. Concentration is the percentage of that cluster carrying the given label, and recovery is the percentage of all Boston sequences with that label found in the cluster. The complete saved exposure counts are checked against the prepared cases and partition summaries before filtering. The CSV also retains the source rule and frozen setting IDs. The default is `balanced_M0`; use `--criterion all` to export all three criteria.

## Interpretation

- **Synthetic D/S labels** (EDS, GD_S, LOGIT_D): source operating rules/classifiers from baseline (not separate Boston measurements). All scorers: same observed Boston GD/TD vectors.
- **LOGIT_S/LOGIT_D**: baseline's fitted logistic classifiers (no retraining).
- **Raw trees**: IQ-TREE ML trees (aligned reference root, reference-pruned). **Dated trees**: IQ-TREE LSD2 calibration + real collection dates; branches = days.
- **TreeCluster thresholds**: subs/site (raw), days (dated). Cluster sizes: method + cutoff dependent; neither tree type guarantees finer/accurate partitions.
- **Exposure labels/TreeCluster agreement**: descriptive external evidence (not transmission truth). Boston: no complete transmission links (no precision/recall evaluation).
- **Candidate coverage** < 1.0: TN93 distance-censored; missing pairs not imputed.

## Troubleshooting

| Symptom                                                       | Check / Action                                                                                           |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `Executable not found: iqtree`                                | Install IQ-TREE ≥2.0.6 or set `phylogeny.executable`                                                     |
| `Reference has no selected ... TreeCluster settings`          | Baseline must have completed `--stage evaluate` with TreeCluster enabled + selected settings             |
| `Boston scorers must be a unique nonempty subset`             | Valid scorer names; ES/ESS, ED/EDS = aliases (cannot combine)                                            |
| `Scientific implementation/dependencies differ from baseline` | Boston adapter (`inputs/boston.py`) excluded from baseline checks; other module changes = fresh baseline |
| IQ-TREE/LSD2 inference failure                                | Inspect `backend/run-*/inference.log`; check reference length, unique case IDs, collection dates         |

## Scientific notes

Boston empirical application: frozen baseline settings on real outbreak data. Does **not** claim transmission truth recovery (no complete epidemiological links). Descriptive analysis: cluster sizes, exposure composition, method agreement (no correctness assertion).

Natural-history sensitivity: **perturbation workflow** (baseline/matched inference × baseline/updated Leiden resolution, four EpiLink scorers). Boston: frozen full-graph resolutions directly; all observed pairs retained.

## See also

- [Operational guide](../../OPERATIONS.md): setup, configuration, troubleshooting
- [Synthetic baseline](../01_synthetic_baseline/README.md): metric definitions, operating criteria
- [Perturbation](../02_synthetic_perturbation/README.md): four-mode EpiLink clustering sensitivity
- [Output reference](../../OUTPUTS.md#10-boston-inputs-and-results): column definitions, joins
