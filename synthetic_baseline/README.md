# Synthetic Baseline Assessment

Comprehensive evaluation of EpiLink compatibility scores, genetic distance, and logistic probabilities on the matched-baseline synthetic epidemic.

## Scientific Questions

This baseline assessment addresses four interconnected questions:

### 1. Which true relationships generate identical observations?

Two pairs with the same genetic distance and sampling-time difference cannot be distinguished by any scorer using only those inputs. We quantify this **identifiability ceiling** by grouping pairs into identical `(GD, TD)` cells and measuring the fraction of cells containing both target and non-target relationships.

### 2. How much does a high score change what we know?

We compare three scoring families across three near-transmission horizons:

- **Compatibility scores (CS)**: Raw summed EpiLink target compatibility
- **Genetic-only (-GD)**: Negative genetic distance ranking
- **Logistic probabilities**: Supervised `P(M ≤ h | GD, TD)` trained on independent observations

For each endpoint (`M==0`, `M≤1`, `M≤2`), we report average precision, enrichment, and the **M≥3 contamination fraction** at fixed operating points.

### 3. What epidemiological structure do clusters capture?

Direct/shared-infector relatedness is **not transitive**, but cluster membership is. Even with perfect pairwise information, partitioning methods cannot recover overlapping transmission neighborhoods without error. We quantify this **structural ceiling** by running Leiden on the oracle graph containing only true `M=0` edges.

### 4. Do scorer assumptions and benchmarks hold?

We validate the evaluation machinery by:

- Reproducing historical baseline scores
- Comparing scorer draws against observed target pairs
- Training logistic/lookup benchmarks on independent observation realizations

## Usage

### Run Full Baseline Assessment

```bash
cd /Users/ydnkka/Desktop/PhD\ Project/Projects/epilink-evaluation
/opt/homebrew/Caskroom/miniconda/base/envs/epilik_evaluation/bin/python -m synthetic_baseline.run
```

### Run Specific Stages

```bash
# Data preparation only
python -m synthetic_baseline.run --stage prepare

# Observations and pairwise only
python -m synthetic_baseline.run --stage 01 --stage 02

# Full assessment
python -m synthetic_baseline.run --stage all
```

### Full Sweep Mode

```bash
# Extended resolution/threshold sweeps (slower, more comprehensive)
python -m synthetic_baseline.run --full-sweep
```

## Output Structure

```text
synthetic_baseline/outputs/runs/matched/baseline/seed_12345/
├── manifest.json                    # Run provenance and settings
├── 01_observations/
│   ├── relationship_prevalence.csv  # AD/CA/M distributions
│   ├── ambiguity_summary.csv        # Mixed feature cell statistics
│   └── feature_cells.csv            # Joint (GD, TD) distributions
├── 02_pairwise/
│   ├── ranking_summary.csv          # AP by endpoint and score family
│   ├── operating_points.csv         # Precision/recall at fixed budgets
│   └── contamination_summary.csv    # M≥3 fraction at operating points
├── 03_clusters/
│   ├── partition_summary.csv        # Cluster composition and metrics
│   ├── oracle_target_partition.csv  # Structural ceiling benchmark
│   ├── null_replicates.csv          # Size/time-preserving randomizations
│   └── treecluster_partition_summary.csv  # TreeCluster results
├── 04_validation/
│   ├── scorer_generator_checks.csv  # Draw vs observed comparisons
│   ├── benchmark_ranking.csv        # Logistic/lookup AP comparison
│   └── historical_reproduction.json # Legacy score reproduction status
├── figures/                         # PNG figures for report
├── report.md                        # Markdown report
└── report.html                      # HTML report
```

## Key Metrics

### Pairwise Informativeness

| Metric                     | Interpretation                                           |
| -------------------------- | -------------------------------------------------------- |
| **Average Precision (AP)** | Area under precision-recall curve; 1.0 = perfect ranking |
| **Target Enrichment**      | Precision / prevalence; >1 indicates informative scoring |
| **M≥3 Contamination**      | Fraction of selected pairs with ≥3 intermediates         |

### Cluster Structure

| Metric                    | Interpretation                                              |
| ------------------------- | ----------------------------------------------------------- |
| **M≤2 Pair Precision**    | Fraction of within-cluster pairs that are near-transmission |
| **M≤2 Pair Recall**       | Fraction of all M≤2 pairs captured in clusters              |
| **Direct Edge Retention** | Fraction of true AD(0)/CA(0,0) edges within clusters        |
| **Oracle BCubed F1**      | Structural ceiling for partition-based recovery             |

## Interpretation Guidelines

### High Enrichment ≠ High Precision

A score can show 100× enrichment over baseline while having only 30% precision if the target prevalence is 0.3%. Always report **absolute precision** alongside enrichment.

### Oracle Ceiling Reveals Structural Limits

If the oracle (perfect M=0 edges) achieves only 15% M≤2 recall at a given resolution, no method using that clustering approach can exceed 15% recall—regardless of pairwise accuracy.

### M≥3 Contamination Is Critical

A method selecting 1000 pairs with 40% M≥3 contamination is including 400 epidemiologically distant pairs. For outbreak investigation, this may be unacceptable even with 70% target recall.

### Null Baselines Separate Signal from Structure

If size-preserving randomization achieves 80% of the observed cluster composition, the method is primarily capturing cluster-size structure rather than transmission signal.

## Configuration

Edit `config.yaml` to modify:

- `seeds`: Evaluation seeds (must differ from `training_seed`)
- `clusters.selected_fractions`: Top-score fractions for graph construction
- `treecluster.threshold_days`: Temporal thresholds for TreeCluster
- `figures.primary_selection_fraction`: Operating point for main report tables

## Dependencies

Requires the `epilik_evaluation` conda environment with:

- `epilink` (synthetic epidemic simulator)
- `igraph`, `networkx` (graph algorithms)
- `scikit-learn` (logistic regression)
- `pandas`, `numpy`, `scipy` (numerical operations)
- `matplotlib` (figures)
- `TreeCluster.py` (optional, for TreeCluster comparisons)

## Citation Notes

This baseline assessment uses the relationship encoding and ambiguity analysis from `synthetic_exploration/` and the informativeness comparisons from `synthetic_informativeness/`. The oracle cluster benchmark and 8-category relationship composition are new additions.

## License

Part of the EpiLink evaluation framework.
