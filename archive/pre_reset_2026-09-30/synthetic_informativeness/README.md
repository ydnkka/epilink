# Synthetic Informativeness

Fresh matched-baseline analyses for assessing whether compatibility scores add useful information beyond genetic distance alone and supervised logistic probabilities.

The directory is intentionally separate from `synthetic_exploration/`. It reuses the existing synthetic data, scoring, truth, and clustering helpers, but writes a new output tree and report.

## Run

From the repository root:

```bash
python3 -m synthetic_informativeness.run
python3 -m unittest discover -s synthetic_informativeness/tests -v
```

The default configuration is `config.yaml` and targets the matched baseline run:

- condition: `matched`
- scenario: `baseline`
- evaluation seed: `12345`
- logistic training seed: `54321`

Large synthetic data caches are reused from `synthetic_exploration/outputs` by default. Fresh analysis outputs are written to:

```text
synthetic_informativeness/outputs/runs/matched/baseline/seed_12345/
```

Start with `report.md` or `report.html` in that run directory.

## Scientific Target

`M>=3` is treated as negative contamination.

The positive endpoints are analysed separately:

| Endpoint | Positive pairs |
|---|---|
| `M0` | `M == 0` |
| `Mle1` | `M <= 1` |
| `Mle2` | `M <= 2` |

This keeps the primary target, direct transmission and shared-infector pairs, distinct from broader near-transmission horizons. `M>=3` is never trained as a positive endpoint.

## Pairwise Comparisons

`01_pairwise/` compares:

- compatibility scores: `CS` from EDD, EDS, ESD, and ESS;
- genetic-only rankings: `-GD` for deterministic and stochastic genetics;
- logistic probabilities: separately trained `P(M<=h | GD, TD)` for each endpoint and data process.

Main tables:

- `ranking_summary.csv`: average precision and prevalence by endpoint/score;
- `precision_recall.csv`: full tie-aware precision-recall curves;
- `operating_points.csv`: fixed candidate budgets and recall operating points;
- `selection_M_composition.csv`: selected-pair composition by `M0`, `M1`, `M2`, `Mge3`, and undefined `M`;
- `contamination_summary.csv`: compact view of `M>=3` contamination.

## Cluster Comparisons

`02_clusters/` builds pairwise graphs from top-score fractions and evaluates:

- connected components;
- Leiden community detection.

Every within-cluster pair is evaluated, not just retained graph edges. The main cluster outputs are:

- `partition_summary.csv`;
- `cluster_frontier.csv`;
- `within_cluster_M_composition.csv`;
- `memberships.csv`.

Cluster summaries include within-cluster precision/recall for `M0`, `Mle1`, and `Mle2`, plus the within-cluster `M>=3` contamination fraction.

## TreeCluster Comparisons

`03_treecluster/` evaluates TreeCluster partitions when `TreeCluster.py` is available. It compares:

- raw genetic FastME trees;
- temporal dated TreeTime trees.

If TreeCluster is unavailable, the workflow records a skipped status and still finishes the pairwise and graph analyses. The default TreeCluster stage is resumable and runtime-capped so the main report is always rendered; increase `treecluster.max_runtime_seconds` or rerun the workflow to extend the sweep.

TreeCluster outputs are evaluated with the same within-cluster `M` summaries as the graph clusters.

## Interpretation

A score is informative for this purpose if it recovers near-transmission pairs with less `M>=3` contamination than genetic distance alone. Logistic probabilities are supervised comparators trained on a separate observation realization on the same transmission tree; they are not independent epidemic validation.
