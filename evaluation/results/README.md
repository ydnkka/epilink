# Manuscript results

Generate numbered manuscript displays from saved evaluation runs. From the repository root, after installing the project, run `python -m evaluation.results.fig01` (or `tab01`, etc.). Scripts accept `--run-dir <run>` to pin their source run and `--output-dir <directory>` to override the destination. Figure scripts accept `--format pdf|png|both` (default `both`). Supplementary `fig19` and `fig20` also accept `--criterion`.

Outputs default to `evaluation/results/outputs/<study>/<run-id>/` and are named after the producing script: `fig##_description.pdf` / `.png` or `tab##_description.tex`. Scripts with multiple criteria append that name to each file. Evidence CSVs, JSON summaries and captions share their figure's prefix. The output tree is ignored by Git; pinned inputs remain in the study output directories. Figure exports use PDF/PNG.

## Figure sequence

Figures run consecutively from `fig01` to `fig20`, in study order: diagnostics,
baseline, perturbation, then Boston. Each row gives the producer module and output
stem; append `.pdf` or `.png` for the figure file.

| Study        | Producer | Output stem                              | Content                                                                   |
| ------------ | -------- | ---------------------------------------- | ------------------------------------------------------------------------- |
| Diagnostics  | `fig01`  | `fig01_backbone_characterisation`        | Offspring distribution, transmission concentration and generation profile |
| Diagnostics  | `fig02`  | `fig02_diagnostics_figure`               | Six-panel feature ambiguity and endpoint-oracle graph controls            |
| Baseline     | `fig03`  | `fig03_primary_compatibility_surfaces`   | ED/ES compatibility surfaces on one shared input grid                     |
| Baseline     | `fig04`  | `fig04_pairwise_discrimination`          | Development precision–recall curves, with held-out AP in the legend       |
| Baseline     | `fig05`  | `fig05_components`                       | Connected-component trade-offs                                            |
| Baseline     | `fig06`  | `fig06_leiden`                           | Full-graph Leiden trade-offs                                              |
| Baseline     | `fig07`  | `fig07_epilink_resolution_regret`        | Development-only shared-resolution regret                                 |
| Baseline     | `fig08`  | `fig08_treecluster_raw`                  | Raw IQ-TREE/TreeCluster trade-offs                                        |
| Baseline     | `fig09`  | `fig09_treecluster_dated`                | LSD2-dated TreeCluster trade-offs                                         |
| Baseline     | `fig10`  | `fig10_graph_cluster_operating_bars`     | Held-out graph-clustering metrics                                         |
| Baseline     | `fig11`  | `fig11_treecluster_operating_bars`       | Held-out TreeCluster metrics                                              |
| Baseline     | `fig12`  | `fig12_cluster_recovery_contamination`   | Compact held-out graph/tree comparison                                    |
| Perturbation | `fig13`  | `fig13_cluster_sensitivity`              | Four-mode clustering sensitivity                                          |
| Perturbation | `fig14`  | `fig14_cluster_f1_contamination_ranges`  | Paired means and seed-level ranges                                        |
| Perturbation | `fig15`  | `fig15_epilink_mode_effect`              | Inference contrast at baseline/updated resolution                         |
| Perturbation | `fig16`  | `fig16_parameter_sensitivity_overview`   | Compact four-mode overview                                                |
| Boston       | `fig17`  | `fig17_boston_exposure_tradeoffs`        | Exposure concentration and recovery                                       |
| Boston       | `fig18`  | `fig18_boston_partition_context`         | Partition burden and graph/tree agreement                                 |
| Boston       | `fig19`  | `fig19_boston_all_exposures_<criterion>` | All-exposures supplement                                                  |
| Boston       | `fig20`  | `fig20_boston_all_agreement_<criterion>` | All-agreement supplement                                                  |

Study READMEs describe the displays and their interpretation:

- [Diagnostics](../00_synthetic_diagnostics/README.md)
- [Baseline](../01_synthetic_baseline/README.md#manuscript-figures-and-tables)
- [Perturbation](../02_synthetic_perturbation/README.md#manuscript-displays-from-saved-results)
- [Boston](../03_boston_application/README.md#manuscript-displays-from-frozen-results)

Reports and their workflow-generated figures remain with their respective runs.

`fig01` reads the checksummed `backbone` diagnostic stage; it also works with a
completed standalone `--stage backbone` run. Generate it with
`python -m evaluation.results.fig01 --run-dir <diagnostics-run>`. Its exports
include PDF/PNG, plotted CSV tables, a JSON summary with source identity and a
Markdown caption. Superspreading uses the inclusive `offspring >= Poisson percentile`
rule at the mean over all backbone cases, including zero offspring.

`fig03` passes one identical 0–15 SNP × 0–20 day input grid to the deterministic
(ED) and stochastic (ES) **EpiLink inference models**, with primary M=0 target
`AD(0)` or `CA(0,0)`. Both models use the pinned run's inference parameters,
Monte Carlo sample count and scorer seed. The shared colorbar shows a compatibility
score rather than a probability.

`fig07` saves its calculation CSVs with the same prefix. `fig12` and `fig16` also
save plotted means to same-named CSV files. These are descriptive extracts of
validated saved results, without new method selection.

## Tables

| Producer | Output                                | Content                                                     |
| -------- | ------------------------------------- | ----------------------------------------------------------- |
| `tab01`  | `tab01_operating_points.tex`          | Compact held-out pairwise and clustering comparison         |
| `tab02`  | `tab02_operating_points_full.tex`     | All primary-target selected methods and cluster summaries   |
| `tab03`  | `tab03_fresh_control_performance.tex` | Fresh controls used for paired sensitivity comparisons      |
| `tab04`  | `tab04_boston_frozen_exposures.tex`   | Boston exposure counts and representative-cluster summaries |

The main baseline table includes both ED and ES inference within each genetic
observation process. The full table retains connected components and one
full-graph `leiden/<scorer>` pipeline per scorer. Perturbation figures show
EDD/ESD and EDS/ESS in separate scorer panels, with four inference/resolution modes.
Table captions are generated inside the `.tex` files, ready for inclusion with `\input`.

## Rebuild

Rebuild all figures in numerical order and all four tables together:

```bash
python -m evaluation.results.build
```

Use `--diagnostic-run`, `--baseline-run`, `--perturbation-run`, and `--boston-run`
to pin the sources, and `--format pdf|png|both` to choose figure formats. Record
the pinned run directories when preparing manuscript results.
