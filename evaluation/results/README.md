# Manuscript results

Generate numbered manuscript displays from saved evaluation runs. From the repository root, after installing the project, run `python -m evaluation.results.fig01` (or `tab01`, etc.). Scripts accept `--run-dir <run>` to pin their source run and `--output-dir <directory>` to override the destination. Figure scripts accept `--format pdf|png|both` (default `both`). Supplementary `fig17`, `fig20`, and `fig21` also accept `--criterion`.

Outputs default to `evaluation/results/outputs/<study>/<run-id>/` and are named after the producing script: `fig##_description.pdf` / `.png` or `tab##_description.tex`. Scripts with multiple endpoints or criteria append that name to each file. The `fig06` calculation CSVs share its prefix. The output tree is ignored by Git; pinned inputs remain in the study output directories. Previously generated SVGs are retained alongside migrated artifacts, but new figure exports use PDF/PNG.

| Study | Figures | Tables |
| --- | --- | --- |
| [Diagnostics](../00_synthetic_diagnostics/README.md) | `fig01` feature ambiguity and known-truth controls | — |
| [Baseline](../01_synthetic_baseline/README.md#manuscript-figures-and-tables) | `fig02` pairwise discrimination; `fig03` components; `fig04` binary Leiden; `fig05` native Leiden; `fig06` resolution regret; `fig07` raw TreeCluster; `fig08` dated TreeCluster; `fig09` graph operating bars; `fig10` TreeCluster operating bars | `tab01` main operating points; `tab02` full operating points |
| [Perturbation](../02_synthetic_perturbation/README.md#manuscript-displays-from-saved-results) | `fig11` pairwise sensitivity; `fig12` cluster sensitivity; `fig13` pairwise ranges; `fig14` cluster ranges; `fig15` mode effect; `fig16` all-pairwise supplement; `fig17` all-pipelines supplement | `tab03` fresh-control performance |
| [Boston](../03_boston_application/README.md#manuscript-displays-from-frozen-results) | `fig18` exposure trade-offs; `fig19` partition context; `fig20` all-exposures supplement; `fig21` all-agreement supplement | `tab04` frozen exposures |

The [diagnostics manuscript draft and caption](notes/fig01.md) link to its pinned figure and source evidence. Study READMEs describe the displays and their interpretation in detail. Reports and their workflow-generated figures remain with their respective runs.
