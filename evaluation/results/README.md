# Manuscript results

Generate numbered manuscript displays from saved evaluation runs. From the repository root, after installing the project, run `python -m evaluation.results.fig01` (or `tab01`, etc.). Scripts accept `--run-dir <run>` to pin their source run and `--output-dir <directory>` to override the destination. Figure scripts accept `--format pdf|png|both` (default `both`). Supplementary `fig20` and `fig21` also accept `--criterion`.

Outputs default to `evaluation/results/outputs/<study>/<run-id>/` and are named after the producing script: `fig##_description.pdf` / `.png` or `tab##_description.tex`. Scripts with multiple endpoints or criteria append that name to each file. The `fig06` calculation CSVs share its prefix. The output tree is ignored by Git; pinned inputs remain in the study output directories. Previously generated SVGs are retained alongside migrated artifacts, but new figure exports use PDF/PNG.

| Study                                                                                         | Figures                                                                                                                                                                                                                                                                                           | Tables                                                       |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| [Diagnostics](../00_synthetic_diagnostics/README.md)                                          | `fig01` feature ambiguity and known-truth controls; `fig24` offspring distribution, transmission concentration and generation profile                                                                                                                                                              | —                                                            |
| [Baseline](../01_synthetic_baseline/README.md#manuscript-figures-and-tables)                  | `fig00` primary-target compatibility surfaces; `fig02` pairwise discrimination; `fig03` components; `fig04` binary Leiden; `fig05` native Leiden; `fig06` resolution regret; `fig07` raw TreeCluster; `fig08` dated TreeCluster; `fig09` graph operating bars; `fig10` TreeCluster operating bars | `tab01` main operating points; `tab02` full operating points |
| [Perturbation](../02_synthetic_perturbation/README.md#manuscript-displays-from-saved-results) | `fig12` four-mode clustering sensitivity; `fig14` paired mean/ranges; `fig15` inference contrast at baseline/updated resolution; `fig23` four-mode overview | `tab03` four-mode fresh-control clustering |
| [Boston](../03_boston_application/README.md#manuscript-displays-from-frozen-results)          | `fig18` exposure trade-offs; `fig19` partition context; `fig20` all-exposures supplement; `fig21` all-agreement supplement                                                                                                                                                                        | `tab04` frozen exposures                                     |

The [diagnostics manuscript draft and caption](notes/fig01.md) links to pinned figures and source evidence. Study READMEs describe the displays and their interpretation in detail. Reports and their workflow-generated figures remain with their respective runs.

`fig24` reads the checksummed `backbone` diagnostic stage; it also works with a completed standalone `--stage backbone` run. Generate it with `python -m evaluation.results.fig24 --run-dir <diagnostics-run>`. Its `fig24_backbone_characterisation` outputs include PDF/PNG, the plotted CSV tables, a JSON summary with source identity, and a Markdown caption. Superspreading uses the inclusive `offspring >= Poisson percentile` rule at the mean over all backbone cases, including zero offspring. Existing script and manuscript numbering is retained; `fig24` has no assigned manuscript figure number.

## Main results and appendix

The [results draft](notes/results.md) contains a concise main narrative, four main
figure captions, LaTeX table inputs, and captions for the selected appendix figures.
Manuscript numbering is independent of the script identifiers:

| Manuscript display | Producer | Content                                                                           |
| ------------------ | -------- | --------------------------------------------------------------------------------- |
| Figure 1           | `fig02`  | Development precision–recall curves, with held-out AP in the legend               |
| Figure 2           | `fig22`  | Held-out F1 versus distant-pair contamination for all selected graph/tree methods |
| Figure 3           | `fig23`  | Four-mode full-graph EpiLink clustering F1 and contamination sensitivity |
| Figure 4           | `fig18`  | Boston exposure concentration and recovery                                        |
| Table 1            | `tab01`  | Compact held-out pairwise and clustering comparison                               |
| Table 2            | `tab04`  | Boston exposure counts and representative-cluster summaries                       |
| Appendix Table A1  | `tab02`  | All primary-target selected methods and cluster summaries                         |
| Appendix Table A2  | `tab03`  | Fresh controls used for paired sensitivity comparisons                            |

`fig22` and `fig23` also save the plotted means to same-named CSV files. These
are descriptive extracts of validated saved results, without new method selection.
The main table includes both ED and ES inference within each genetic observation
process, and the score-weighted ES/logistic Leiden comparisons used downstream.
The appendix table retains connected components and active full-graph Leiden policies. EpiLink Leiden uses native weights only. The former pairwise/all-method perturbation producers (`fig11`, `fig13`, `fig16`, `fig17`) have been removed; script-number gaps are intentional. Historical manuscript drafts and retained generated outputs describe their pinned earlier analyses.

Rebuild the main and supplementary displays and all four tables together:

```bash
python -m evaluation.results.build
```

Use `--diagnostic-run`, `--baseline-run`, `--perturbation-run`, and `--boston-run`
to pin the sources, and `--format pdf|png|both` to choose figure formats. The
results draft records its specific source runs; its figure links and table inputs
refer to those runs rather than to a moving current-run pointer. Table captions
are generated inside the `.tex` files, ready for inclusion with `\input`.

`fig00` passes one identical 0–15 SNP × 0–20 day input grid to the deterministic (ED) and stochastic (ES) **EpiLink inference models**, with primary M=0 target `AD(0)` or `CA(0,0)`. It does not compare two observed-genetics datasets. Both models use the pinned run's inference parameters, Monte Carlo sample count, and scorer seed. The shared colorbar shows a compatibility **score**, not a probability.
