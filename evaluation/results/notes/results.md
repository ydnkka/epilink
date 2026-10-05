# Results

## Main results draft

The synthetic benchmark contained 5,051 cases on one fully sampled transmission tree. Direct-transmission and shared-source pairs were rare, comprising 0.14% of all unordered pairs. Adding sampling-time information reduced the overlap between target and non-target genetic observations, but every target pair still shared its exact genetic and temporal observations with at least one non-target pair. Clustering also imposed a separate trade-off: connected components formed from exact target links joined the entire transmission tree into one group, while Leiden partitions separated some target pairs even when given those exact links (Appendix Figure A1). The following comparisons assess performance on new observations of this same transmission structure, after selecting settings during development.

### Pairwise discrimination depended on the genetic inference formulation

Stochastic EpiLink inference ranked recent relationships more effectively than genetic distance alone under both genetic observation processes, while supervised logistic regression had the highest average precision (AP; Figure 1). Under stochastic genetic observations, mean held-out AP was 0.253 for stochastic EpiLink (ESS), compared with 0.177 for genetic distance and 0.330 for logistic regression. Pairwise discrimination was lower under stochastic than deterministic genetic observations for every matched method comparison.

The choice of EpiLink inference formulation mattered independently of the process generating the genetic observations. Stochastic inference also outperformed deterministic inference when applied to deterministic genetic observations; deterministic EpiLink inference ranked below genetic distance alone in both observation processes. At the cutoffs selected during development, deterministic EpiLink tended to recover a larger fraction of target pairs with lower precision than stochastic EpiLink. Thus, greater recovery at one cutoff did not imply better ranking across cutoffs. Table 1 reports held-out precision, recall, and F1 at the selected settings, while Figure 1 shows the development precision–recall trade-offs and the held-out AP summaries.

### Clustering changed the recovery–contamination balance

At their separately selected settings, Leiden generally recovered recent relationships more effectively than connected components using the same pairwise scorer (Figure 2). For ESD under deterministic genetic observations, binary-edge Leiden achieved mean held-out F1 of 0.497, compared with 0.180 for connected components. Connected components frequently contained a greater fraction of distant relationships, despite the stricter cutoffs selected for that method. The development comparisons showed that both retaining more graph edges and changing Leiden resolution could alter the balance between target recovery and inclusion of non-target pairs (Appendix Figures A2–A4).

Retaining the original scores as edge weights did not consistently improve Leiden performance. Binary and score-weighted stochastic EpiLink graphs gave similar F1 at their selected settings, whereas score-weighted logistic graphs improved on their binary counterparts. For ESD, score weighting slightly increased precision and reduced recall; for ESS, it increased recall and reduced precision (Table 1). These differences show that the consequences of weighting depended on the scorer and observation process. A separate development analysis identified a compromise numerical resolution across EpiLink variants, but the held-out comparisons used each method's individually selected resolution (Appendix Figure A5).

Phylogenetic clustering remained competitive with the graph methods. Under deterministic observations, selected undated and dated TreeCluster methods had similar F1. Under stochastic observations, dated TreeCluster achieved F1 of 0.431 and distant-pair contamination of 9.0%, compared with 0.358 and 27.4% for ESS-based binary Leiden. Dating therefore accompanied a more favourable trade-off in that comparison. This result compares the complete, separately selected tree procedures, including tree reconstruction, dating, and cutoff selection. Table 1 and Appendix Table A1 give the exact metrics, and Appendix Figures A6–A9 show the tree-setting comparisons and detailed held-out results. All cluster metrics include every pair within a cluster.

### Parameter changes affected ranking and clustering differently

Sensitivity was concentrated in changes to incubation timing, the substitution rate, and clock relaxation; changes in testing-delay parameters generally had smaller effects at the evaluated levels (Figure 3). Reducing the substitution rate or mean incubation period lowered AP across the focused methods, particularly under deterministic genetic observations. For example, a 25% reduction in substitution rate reduced ESD AP by 16.0 percentage points, compared with 3.6 points for ESS, even when EpiLink used the generating parameter values. The corresponding graph-clustering comparisons generally showed lower F1 and more distant-pair contamination. Appendix Table A2 reports the fresh controls used for these paired changes.

Ranking changes did not necessarily predict performance at the fixed clustering settings. Increasing incubation-period variability improved genetic-distance and logistic AP, yet sharply reduced F1 for their selected clusters and for TreeCluster. Under deterministic observations, logistic score-weighted Leiden lost 51.3 percentage points of F1, while matched ESD score-weighted Leiden lost 3.0 points. Under stochastic observations, matched ESS clustering instead improved. Distant-pair contamination could decline alongside substantial F1 losses, so lower contamination alone did not indicate maintained target recovery. Scenario-specific ranges and complete method comparisons are shown in Appendix Figures A10–A18.

Updating EpiLink's biological parameters changed the sensitivity pattern, without retraining or reselecting cutoffs. When incubation-period variability increased by 25%, ESD AP fell by 21.0 percentage points with baseline-fixed inference, compared with 2.5 points with matched inference. Parameter matching was not uniformly beneficial: for some scenarios, including increased clock relaxation under deterministic observations, matched inference produced larger losses than baseline-fixed inference. Appendix Figure A12 shows these paired contrasts for both AP and clustering F1.

### Boston clusters differed in exposure concentration and recovery

The settings transferred from the synthetic benchmark produced markedly different exposure groupings in Boston (Figure 4; Table 2). ESD-based score-weighted Leiden grouped 26 of the 28 Conference-labelled cases in a 28-case cluster, giving both concentration and recovery of 92.9%. It also captured most skilled-nursing-facility cases in a cluster strongly concentrated for that exposure. Logistic score-weighted Leiden gave similar exposure groupings. Genetic-distance Leiden produced a pure skilled-nursing-facility cluster but recovered a smaller share of that exposure group, and its Conference cluster was less concentrated.

The tree-based methods illustrated opposite grouping extremes. Undated TreeCluster captured both exposure groups within the same 679-case cluster, giving complete recovery but low concentration. Dated TreeCluster divided cases into much smaller groups, including highly concentrated clusters that recovered only small fractions of the exposure groups. Cluster-size summaries and low graph–tree adjusted Rand agreement reflected these different partitions (Appendix Figure A19); full primary-setting exposure and agreement comparisons are in Appendix Figures A20–A21. These results describe correspondence with recorded exposures in the absence of complete transmission relationships.

## Main figures and tables

### Figure 1. Pairwise discrimination

![Pairwise precision–recall comparisons for deterministic and stochastic genetic observations.](../outputs/01_synthetic_baseline/86dfffb165a13ab986c7/fig02_pairwise_discrimination.png)

**Figure 1. Identification of recent transmission relationships from genetic and sampling-time differences.** Panels distinguish deterministic (A) and stochastic (B) genetic observations. Lines show precision–recall curves for three development realizations; diamonds show mean performance at the cutoffs selected during development. **AP values in the legends are means from three separate held-out realizations**, rather than summaries of the plotted curves. The target comprises direct transmission and infection from a shared source (M=0). Precision is the target-pair fraction among selected pairs; recall is the fraction of all target pairs recovered. ED and ES denote deterministic and stochastic EpiLink inference, respectively. The final D/S letter in each model label denotes the genetic observation process. All realizations share one transmission tree.

[Figure 1 PDF](../outputs/01_synthetic_baseline/86dfffb165a13ab986c7/fig02_pairwise_discrimination.pdf).

### Figure 2. Cluster recovery and distant-pair contamination

![Held-out target-pair F1 versus distant-pair contamination for graph and tree clusters.](../outputs/01_synthetic_baseline/86dfffb165a13ab986c7/fig22_cluster_recovery_contamination.png)

**Figure 2. Cluster performance at settings selected during development.** Points show mean target-pair F1 and distant-pair contamination across three held-out observation realizations. Panels distinguish deterministic (A) and stochastic (B) genetic observations. Colours identify pairwise scorers; symbols identify connected components, Leiden with binary edges or original score weights, and TreeCluster on undated or dated trees. TreeCluster points are black. Higher F1 and lower contamination indicate a more favourable balance. The target is M=0, while distant pairs have M≥3; contamination therefore includes only part of the target's false positives. Every within-cluster pair is assessed, including pairs without a retained graph edge. Some points coincide, particularly genetic-distance connected components and binary Leiden under stochastic observations. Tables 1 and A1 report descriptive standard deviations and the underlying precision and recall.

[Figure 2 PDF](../outputs/01_synthetic_baseline/86dfffb165a13ab986c7/fig22_cluster_recovery_contamination.pdf).

### Table 1. Held-out synthetic performance

The generated table contains the main pairwise, binary-Leiden, score-weighted ES/logistic Leiden, and tree comparisons. Its caption is included in the LaTeX file. Paths in the following `\input` commands are relative to the repository root; adjust their prefix when including them from the thesis source directory.

```latex
\input{evaluation/results/outputs/01_synthetic_baseline/86dfffb165a13ab986c7/tab01_operating_points.tex}
```

### Figure 3. Sensitivity to biological parameters

![Paired changes in average precision, cluster F1, and distant-pair contamination across biological parameter scenarios.](../outputs/02_synthetic_perturbation/5d2a64dea949b408fbd2/fig23_parameter_sensitivity_overview.png)

**Figure 3. Changes in performance when biological parameters were varied.** Rows show changes in pairwise AP (A–B), cluster F1 (C–D), and distant-pair contamination (E–F); columns distinguish deterministic and stochastic genetic observations. Each cell is the equally weighted mean percentage-point difference from fresh unperturbed controls generated using the same three simulation seeds. EpiLink uses the scenario-specific biological parameters. Logistic models, cutoffs, and clustering settings remain those selected in the baseline study. Blue indicates higher AP or F1, or lower contamination; red indicates the converse. **Each metric has its own colour scale**, shared across the two observation processes. Most parameter levels are 0.75 or 1.25 times baseline; clock-relaxation values 0 and 0.66 are absolute, with baseline 0.33. Variability denotes the coefficient of variation. Cluster columns use score-weighted EpiLink ES and logistic Leiden, binary-edge genetic-distance Leiden, and undated/dated TreeCluster (labelled Tree). Grey cells indicate undefined changes; a reduced defined-realization count is shown where applicable.

[Figure 3 PDF](../outputs/02_synthetic_perturbation/5d2a64dea949b408fbd2/fig23_parameter_sensitivity_overview.pdf).

### Figure 4. Boston exposure groupings

![Boston exposure-group recovery versus concentration in representative clusters.](../outputs/03_boston_application/4d8b3723d4708fd95783/fig18_boston_exposure_tradeoffs.png)

**Figure 4. Exposure concentration and recovery in Boston clusters.** Panels show Conference (A) and skilled-nursing-facility (B) exposure labels among 772 cases. Each point represents one method's cluster containing the most cases with the indicated label, among clusters with at least two cases. Recovery is the fraction of all cases with that exposure captured by the cluster; concentration is their fraction of the cluster's membership. Point area increases with cluster size. Dashed lines show the exposure's prevalence among all Boston cases. Settings were selected using synthetic development observations and applied without adjustment using Boston exposure labels. Graph methods use the reported distance-censored TN93 observations; trees were reconstructed independently from the alignment. ESD, LGD, and GDD identify synthetic-source scoring rules applied to the same empirical observations. These measures describe exposure correspondence.

[Figure 4 PDF](../outputs/03_boston_application/4d8b3723d4708fd95783/fig18_boston_exposure_tradeoffs.pdf).

### Table 2. Boston exposure and cluster summaries

The generated table provides labelled-case counts and denominators, representative-cluster sizes, concentration and recovery, and whole-partition cluster summaries. Its caption is included in the LaTeX file.

```latex
\input{evaluation/results/outputs/03_boston_application/4d8b3723d4708fd95783/tab04_boston_frozen_exposures.tex}
```

## Appendix materials

Appendix numbering below is independent of the figure-script identifiers. M=0 denotes direct transmission or a shared source; M≤1 and M≤2 include progressively more distant relationships. ED/ES specify EpiLink's inference formulation, while a final D/S specifies the synthetic genetic observation process. Score weights are the original positive EpiLink or logistic scores. Synthetic variability describes repeated observations on one fixed transmission tree.

### A. Observation ambiguity and development comparisons

#### Figure A1. Observation ambiguity and known-relationship controls

![Observation ambiguity and clustering controls using exact transmission relationships.](../outputs/00_synthetic_diagnostics/a529043958047472709cf4d6f2b390e9d1a39e2ff9c39acc53f2bba3abd927d8/fig01_diagnostics_figure.png)

**Figure A1. Ambiguity in pairwise observations and the consequences of partitioning known relationships.** Columns show M=0, M≤1, and M≤2. A–C give the fraction of distinct observed genetic-distance or genetic/time combinations containing both target and non-target pairs. D–F show Leiden applied to graphs connecting exactly the target pairs. G–I, J–L, and M–O show average-clade, maximum-clade, and single-linkage TreeCluster applied to the known transmission topology, with one unit of length per transmission edge. Tree cutoffs count transmission hops, which differ from M. Cluster metrics assess all within-cluster pairs. Distant-pair contamination always counts M≥3. Bars are equal-realization means; symmetric whisker half-lengths are the mean-to-maximum difference for bars and mean-to-minimum difference for F1. Graph/tree controls are unchanged across observations and do not provide independent epidemic replicates. These controls describe the evaluated representations, rather than universal performance ceilings.

#### Figure A2. Connected-component settings

![Development trade-offs for connected-component clustering.](../outputs/01_synthetic_baseline/86dfffb165a13ab986c7/fig03_components.png)

**Figure A2. Effects of graph cutoffs on connected-component clustering.** Columns distinguish genetic observation processes. Upper panels show M=0 precision versus recall; lower panels show F1 versus M≥3 distant-pair contamination. Lines follow evaluated cutoffs for each scorer. Diamonds mark the settings selected by mean development F1. Values are equally weighted means across three development realizations and include every within-cluster pair.

#### Figure A3. Binary-edge Leiden settings

![Development trade-offs for Leiden with binary edges.](../outputs/01_synthetic_baseline/86dfffb165a13ab986c7/fig04_leiden_binary.png)

**Figure A3. Effects of graph cutoffs and resolution on binary-edge Leiden clustering.** Every retained edge has weight one. Dots show evaluated cutoff–resolution combinations for M=0; lines connect non-dominated mean precision–recall or F1–contamination values. These lines summarize the trade-off envelope, rather than a single parameter sequence. Diamonds mark the settings selected during development, including when they lie outside the displayed envelope. Distant-pair contamination counts M≥3.

#### Figure A4. Score-weighted Leiden settings

![Development trade-offs for Leiden retaining the original scores as weights.](../outputs/01_synthetic_baseline/86dfffb165a13ab986c7/fig05_leiden_native.png)

**Figure A4. Effects of graph cutoffs and resolution on score-weighted Leiden clustering.** Retained positive EpiLink or logistic scores supply edge weights. Dots show evaluated cutoff–resolution combinations; lines show non-dominated development trade-offs; diamonds show selected settings. Upper panels assess M=0 precision and recall, and lower panels assess F1 and M≥3 contamination. Genetic-distance graphs use binary edges and are therefore absent from this display. Numerical resolutions act on each scorer's weight scale.

#### Figure A5. A shared development resolution

![Development F1 losses when the same numerical Leiden resolution is used for all EpiLink variants and weighting policies.](../outputs/01_synthetic_baseline/86dfffb165a13ab986c7/fig06_epilink_resolution_regret.png)

**Figure A5. Performance cost of using one numerical Leiden resolution across EpiLink variants.** At each common resolution, a separate graph cutoff is selected for each of four EpiLink variants under binary and score-weighted edges. The line gives the mean F1 loss, in percentage points, relative to each variant's best setting over its full development grid; shading spans the middle half of the eight losses. The marked resolution, 0.3, minimizes the largest loss over the evaluated common grid. Shading is not a confidence interval. This is a development-only compromise across differently scaled weights; the held-out comparison uses individually selected resolutions.

#### Figure A6. Undated-tree clustering settings

![Development comparisons of TreeCluster methods on undated genetic-distance trees.](../outputs/01_synthetic_baseline/86dfffb165a13ab986c7/fig07_treecluster_raw.png)

**Figure A6. Effects of clustering rule and genetic-distance cutoff on undated-tree clusters.** Lines follow cutoffs for maximum-clade, average-clade, and single-linkage TreeCluster. Upper panels show M=0 precision and recall; lower panels show F1 and M≥3 contamination. Diamonds identify the rule and cutoff selected jointly during development within each genetic observation process. Cutoffs are evolutionary distances on the reconstructed genetic-distance tree. Metrics assess all within-cluster pairs.

#### Figure A7. Dated-tree clustering settings

![Development comparisons of TreeCluster methods on trees dated using sampling times.](../outputs/01_synthetic_baseline/86dfffb165a13ab986c7/fig08_treecluster_dated.png)

**Figure A7. Effects of clustering rule and branch-time cutoff on dated-tree clusters.** Lines follow evaluated time cutoffs for each TreeCluster rule, and diamonds identify the rule and cutoff selected jointly during development. Upper panels show M=0 precision and recall; lower panels show F1 and M≥3 contamination. Time cutoffs represent evolutionary branch time in days, rather than the span between sampling dates alone. The comparison includes the effects of dating and its associated rooting or topology changes.

### B. Detailed held-out performance

#### Figure A8. Graph-clustering performance

![Detailed held-out precision, recall, F1, and contamination for all selected graph-clustering settings.](../outputs/01_synthetic_baseline/86dfffb165a13ab986c7/fig09_graph_cluster_operating_bars.png)

**Figure A8. Held-out performance of selected graph-clustering methods.** Rows show connected components, binary-edge Leiden, and score-weighted Leiden; columns distinguish genetic observation processes. Bars give mean M=0 precision, recall, F1, and M≥3 distant-pair contamination at settings selected during development. Whiskers show sample SD across three held-out observation realizations on the same tree. Genetic-distance score weighting is not used. Assessment includes every within-cluster pair.

#### Figure A9. Tree-clustering performance

![Detailed held-out performance of selected undated and dated TreeCluster methods.](../outputs/01_synthetic_baseline/86dfffb165a13ab986c7/fig10_treecluster_operating_bars.png)

**Figure A9. Held-out performance of selected undated and dated TreeCluster procedures.** Panels distinguish tree types; categories identify the genetic observation process and its selected rule and cutoff. Bars give mean M=0 precision, recall, F1, and M≥3 contamination; whiskers show sample SD over three held-out realizations. Rule and cutoff were selected separately for each tree type and observation process. Undated cutoffs are displayed as SNP counts converted from substitutions per site using the simulated sequence length; dated cutoffs are days.

#### Table A1. Complete primary-target comparison

```latex
\input{evaluation/results/outputs/01_synthetic_baseline/86dfffb165a13ab986c7/tab02_operating_points_full.tex}
```

The generated caption defines direct-transmission and shared-source retention, selected-pair counts, singleton shares, and largest-cluster shares, alongside precision, recall, F1, and contamination.

### C. Sensitivity ranges, parameter mismatch, and broader targets

#### Table A2. Fresh unperturbed controls

```latex
\input{evaluation/results/outputs/02_synthetic_perturbation/5d2a64dea949b408fbd2/tab03_fresh_control_performance.tex}
```

These are the fresh controls paired with the sensitivity scenarios, rather than the earlier held-out baseline observations. The generated caption specifies the settings and denominators.

#### Figure A10. Pairwise sensitivity ranges

![Mean paired changes in average precision and ranges across observation realizations.](../outputs/02_synthetic_perturbation/5d2a64dea949b408fbd2/fig13_pairwise_ap_ranges.png)

**Figure A10. Mean changes and observation-to-observation ranges in pairwise ranking performance.** Dots show mean M=0 AP differences from same-seed fresh controls; whiskers span the minimum and maximum of the three paired differences for that particular scenario. Colours identify EpiLink ES, logistic regression, and genetic distance. EpiLink uses matched biological parameters. Open dots indicate fewer than three defined differences; n=0 indicates none. Ranges describe conditional observation variation, not confidence intervals or variation pooled across different scenarios.

#### Figure A11. Cluster sensitivity ranges

![Mean paired changes in cluster F1 and contamination with ranges across observation realizations.](../outputs/02_synthetic_perturbation/5d2a64dea949b408fbd2/fig14_cluster_f1_contamination_ranges.png)

**Figure A11. Mean changes and observation-to-observation ranges in cluster performance.** Upper panels show paired changes in M=0 F1; lower panels show paired changes in M≥3 contamination. Dots are equally weighted means and whiskers span the three scenario-specific realization differences. EpiLink uses matched parameters; all methods retain the baseline-selected settings. Positive F1 changes are favourable, while positive contamination changes indicate more distant pairs. Each metric has its own axis scale. Open dots and n labels identify reduced defined-value coverage.

#### Figure A12. Updating EpiLink parameters

![Within-realization contrasts between matched and baseline-fixed EpiLink inference.](../outputs/02_synthetic_perturbation/5d2a64dea949b408fbd2/fig15_epilink_mode_effect.png)

**Figure A12. Effect of updating EpiLink's biological parameters.** Panels compare matched with baseline-fixed inference for ESD/ESS pairwise AP (upper row) and score-weighted Leiden F1 (lower row). Each contrast is the difference between the two modes' same-seed, control-paired changes, calculated before averaging. Diamonds show means and whiskers the range across three realizations. Positive values favour updating the parameters. Cutoffs and clustering settings remain fixed; the comparison does not change deterministic versus stochastic genetic inference.

#### Figure A13. Primary-target sensitivity for all pairwise models

![Paired changes in primary-target AP for all eight pairwise scoring rules.](../outputs/02_synthetic_perturbation/5d2a64dea949b408fbd2/fig16_pairwise_ap_all_M0.png)

**Figure A13. Sensitivity of all pairwise models for direct transmission or shared-source infection.** Cells give mean percentage-point changes in M=0 AP from fresh same-seed controls. Columns distinguish genetic observation processes and include both ED and ES inference, genetic distance, and logistic regression. EpiLink uses matched biological parameters. Blue indicates increased AP and red decreased AP. Values with fewer than three defined differences carry their available-realization count.

#### Figure A14. Sensitivity at relationship horizon M≤1

![Paired changes in AP for the broader relationship target M less than or equal to one.](../outputs/02_synthetic_perturbation/5d2a64dea949b408fbd2/fig16_pairwise_ap_all_Mle1.png)

**Figure A14. Pairwise sensitivity when the assessment includes relationships with M≤1.** Cells give mean AP changes from fresh same-seed controls, in percentage points, using matched EpiLink parameters. The broader target includes additional intervening infections, but the evaluated compatibility scores and logistic models retain their M=0 definitions. Blue indicates increased AP and red decreased AP. The same colour scale is used for the two genetic observation processes.

#### Figure A15. Sensitivity at relationship horizon M≤2

![Paired changes in AP for the broader relationship target M less than or equal to two.](../outputs/02_synthetic_perturbation/5d2a64dea949b408fbd2/fig16_pairwise_ap_all_Mle2.png)

**Figure A15. Pairwise sensitivity when the assessment includes relationships with M≤2.** Cells show mean percentage-point AP changes from fresh same-seed controls. EpiLink uses matched biological parameters. Assessment includes more distant relationships while keeping the M=0 compatibility-score target and fitted logistic models unchanged. Blue indicates increased AP and red decreased AP; grey indicates an undefined change.

#### Figure A16. Primary-target sensitivity for every selected method

![Paired changes in F1 and distant-pair contamination for every primary-target selected method.](../outputs/02_synthetic_perturbation/5d2a64dea949b408fbd2/fig17_all_pipelines_balanced_M0.png)

**Figure A16. Sensitivity of all methods at settings selected for M=0.** Rows within each genetic observation process identify pairwise rules, connected components, Leiden weight policies, and undated/dated TreeCluster. Left panels show mean paired F1 changes; right panels show mean paired M≥3 contamination changes. EpiLink uses matched parameters. Blue denotes increased F1 or reduced contamination, with separate metric colour scales. Grey cells and counts identify undefined or incompletely defined differences.

#### Figure A17. Sensitivity of settings selected for M≤1

![Paired changes for every method at settings selected for the M less than or equal to one target.](../outputs/02_synthetic_perturbation/5d2a64dea949b408fbd2/fig17_all_pipelines_balanced_Mle1.png)

**Figure A17. Sensitivity of all methods at settings selected for M≤1.** The left panels assess F1 against the broader target and the right panels retain the M≥3 contamination definition. Each method uses its own settings selected for mean development M≤1 F1, applied unchanged to the perturbed observations. Pairwise score definitions remain those for M=0. Cells summarize same-seed, fresh-control differences under matched EpiLink parameters, with blue indicating the favourable direction for each metric.

#### Figure A18. Sensitivity of settings selected for M≤2

![Paired changes for every method at settings selected for the M less than or equal to two target.](../outputs/02_synthetic_perturbation/5d2a64dea949b408fbd2/fig17_all_pipelines_balanced_Mle2.png)

**Figure A18. Sensitivity of all methods at settings selected for M≤2.** Left panels show F1 changes for M≤2, while right panels show M≥3 contamination changes. Methods retain their settings selected for mean development M≤2 F1; EpiLink and logistic pairwise scores retain their original M=0 target definitions. Changes are means of fresh same-seed control differences under matched EpiLink parameters. Separate colour scales apply to F1 and contamination.

### D. Boston partition context and full primary-setting comparisons

#### Figure A19. Cluster structure and graph–tree agreement

![Boston singleton and largest-cluster shares, with agreement between graph and tree partitions.](../outputs/03_boston_application/4d8b3723d4708fd95783/fig19_boston_partition_context.png)

**Figure A19. Cluster structure and agreement between graph and tree partitions in Boston.** A shows the percentages of all cases assigned to singleton clusters or the largest cluster for each focused method. B compares the three graph partitions with undated and dated TreeCluster partitions. Cell colour and ARI text give the adjusted Rand index; AMI text gives adjusted mutual information. Both account for agreement expected by chance. Settings were selected in the synthetic benchmark. These indices assess similarity between methods' groupings, not accuracy against transmission truth.

#### Figure A20. Exposure summaries for every primary-setting partition

![Exposure concentration and recovery for every selected Boston clustering method at primary-target settings.](../outputs/03_boston_application/4d8b3723d4708fd95783/fig20_boston_all_exposures_balanced_M0.png)

**Figure A20. Exposure concentration and recovery for all Boston partitions selected for M=0 in simulation.** Each row identifies a graph or tree method and each column an exposure group. Concentration and recovery describe one representative eligible cluster per exposure, chosen by its labelled-case count. Values are percentages. Grey cells indicate that no eligible exposure cluster exists. D/S tree labels denote the synthetic source of the selected cutoff; both source rules are applied to the same Boston trees. Exposure labels were not used to select analytical settings.

#### Figure A21. Agreement for every primary-setting graph–tree comparison

![Adjusted Rand index and adjusted mutual information for every primary-setting Boston graph-versus-tree comparison.](../outputs/03_boston_application/4d8b3723d4708fd95783/fig21_boston_all_agreement_balanced_M0.png)

**Figure A21. Agreement between all primary-setting graph and tree partitions in Boston.** Left and right panels show adjusted Rand index and adjusted mutual information, respectively. Rows identify graph methods; columns identify undated or dated TreeCluster using cutoffs developed under deterministic or stochastic synthetic observations. D/S source labels distinguish selected rules, rather than alternative empirical datasets. Colour scales are specific to each agreement measure. These summaries compare partitions without assuming either method recovers the true transmission history.

## Source evidence and rebuilding the displays

This draft uses the following saved studies:

| Study | Run | Principal numerical evidence |
| --- | --- | --- |
| Diagnostics | `a529043958047472709cf4d6f2b390e9d1a39e2ff9c39acc53f2bba3abd927d8` | [Diagnostic results and source tables](fig01.md) |
| Baseline | `86dfffb165a13ab986c7` | [Held-out AP and operating metrics](../../01_synthetic_baseline/outputs/baseline/runs/86dfffb165a13ab986c7/report.md) |
| Sensitivity | `5d2a64dea949b408fbd2` | [Paired AP and operating-metric changes](../../02_synthetic_perturbation/outputs/perturbation/runs/5d2a64dea949b408fbd2/report.md) |
| Boston | `4d8b3723d4708fd95783` | [Exposure counts, cluster sizes, and partition agreement](../../03_boston_application/outputs/boston/runs/4d8b3723d4708fd95783/report.md) |

Figures 2 and 3 also have same-named CSV extracts beside their PDF/PNG files. Main-text synthetic values are equally weighted means over three observation realizations. Sensitivity differences are paired with fresh controls, and exposure recovery refers to one representative cluster. Boston evidence comes from the completed frozen-transfer report and complete graph/tree assessment; its manifest also records a subsequent unfinished exploration stage.

Rebuild these specific displays from the repository root in the evaluation environment:

```bash
python -m evaluation.results.build \
  --diagnostic-run evaluation/00_synthetic_diagnostics/outputs/diagnostics/runs/a529043958047472709cf4d6f2b390e9d1a39e2ff9c39acc53f2bba3abd927d8 \
  --baseline-run evaluation/01_synthetic_baseline/outputs/baseline/runs/86dfffb165a13ab986c7 \
  --perturbation-run evaluation/02_synthetic_perturbation/outputs/perturbation/runs/5d2a64dea949b408fbd2 \
  --boston-run evaluation/03_boston_application/outputs/boston/runs/4d8b3723d4708fd95783
```

Manuscript-to-script mapping: Figures 1–4 use `fig02`, `fig22`, `fig23`, and `fig18`; Tables 1–2 use `tab01` and `tab04`. Appendix tables use `tab02` and `tab03`. The builder also refreshes the standalone sensitivity heatmaps and Boston secondary-setting displays for further supplementary use. Figure captions above can be transferred to the manuscript; table captions are supplied by their `\input` files.
