# Evaluation methods

## Study design and synthetic benchmark

We evaluated whether EpiLink identifies close transmission relationships from genetic distances and sampling times, and whether its pairwise scores produce useful clusters. Our synthetic benchmark used infection-history and transmission-event outputs from the Scotland Coronavirus transmission Model (SCoVMod; Banks et al., 2022). The reconstructed, fully sampled transmission tree contained 5,051 cases. We kept this one tree fixed while generating separate observations for model training, method selection, and held-out assessment. Simulated dates and genetic distances followed the baseline EpiLink process assumptions, with deterministic and stochastic genetic-distance observations considered separately. Mutation expectations used a nominal 29,903-site genome; simulated sequences and their observed Hamming distances used 5,000 sites.

We defined relationship distance, M, using the full transmission tree. For an ancestor–descendant pair, M counts intervening cases; for pairs sharing an ancestor, it sums the intervening cases on both branches. The primary target, **M=0**, comprised direct transmission and pairs sharing an infector. M≤1 and M≤2 included progressively more distant relationships. Each observed, unordered pair was counted once, without self-pairs. The broader targets assessed the same scores trained for M=0; we did not fit new classifiers for them.

| Analysis stage | Observations or data | Purpose |
| --- | --- | --- |
| Synthetic diagnostics | Three development realizations | Examine feature ambiguity and known-truth graph/tree controls before method comparison. |
| Model training | Two separate realizations | Fit the supervised logistic comparators. |
| Baseline development | The three diagnosed realizations | Compare methods and choose pairwise and clustering settings. |
| Held-out evaluation | Three new realizations | Assess the previously frozen settings without reselection. |
| Parameter sensitivity | Three fresh realizations per scenario, paired with a fresh unperturbed control | Measure changes from the same-seed control at frozen settings. |
| Boston application | 772 published sequences | Describe how frozen settings group cases with recorded exposures. |

## Pairwise scores and comparators

EpiLink assigns a compatibility score to a pair from its genetic distance and absolute sampling-time difference (rounded to days). We compared **ED**, which uses deterministic genetic inference, and **ES**, which allows stochastic mutation counts. These are choices made by the *inference model*, not two kinds of genetic input. In the synthetic benchmark, a final D or S in a three-letter label identifies which genetic-distance observation process supplied the input: EDD and EDS use ED, while ESD and ESS use ES. We compared models on the same pairs within each observed process. For context, we also ranked pairs by genetic distance alone and fitted logistic regression using genetic and temporal distances. Logistic models were trained only on the separate training realizations; EpiLink compatibility scores were not treated as calibrated probabilities.

![The same SNP/day input grid scored by deterministic EpiLink ED and stochastic EpiLink ES.](../outputs/01_synthetic_baseline/86dfffb165a13ab986c7/fig00_primary_compatibility_surfaces.png)

*Figure (method illustration). Primary-target compatibility for ED and ES on the same 0–15 SNP by 0–20 day input grid, using the pinned baseline inference parameters. Colour represents a compatibility score, not a probability. [Figure script](../fig00.py).*

## Clustering and assessment

We converted pairwise scores into graphs at candidate cutoffs, then compared connected components and Leiden clustering (Traag et al., 2019). Leiden was evaluated with binary and, where applicable, native edge weights; its Constant Potts Model (CPM) resolution was varied on development observations. Independent restarts were chosen by the clustering objective rather than by transmission truth. For a phylogenetic comparator, we built genetic-distance trees with FastME, dated trees with TreeTime, and applied raw- and dated-tree TreeCluster methods (Balaban et al., 2019). The raw and dated pipelines used their respective genetic-distance and time-based thresholds.

Pairwise discrimination was summarised with precision–recall curves and average precision. At selected cutoffs we assessed precision, recall, and F1 against known relationships. For a cluster, the selected pairs were **all pairs within that cluster**, including pairs without a graph edge. We also recorded cluster sizes and the fraction of selected pairs with M≥3 as distant-pair contamination; this last measure does not include every false positive for M=0. Before comparing fitted methods, we examined whether exact genetic/temporal feature cells mixed target and non-target pairs, and assessed clusters formed from exact target edges or the known transmission topology. These diagnostic controls help interpret information loss and partitioning trade-offs; they are not universal performance ceilings ([diagnostics figure and results](fig01.md)).

## Development selection and held-out evaluation

Pairwise cutoffs and clustering settings were selected independently on development observations. Candidate pairwise cutoffs came from the observed development scores; graph and TreeCluster methods used their declared grids. The primary selection criterion maximised mean M=0 F1 across development realizations, with separate secondary selections for M≤1 and M≤2. Selected settings, fitted logistic models, and method definitions were then frozen and replayed unchanged on new held-out observations. No held-out result was used to choose a cutoff, clustering method, or resolution.

## Parameter sensitivity

We changed one generation parameter at a time while retaining the transmission tree. Five parameters were evaluated at 75% and 125% of baseline; clock relaxation used the absolute values below. The six parameters at two levels gave 12 perturbed scenarios, plus an unperturbed control generated using the same fresh seeds. Baseline biological parameters drew on published SARS-CoV-2 estimates (Hart et al., 2021; Duchêne et al., 2020; van Dorp et al., 2020).

| Generation parameter | Baseline value | Perturbed levels |
| --- | ---: | --- |
| Incubation mean | 5.505 days | 0.75×, 1.25× |
| Incubation coefficient of variation | 0.415 | 0.75×, 1.25× |
| Testing-delay mean | 1.0 day | 0.75×, 1.25× |
| Testing-delay coefficient of variation | 1.0 | 0.75×, 1.25× |
| Substitution rate | 0.001 substitutions/site/year | 0.75×, 1.25× |
| Clock relaxation | 0.33 | 0, 0.66 (absolute values) |

Each scenario was scored both with its own biological parameters (**matched**) and with the baseline biological parameters (**baseline-fixed**). These two modes change EpiLink's parameter inputs, not whether ED or ES inference is used. We compared each scenario with its same-seed, same-mode fresh control; the previously held-out baseline observations were not the control. Logistic models and all selected operating settings stayed fixed, so this analysis measured sensitivity without retraining or retuning.

## Boston application and summaries

We applied the baseline-frozen scoring rules, fitted logistic models, and clustering definitions to 772 published Boston SARS-CoV-2 sequences and exposure metadata (Lemieux et al., 2021). The graph input used pairwise TN93 genetic distances reported below 0.0005 substitutions per site and absolute collection-date differences; missing, distance-censored pairs were unobserved, not zero-distance pairs. All EpiLink rules received this **same empirical input**. Their D/S suffixes retain the names of the synthetic source rules, rather than denoting separate Boston datasets. Raw and dated TreeCluster comparisons used trees reconstructed from the alignment, independently of the censored graph-distance table. No setting was selected using Boston exposure labels.

With no complete Boston transmission tree, we described cluster sizes, Conference and skilled-nursing-facility exposure concentration and recovery, and graph–tree partition agreement rather than claiming transmission accuracy. For each exposure, the representative eligible cluster contained the most labelled cases; concentration was their fraction of that cluster, and recovery was their fraction of all cases with that exposure. Across synthetic realizations we reported equally weighted means and descriptive standard deviations or ranges; paired sensitivity changes used each realization's fresh control. Pairs and observations on the same fixed tree were not treated as independent epidemic replicates. Full parameter grids, metric definitions, and pinned run provenance are recorded in the [baseline](../../01_synthetic_baseline/README.md), [perturbation](../../02_synthetic_perturbation/README.md), and [Boston](../../03_boston_application/README.md) protocols and the [output reference](../../../OUTPUTS.md).
