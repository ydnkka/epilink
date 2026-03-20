## Methods

### EpiLink compatibility model overview

For each sampled pair of cases $i$ and $j$, `EpiLink` assesses whether the observed testing-time difference and consensus-level genetic distance are compatible with a finite set, $\mathcal{S}M$, of _recent latent transmission_ histories (Figure 1). Let $t_{\mathrm{test},k}$ denote the testing time of sampled case $k \in {i,j}$. Let $t_{i,j}$ denote the observed signed difference in testing times, and $g_{ij}$ the observed consensus-level genetic distance, measured as the number of nucleotide differences between the sampled genomes. Candidate latent histories are drawn from
$$
\mathcal{S}_M = \left\{H_{\mathrm{AD}}(m): m=0,\ldots,M\right\} \cup \left\{H_{\mathrm{CA}}(m_i,m_j): m_i,m_j \ge 0,\; m_i+m_j \le M\right\}
$$
where $H_{\mathrm{AD}}(m)$ denotes an ancestor-descendant history with $m$ unsampled intermediates and $H_{\mathrm{CA}}(m_i,m_j)$ denotes a common-ancestor history in which the two sampled cases descend from a shared unsampled source with branch depths $m_i$ and $m_j$. The special cases $H_{\mathrm{AD}}(0)$ and $H_{\mathrm{CA}}(0,0)$ correspond to direct transmission and co-primary infection, respectively.

![Latent transmission histories considered by `EpiLink`.](epilink_scenarios.svg)
***Figure 1: Latent transmission histories considered by `EpiLink`**. Schematic representation of the two families of latent transmission histories linking a sampled pair of cases $i$ and $j$. In the ancestor--descendant history $H_{\mathrm{AD}}(m)$, case $j$ descends from case $i$ through $m$ unsampled intermediates; the special case $m=0$ corresponds to direct transmission. In the common-ancestor history $H_{\mathrm{CA}}(m_i,m_j)$, both sampled cases descend from a shared unsampled source, with branch depths $m_i$ and $m_j$; the special case $(m_i,m_j)=(0,0)$ corresponds to co-primary infection. Open circles denote sampled cases and shaded circles denote unsampled latent cases. These latent histories define the scenario set over which temporal and genetic compatibility scores are evaluated.*


The temporal component extends the $E/P/I$ infectiousness framework of [1] by incorporating the delay from symptom onset to testing. The latent ($E$), presymptomatic infectious ($P$), and symptomatic infectious ($I$) stages are each modelled with Gamma distributions, with fitted parameter values, and the testing delay is likewise modelled as a Gamma-distributed quantity. Let $y_E$, $y_P$, and $y_I$ denote the durations of the three stages, and let $x_{\mathrm{test}}$ denote the onset-to-test delay. Defining the incubation period as $\tau_{\mathrm{inc}} = y_E + y_P$, and the generation interval as $\tau_{\mathrm{gen}} = \tau_{\mathrm{inc}} + y_{\mathrm{toit}}$, where $y_{\mathrm{toit}}$ is the time from onset of infectiousness to transmission, the model-implied difference in testing times under scenario $s$ is
$$
T_s = A_j(s) - A_i(s) + \tau_{\mathrm{inc},j} + x_{\mathrm{test},j} - \tau_{\mathrm{inc},i} - x_{\mathrm{test},i}
$$
where $A_k(s)$ denotes the elapsed time from the scenario-specific reference infection to infection of sampled case $k$. Monte Carlo sampling from these latent timing distributions yields the scenario-specific temporal distribution against which the observed testing-time difference $t_{ij}$ is compared.

The genetic component assumes that within-host divergence is negligible on the timescales of interest and that consensus divergence accumulates along transmission-linked lineages. For each scenario, we define an effective transmission-related branch length $B_s$ in days separating the sampled genomes. If $r$ is the median substitution rate per site per year and $L$ is genome length, the corresponding per-genome daily substitution rate is $\lambda = rL/365$. Given branch-length draw $B_s^{(n)}$, the genetic draw is taken as either $G_s^{(n)} = \lambda B_s^{(n)}$ under a deterministic mutational process or
$$
G_s^{(n)} \mid B_s^{(n)}, \lambda \sim \mathrm{Poisson}\!\left(\lambda B_s^{(n)}\right)
$$
under a stochastic mutational process. A relaxed uncorrelated log-normal clock can be accommodated by drawing a branch-specific rate $\lambda^{(n)}$.

Observed temporal and genetic distances are scored separately using percentile-based compatibility functions and then combined multiplicatively. For any observed value $x_{\mathrm{obs}}$ and scenario-specific Monte Carlo draws, we compute the empirical percentile $p_s(x_{\mathrm{obs}})$ and corresponding centrality score $C_s(x_{\mathrm{obs}})=1-2|p_s(x_{\mathrm{obs}})-0.5|$. Applying this to $t_{ij}$ and $g_{ij}$ yields temporal and genetic compatibilities, $C_{T,s}(i,j)$ and $C_{G,s}(i,j)$, and the scenario-level compatibility
$$
C_s(i,j)=C_{T,s}(i,j)\,C_{G,s}(i,j)
$$
To represent broader hypotheses, compatibilities may be summed across a user-defined target subset $\mathcal{S}_\star \subseteq \mathcal{S}_M$. These scores quantify agreement with the chosen transmission scenarios rather than calibrated posterior probabilities.

### References

1. Hart, William S., Philip K. Maini, and Robin N. Thompson (2021) 
High infectiousness immediately before COVID-19 symptom onset 
highlights the importance of continued contact tracing. eLife 10:e65534.


## Supplementary Information

### S1. Detailed derivation of the EpiLink compatibility model

#### S1.1 Overview

`EpiLink` evaluates whether an observed pair of sampled cases is compatible with a candidate transmission scenario by combining epidemiological timing and genomic divergence. For each sampled pair $(i,j)$, the observed data are the testing-time difference
$$
t_{ij} = t_{\mathrm{test},j} - t_{\mathrm{test},i},
$$
and the observed consensus-level genetic distance $g_{ij}$. Each candidate scenario $s$ induces a latent distribution over expected testing-time differences and transmission-related branch lengths, which in turn defines an expected distribution of genetic distances. Compatibility is quantified by comparing the observed pair with these scenario-specific Monte Carlo distributions.

#### S1.2 Natural-history model and observed times

The temporal model is based on an $E/P/I$ infection process with Gamma-distributed latent ($E$), presymptomatic infectious ($P$), and symptomatic infectious ($I$) stages, together with a Gamma-distributed delay from symptom onset to testing. Let $y_E$, $y_P$, and $y_I$ denote the stage durations, and let $x_{\mathrm{test}}$ denote the delay from symptom onset to testing. We write
$$
\tau_{\mathrm{inc}} = y_E + y_P, \qquad \tau_{\mathrm{gen}} = \tau_{\mathrm{inc}} + y_{\mathrm{toit}}, \tag{S1}
$$
where $y_{\mathrm{toit}}$ is the time from onset of infectiousness to transmission. The total time from infection to testing is therefore $\tau_{\mathrm{inc}} + x_{\mathrm{test}}$.

For a given scenario $s$, let $t_x$ denote the infection time of a latent reference point: the infection time of case $i$ under an ancestor-descendant scenario, or the infection time of the shared unsampled source under a common-ancestor scenario. Let $A_k(s)$ denote the elapsed time from this reference point to infection of sampled case $k \in \{i,j\}$. Then
$$
t_{\mathrm{test},k}  = t_x + A_k(s) + \tau_{\mathrm{inc},k} + x_{\mathrm{test},k}. \tag{S2}
$$

#### S1.3 Scenario space

Let $M$ denote the maximum hidden depth allowed in the latent relationship between the two sampled cases. The candidate scenario set is
$$
\mathcal{S}_M = \left\{H_{\mathrm{AD}}(m): m=0,\ldots,M\right\} \cup \left\{H_{\mathrm{CA}}(m_i,m_j): m_i,m_j \ge 0,\; m_i+m_j \le M\right\}. \tag{S3}
$$
Here, $H_{\mathrm{AD}}(m)$ denotes an ancestor-descendant scenario in which case $j$ descends from case $i$ through $m$ unsampled intermediates, whereas $H_{\mathrm{CA}}(m_i,m_j)$ denotes a common-ancestor scenario in which both sampled cases descend independently from a shared unsampled source. The cases $H_{\mathrm{AD}}(0)$ and $H_{\mathrm{CA}}(0,0)$ correspond to direct transmission and co-primary infection, respectively.

The common-ancestor labels are ordered. Thus, $H_{\mathrm{CA}}(m_i,m_j)$ and $H_{\mathrm{CA}}(m_j,m_i)$ are generally distinct because temporal compatibility depends on the signed testing-time difference $t_{ij}$.

#### S1.4 Temporal compatibility variable

Subtracting Eq. (S2) for case $i$ from Eq. (S2) for case $j$ gives the model-implied testing-time difference under scenario $s$:
$$
T_s = A_j(s) - A_i(s) + \tau_{\mathrm{inc},j} + x_{\mathrm{test},j} - \tau_{\mathrm{inc},i} - x_{\mathrm{test},i}. \tag{S4}
$$
This variable combines three sources of variation: transmission structure through $A_j(s)-A_i(s)$, biological variability through the incubation periods, and surveillance variability through the testing delays.

For ancestor-descendant scenarios,
$$
A_i\!\left(H_{\mathrm{AD}}(m)\right)=0, \qquad A_j\!\left(H_{\mathrm{AD}}(m)\right)=\sum_{r=0}^{m}\tau_{\mathrm{gen},r}, \tag{S5}
$$
so that
$$T_{ij}\!\left(H_{\mathrm{AD}}(m)\right) = \sum_{r=0}^{m}\tau_{\mathrm{gen},r} +  \tau_{\mathrm{inc},j} + x_{\mathrm{test},j} - \tau_{\mathrm{inc},i} - x_{\mathrm{test},i}. \tag{S6}
$$

For common-ancestor scenarios,
$$
A_i\!\left(H_{\mathrm{CA}}(m_i,m_j)\right) = \sum_{r=1}^{m_i+1}\tau_{\mathrm{gen},r}^{(i)}, \qquad
A_j\!\left(H_{\mathrm{CA}}(m_i,m_j)\right) = \sum_{s=1}^{m_j+1}\tau_{\mathrm{gen},s}^{(j)}, \tag{S7}
$$
which gives
$$
T_{ij}\!\left(H_{\mathrm{CA}}(m_i,m_j)\right) = \sum_{s=1}^{m_j+1}\tau_{\mathrm{gen},s}^{(j)} -
\sum_{r=1}^{m_i+1}\tau_{\mathrm{gen},r}^{(i)} + \tau_{\mathrm{inc},j} + x_{\mathrm{test},j} - \tau_{\mathrm{inc},i} - x_{\mathrm{test},i}. \tag{S8}
$$
For each scenario $s$, Monte Carlo simulation yields draws $T_s^{(1)},\ldots,T_s^{(N)}$ from the induced temporal distribution.

#### S1.5 Branch length and genetic model

To link transmission timing to consensus genetic divergence, we define an effective transmission-related branch length $B_s$, measured in days, separating the sampled genomes. Under ancestor-descendant scenarios,
$$
B_{ij}\!\left(H_{\mathrm{AD}}(m)\right)
=
\sum_{r=0}^{m}\tau_{\mathrm{gen},r} + \tau_{\mathrm{inc},j} + x_{\mathrm{test},j} - \left(\tau_{\mathrm{inc},i} + x_{\mathrm{test},i}\right). \tag{S9}
$$
Thus, for ancestor-descendant histories, the temporal contrast and the branch length are governed by the same latent quantity:
$$
B_{ij}\!\left(H_{\mathrm{AD}}(m)\right)=T_{ij}\!\left(H_{\mathrm{AD}}(m)\right).
$$

Under common-ancestor scenarios, branch length accumulates along both descendant lineages:
$$
B_{ij}\!\left(H_{\mathrm{CA}}(m_i,m_j)\right) = \sum_{r=1}^{m_i+1}\tau_{\mathrm{gen},r}^{(i)} +
\sum_{s=1}^{m_j+1}\tau_{\mathrm{gen},s}^{(j)} + \tau_{\mathrm{inc},i} + x_{\mathrm{test},i} + \tau_{\mathrm{inc},j} + x_{\mathrm{test},j}. \tag{S10}
$$
This distinction is central: $T_s$ is a signed difference that governs temporal ordering, whereas $B_s$ is a summed lineage length that governs genetic divergence.

Let $r$ denote the median substitution rate per site per year and $L$ the genome length. The corresponding per-genome daily substitution rate is
$$
\lambda = \frac{rL}{365}. \tag{S11}
$$
Under a relaxed uncorrelated log-normal clock, one instead samples
$$
r^{(n)} \sim \mathrm{LogNormal}(\mu_r,\sigma_r^2), \qquad  \lambda^{(n)} = \frac{r^{(n)}L}{365}, \tag{S12}
$$
with $\mu_r$ chosen so that the median equals the specified substitution rate.

Given branch-length draw $B_s^{(n)}$, the genetic draw used for scoring is
$$
G_s^{(n)} = \lambda^{(n)} B_s^{(n)} \tag{S13}
$$
under a deterministic mutational process, or
$$
G_s^{(n)} \mid B_s^{(n)}, \lambda^{(n)} \sim \mathrm{Poisson}\!\left(\lambda^{(n)} B_s^{(n)}\right)\tag{S14}
$$
under a stochastic mutational process. This formulation assumes that within-host divergence is negligible on the timescales of interest and that consensus divergence accumulates primarily along transmission-linked lineages.

#### S1.6 Monte Carlo estimation and compatibility scoring

For each scenario $s$, `EpiLink` generates Monte Carlo draws of the temporal and genetic variables:
$$
\left\{T_s^{(n)}, B_s^{(n)}, G_s^{(n)}\right\}_{n=1}^{N}.
$$
Observed quantities are then scored against the corresponding scenario-specific empirical distributions. For any observed quantity $x_{\mathrm{obs}}$ and scenario-specific draws $x_1^{(s)},\ldots,x_{n_s}^{(s)}$, define
$$
p_s(x_{\mathrm{obs}}) = \frac{\#\left\{i : x_i^{(s)} \le x_{\mathrm{obs}}\right\}}{n_s}, \qquad
C_s(x_{\mathrm{obs}}) = 1 - 2\left|p_s(x_{\mathrm{obs}})-0.5\right|. \tag{S15}
$$
Applying this to the temporal and genetic components gives
$$
C_{T,s}(i,j)=C_s(t_{ij}), \qquad C_{G,s}(i,j)=C_s(g_{ij}),
$$
and the scenario-level compatibility
$$
C_s(i,j)=C_{T,s}(i,j)\,C_{G,s}(i,j). \tag{S16}
$$

To represent broader scientific hypotheses, compatibilities may be summed across a user-defined target subset $\mathcal{S}_\star \subseteq \mathcal{S}_M$:
$$
\mathrm{score}_{\mathcal{S}_\star}(i,j) = \sum_{s \in \mathcal{S}_\star} C_s(i,j). \tag{S17}
$$
When $\mathcal{S}_\star$ contains a single scenario, this reduces to the corresponding single-scenario compatibility. When multiple scenarios are aggregated, the summed target score can exceed 1 because it accumulates compatibility across scenarios.

#### S1.7 Interpretation and computational implementation

This framework should be interpreted as a compatibility model rather than a calibrated probabilistic transmission model. High values indicate that the observed timing and genetic distance lie near the centre of the simulated distributions induced by a scenario or target subset; low values indicate poor alignment with the chosen model assumptions and latent-history set. Accordingly, compatibility scores are not posterior probabilities, and low scores do not by themselves prove absence of epidemiological linkage.

For computational efficiency, scenario-specific temporal and branch-length draws are precomputed and cached when the `EpiLink` object is constructed. Cached branch-length draws are then converted into cached genetic draws using the selected mutational process. Pairwise evaluation therefore requires only percentile lookups against stored Monte Carlo samples, making the method well suited to large datasets.

In summary, `EpiLink` links epidemiological and genomic evidence through a shared latent transmission process. Transmission scenarios determine both the expected testing-time difference and the evolutionary separation of the sampled genomes, and compatibility is assessed by asking whether the observed pair is centrally located within the corresponding simulated distributions.

