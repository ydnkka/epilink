# epilink appendix methods subsection

## Formal definition of epilink

`epilink` is a pairwise transmission-linkage model that combines temporal information and viral genetic distance to quantify whether two observed cases are compatible with recent transmission. For each pair of cases $(i,j)$, the observed inputs are a temporal distance $t_{ij}$ (in days) and a genetic distance $g_{ij}$ (in consensus-level nucleotide differences). The model returns the linkage estimate

$$
\hat{P}_A(i,j \mid g_{ij}, t_{ij}) = K_T(t_{ij}) \sum_{m \in A} \tilde{K}_G(g_{ij}, m),
$$

where $K_T(t_{ij})$ is the temporal compatibility of the pair, $\tilde{K}_G(g_{ij},m)$ is the normalized genetic compatibility under a transmission history with $m$ intermediate hosts, and $A$ is the set of intermediary counts included in the target linkage definition. Throughout the main analyses, the default setting was $A=\{0\}$, corresponding to a zero-intermediate linkage definition. In the genetic component, this $m=0$ class includes both direct transmission and co-primary infection from a shared source. In the temporal component, co-primary cases are assumed to have been infected at most one generation interval apart.

## Natural-history model

The temporal component of `epilink` is based on a mechanistic $E/P/I$ infectiousness model for SARS-CoV-2, following Hart et al. (2021). Infection is followed by a latent stage ($E$), a presymptomatic infectious stage ($P$), and a symptomatic infectious stage ($I$). Stage durations are modeled with Gamma distributions:

$$
E \sim \mathrm{Gamma}(k_E, \theta_{\mathrm{inc}}), \qquad
P \sim \mathrm{Gamma}(k_P, \theta_{\mathrm{inc}}), \qquad
I \sim \mathrm{Gamma}(k_I, \theta_I),
$$

with

$$
k_P = k_{\mathrm{inc}} - k_E, \qquad
\theta_I = \frac{1}{k_I \mu}.
$$

The incubation period is therefore

$$
U = E + P.
$$

Let $Y$ denote the time from infectiousness onset to transmission. In `epilink`, $Y$ is sampled from the infectiousness-to-transmission density

$$
f_Y(y) =
\begin{cases}
0, & y < 0, \\
C \left[
\alpha \left(1 - F_P(y)\right)
+
\int_0^y \left(1 - F_I(y-z)\right) f_P(z)\, dz
\right], & y \ge 0,
\end{cases}
$$

where $f_P$ and $F_P$ are the density and cumulative distribution function of the presymptomatic stage, $F_I$ is the cumulative distribution function of the symptomatic stage, $\alpha$ is the relative presymptomatic infectiousness, and $C$ is a normalization constant chosen so that $f_Y$ integrates to one. The generation interval is then

$$
G = E + Y.
$$

This formulation allows the temporal component of `epilink` to reflect presymptomatic transmission and uncertainty in incubation and generation intervals.

## Temporal compatibility

Under the zero-intermediate temporal model, the observed temporal distance between cases is constrained by the generation interval and the difference in incubation periods. This covers either direct transmission between the sampled cases or co-primary infection from a shared source, under the assumption that co-primary infections occur no more than one generation interval apart.

For each Monte Carlo draw $n = 1, \ldots, N$, `epilink` samples incubation periods $U_i^{(n)}$ and $U_j^{(n)}$ and a generation interval $G^{(n)}$. Temporal compatibility is then estimated as

$$
K_T(t_{ij}) =
\frac{1}{N}
\sum_{n=1}^{N}
\mathbf{1}
\left(
\left| t_{ij} + U_i^{(n)} - U_j^{(n)} \right|
\le G^{(n)}
\right),
$$

where $\mathbf{1}(\cdot)$ denotes the indicator function. Thus, $K_T(t_{ij})$ is the proportion of simulated draws for which the observed temporal gap is compatible with the sampled natural-history realization under the zero-intermediate model.

## Molecular clock model

The genetic component models the accumulation of substitutions along the transmission history separating two sampled genomes. `epilink` uses a molecular clock with median substitution rate $\bar{r}$ per site per year and genome length $L$. Under a strict clock, the substitution rate is fixed at $\bar{r}$. Under a relaxed clock, branch-specific rates are sampled from a log-normal distribution,

$$
r^{(n)} \sim \log \mathcal{N}(\log \bar{r}, \sigma^2),
$$

and converted to substitutions per genome per day. For an observed genetic distance $g_{ij}$, draw $n$ implies an approximate observed time to most recent common ancestor (TMRCA),

$$
\tau_{\mathrm{obs}}^{(n)} = \frac{g_{ij}}{2r^{(n)}}.
$$

## Genetic compatibility across transmission depths

To evaluate whether the observed genetic distance is compatible with plausible transmission histories, `epilink` simulates scenario-specific expected TMRCAs. For each draw, the model considers both a transmission-chain configuration and a common-source configuration. When $m=0$, these correspond to direct transmission and co-primary infection from a shared source, respectively.

If $m$ intermediate hosts are allowed, the expected TMRCA under the chain configuration is

$$
\tau_{\mathrm{chain},m}^{(n)} =
\delta_i^{(n)} + \delta_j^{(n)} + \sum_{\ell=1}^{m} G_{\ell}^{(n)},
$$

where $G_{\ell}^{(n)}$ are simulated generation intervals for the intermediate transmissions. In the current implementation, the draw-specific delays from the putative transmission event to the two sampled cases are

$$
\delta_i^{(n)} = \left| G^{(n)} - U_i^{(n)} \right|, \qquad
\delta_j^{(n)} = U_j^{(n)}.
$$

Under the common-source configuration,

$$
\tau_{\mathrm{cs},m}^{(n)} =
\Delta_{\mathrm{inf}}^{(n)} + U_i^{(n)} + U_j^{(n)} + \sum_{\ell=1}^{m} G_{\ell}^{(n)},
$$

where

$$
\Delta_{\mathrm{inf}}^{(n)} = \left| Y_a^{(n)} - Y_b^{(n)} \right|
$$

is a simulated infection-time separation between the two lineages obtained from independently sampled transmission offsets.

`epilink` assigns a raw genetic compatibility weight to each intermediary count $m$ by counting the fraction of Monte Carlo draws for which the observed and simulated TMRCAs agree within one sampled generation interval:

$$
K_G(g_{ij}, m) =
\frac{1}{N}
\sum_{n=1}^{N}
\left[
\mathbf{1}\left(
\left| \tau_{\mathrm{obs}}^{(n)} - \tau_{\mathrm{chain},m}^{(n)} \right|
\le G^{(n)}
\right)
+
\mathbf{1}\left(
\left| \tau_{\mathrm{obs}}^{(n)} - \tau_{\mathrm{cs},m}^{(n)} \right|
\le G^{(n)}
\right)
\right].
$$

These raw weights are normalized across all intermediary counts from $0$ to a user-specified maximum $M$,

$$
\tilde{K}_G(g_{ij}, m) =
\frac{K_G(g_{ij}, m)}{\sum_{m'=0}^{M} K_G(g_{ij}, m')},
$$

so that the genetic term acts as a relative allocation of support across different transmission depths. The final linkage estimate sums the normalized weights over the selected set $A$.

## Monte Carlo estimation

All quantities in `epilink` are estimated by Monte Carlo simulation. For each case pair, the algorithm:

1. Samples incubation periods for the two cases.
2. Samples one or more generation intervals from the infectiousness model.
3. Samples transmission offsets used to represent infection-time differences between lineages.
4. Samples molecular clock rates under either a strict or relaxed clock.
5. Computes temporal compatibility, genetic compatibility across $m = 0, \ldots, M$, and the final linkage estimate.

In the current implementation, the default output uses only $m=0$ in the final summation. Broader linkage definitions can be obtained by including additional intermediary counts, for example $A=\{0,1,2\}$.

`epilink` is therefore intended to provide a transparent, mechanistically informed pairwise linkage measure rather than a full reconstruction of the transmission tree. This framework is particularly useful when symptom-onset timing and genome sequences are available for many candidate pairs and the goal is to prioritize or quantify plausible epidemiological links within a common natural-history and molecular-clock model.
