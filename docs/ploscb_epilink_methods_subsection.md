# epilink methods subsection

## Pairwise linkage inference with epilink

We developed `epilink`, a pairwise transmission-linkage model that combines temporal information and viral genetic distance to quantify whether two observed cases are compatible with recent transmission. For each pair of cases $(i,j)$, the inputs are an observed temporal distance $t_{ij}$ (in days) and an observed genetic distance $g_{ij}$ (in consensus-level nucleotide differences, e.g. SNPs). The model returns a linkage score

$$
\hat{P}_A(i,j \mid g_{ij}, t_{ij}) = K_T(t_{ij}) \sum_{m \in A} \tilde{K}_G(g_{ij}, m),
$$

where $K_T(t_{ij})$ is the temporal compatibility of the pair, $\tilde{K}_G(g_{ij},m)$ is the normalized genetic compatibility under a transmission path containing $m$ intermediate hosts, and $A$ is the set of intermediary counts treated as compatible with the target linkage definition. Unless otherwise stated, we used the default setting $A=\{0\}$, corresponding to a zero-intermediate linkage definition. In the genetic component, this $m=0$ class includes both direct transmission between the sampled cases and co-primary infection from a shared source.

## Natural-history model

The temporal component of `epilink` is based on a mechanistic $E/P/I$ infectiousness model for SARS-CoV-2, following the formulation of Hart et al. (2021). Briefly, infection is followed by a latent stage ($E$), a presymptomatic infectious stage ($P$), and a symptomatic infectious stage ($I$). Stage durations are modeled with Gamma distributions:

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

Let $Y$ denote the time from infectiousness onset to transmission. In `epilink`, $Y$ is drawn from the infectiousness-to-transmission density

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

where $f_P$ and $F_P$ are the density and cumulative distribution function of the presymptomatic period, $F_I$ is the cumulative distribution function of the symptomatic stage, $\alpha$ is the relative infectiousness before symptom onset, and $C$ is a normalization constant chosen so that $f_Y$ integrates to one. The generation interval is then

$$
G = E + Y.
$$

This construction allows the temporal part of the model to reflect both presymptomatic transmission and uncertainty in incubation and generation intervals.

## Temporal compatibility

Under the zero-intermediate temporal model, the observed temporal distance between cases is constrained by the generation interval and the difference in incubation periods. This covers either direct transmission between the sampled cases or co-primary infection from a shared source, under the assumption that the two co-primary infections occur at most one generation interval apart. In each Monte Carlo draw $n = 1, \ldots, N$, `epilink` samples incubation periods $U_i^{(n)}$ and $U_j^{(n)}$ and a generation interval $G^{(n)}$. Temporal compatibility is then estimated as

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

where $\mathbf{1}(\cdot)$ denotes the indicator function. Thus, $K_T(t_{ij})$ is the proportion of simulated draws for which the observed gap is consistent with the sampled natural-history realization under either direct transmission or co-primary infection separated by no more than one generation interval. Small temporal gaps that align with plausible incubation-period differences and generation intervals receive high temporal support, whereas implausibly large or mismatched gaps receive low support.

## Molecular clock model and genetic compatibility

The genetic component models the accumulation of substitutions along the transmission history separating two sampled genomes. `epilink` uses a molecular clock with median substitution rate $\bar{r}$ per site per year and genome length $L$. Under a strict clock, the substitution rate is fixed at $\bar{r}$; under a relaxed clock, branch-specific rates are sampled from a log-normal distribution,

$$
r^{(n)} \sim \log \mathcal{N}(\log \bar{r}, \sigma^2),
$$

and converted to substitutions per genome per day. For an observed genetic distance $g_{ij}$, draw $n$ implies an approximate observed time to most recent common ancestor (TMRCA)

$$
\tau_{\mathrm{obs}}^{(n)} = \frac{g_{ij}}{2 r^{(n)}}.
$$

To compare this with epidemiologically plausible transmission histories, `epilink` simulates scenario-specific expected TMRCAs. For each draw, the model considers both a transmission-chain configuration and a common-source configuration. When $m=0$, these correspond to direct transmission and co-primary infection from a shared source, respectively. If $m$ intermediate hosts are allowed, the expected TMRCA under the chain scenario is

$$
\tau_{\mathrm{chain},m}^{(n)} =
\delta_i^{(n)} + \delta_j^{(n)} + \sum_{\ell=1}^{m} G_{\ell}^{(n)},
$$

where $G_{\ell}^{(n)}$ are simulated generation intervals for the intermediate transmissions. In the current implementation, the draw-specific delays from the putative transmission event to the two sampled cases are

$$
\delta_i^{(n)} = \left| G^{(n)} - U_i^{(n)} \right|, \qquad
\delta_j^{(n)} = U_j^{(n)}.
$$

Under the common-source alternative,

$$
\tau_{\mathrm{cs},m}^{(n)} =
\Delta_{\mathrm{inf}}^{(n)} + U_i^{(n)} + U_j^{(n)} + \sum_{\ell=1}^{m} G_{\ell}^{(n)},
$$

where $\Delta_{\mathrm{inf}}^{(n)} = |Y_a^{(n)} - Y_b^{(n)}|$ is a simulated infection-time separation between the two lineages obtained from independently sampled transmission offsets. `epilink` assigns a raw genetic compatibility weight to each intermediary count $m$ by counting the fraction of draws for which the observed and simulated TMRCAs agree within one sampled generation interval:

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

These raw weights are then normalized across all intermediary counts from $0$ to a user-specified maximum $M$,

$$
\tilde{K}_G(g_{ij}, m) =
\frac{K_G(g_{ij}, m)}{\sum_{m'=0}^{M} K_G(g_{ij}, m')},
$$

so that the genetic term acts as a relative allocation of support across different transmission depths. The final linkage estimate sums the normalized weights over the set $A$ of intermediary counts deemed compatible with the target definition of linkage.

## Monte Carlo estimation and default settings

All quantities are estimated by Monte Carlo simulation. For each pair, `epilink` samples incubation periods, generation intervals, transmission offsets, and molecular clock rates, and then evaluates the temporal and genetic compatibility functions above. In the current implementation, the default output uses only $m=0$ in the final summation, corresponding to a zero-intermediate linkage definition. In the genetic component, this includes both direct transmission and co-primary infection from a shared source, although broader linkage definitions can be obtained by including additional intermediary counts (for example, $A=\{0,1,2\}$).

In practice, the method is intended to provide a transparent, mechanistically informed pairwise linkage score rather than a full reconstruction of the transmission tree. This makes it well suited to settings where symptom-onset timing and genome sequences are available for many candidate pairs, and where the goal is to prioritize or quantify plausible epidemiological links using a common natural-history and molecular-clock framework.
