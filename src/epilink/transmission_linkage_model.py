"""
Estimate the probability of a recent transmission link between two cases.

This module combines genetic distance (SNP counts) and temporal distance (days) to estimate
the probability of a recent transmission link between two cases.

The implementation uses two main numerical kernels:

- **Temporal kernel** (:func:`_temporal_kernel`): Evaluates the likelihood of observed
  temporal spacing between case sampling dates, given epidemiological parameters
  (incubation periods, generation intervals).
- **Genetic kernel** (:func:`_genetic_kernel`): Evaluates the likelihood of observed
  genetic distance (SNP counts) as a function of molecular clock rate and evolutionary
  time (TMRCA), considering scenarios with 0 or more intermediate hosts.

Public API
----------
* :func:`linkage_probability` - Compute linkage probability with optional caching optimization
* :func:`pairwise_linkage_probability_matrix` - Evaluate over distance grids
* :func:`temporal_linkage_probability` - Temporal evidence only
* :func:`genetic_linkage_probability` - Genetic evidence only
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from .infectiousness_profile import TOIT, MolecularClock

ArrayLike = npt.ArrayLike
NDArrayFloat = npt.NDArray[np.float64]


# =============================================================================
# Numba kernels and Monte Carlo simulation
# =============================================================================


@njit(parallel=True, fastmath=True)
def _temporal_kernel(
    temporal_distance_ij: np.ndarray,
    diff_incubation_ij: np.ndarray,
    generation_interval: np.ndarray,
) -> np.ndarray:
    r"""
    Monte Carlo estimate of temporal evidence for transmission linkage.

    For each observed temporal distance :math:`t_i` (days between case sampling dates),
    evaluates the fraction of Monte Carlo scenarios where the adjusted interval is
    consistent with transmission:

    .. math::

        P_t(i) = \frac{1}{N} \sum_{j=1}^{N} \mathbb{1}\left\{ |t_i + \Delta\tau_{inc}| \leq GI \right\}

    where :math:`\Delta\tau_{inc}` is the incubation difference and :math:`GI` is
    the generation interval.

    Parameters
    ----------
    temporal_distance_ij : numpy.ndarray
        Observed intervals between case sampling dates (days), shape ``(K,)``.
    diff_incubation_ij : numpy.ndarray
        Simulated differences in incubation periods, shape ``(N,)``.
    generation_interval : numpy.ndarray
        Simulated generation intervals, shape ``(N,)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(K,)`` array of temporal evidence P_t values.
        Entry [k] = fraction of scenarios where the adjusted interval is valid.
    """
    num_distances = temporal_distance_ij.shape[0]
    mc_simulations = diff_incubation_ij.shape[0]
    out = np.empty(num_distances, dtype=np.float64)
    for i in prange(num_distances):
        t = temporal_distance_ij[i]
        count = 0
        for j in range(mc_simulations):
            if abs(t + diff_incubation_ij[j]) <= generation_interval[j]:
                count += 1
        out[i] = count / mc_simulations
    return out



@njit(parallel=True, fastmath=True)
def _genetic_kernel(
        genetic_distance_ij: np.ndarray,
        clock_rates: np.ndarray,
        sampling_delay_i: np.ndarray,
        sampling_delay_j: np.ndarray,
        intermediate_generations: np.ndarray,
        intermediates: int,
        diff_infection_ij: np.ndarray,
        incubation_periods: np.ndarray,
        generation_time_xi: np.ndarray,
) -> np.ndarray:
    r"""
    Compute genetic evidence for transmission linkage across intermediate scenarios.

    For each observed genetic distance (SNP count), evaluates consistency with
    expected accumulation under direct transmission (M=0) or chains through
    intermediate hosts (M>0). Uses Monte Carlo integration to compute the
    fraction of scenarios where observed TMRCA matches expected TMRCA.

    Parameters
    ----------
    genetic_distance_ij : numpy.ndarray
        Nucleotide differences between two sequences i and j, shape ``(K,)``.
    clock_rates : numpy.ndarray
        Sampled substitution rates (mutations/day), shape ``(N,)``.
    sampling_delay_i : numpy.ndarray
        Time from transmission to sampling for case i (days), shape ``(N,)``.
        For direct transmission (i → j): approximately :math:`|\tau_g - \tau_{inc}|`.
    sampling_delay_j : numpy.ndarray
        Time from transmission to sampling for case j (days), shape ``(N,)``.
        For direct transmission (i → j): approximately :math:`\tau_{inc}`.
    intermediate_generations : numpy.ndarray
        Generation intervals for transmission segments, shape ``(N, M+1)``.
        Column 0 = direct generation interval; columns 1..M = intermediate steps.
    intermediates : int
        Maximum number of intermediate hosts (M).
    diff_infection_ij : numpy.ndarray
        Absolute difference in inferred exposure dates for common-source scenarios,
        shape ``(N,)`` (days).
    incubation_periods : numpy.ndarray
        Incubation periods for two cases, shape ``(N, 2)``.
    generation_time_xi : numpy.ndarray
        Time from common source X to case i, shape ``(N,)`` (days).

    Returns
    -------
    numpy.ndarray
        Genetic evidence matrix, shape (K, M+1).
        Entry [k, m] = probability that k-th SNP distance is consistent
        with m intermediate hosts (m=0 for direct transmission).

    Notes
    -----
    **Direct transmission (m=0):**

    Expected TMRCA equals :math:`T_\mathrm{MRCA}^{exp} = \psi_i + \psi_j`,
    which is compared with observed TMRCA inferred from SNPs:

    .. math::

        T_\mathrm{MRCA}^{obs} = \frac{d}{2\lambda}

    where d is SNP distance and λ is the substitution rate (mutations/day).

    **Intermediate scenarios (m>0):**

    Two transmission paths are evaluated for each scenario:

    1. **Successive transmission**: Direct chain through intermediates
    2. **Common ancestor**: Both cases infected from a common source

    A scenario is accepted if either path's expected TMRCA matches the observed value
    (within tolerance).
    """
    num_distances = genetic_distance_ij.shape[0]
    mc_simulations = generation_time_xi.shape[0]

    out = np.zeros((num_distances, intermediates + 1), dtype=np.float64)

    # Invariants across m (computed once for efficiency)
    direct_tmrca_expected = sampling_delay_i + sampling_delay_j  # (N,)
    incubation_period_sum = incubation_periods[:, 0] + incubation_periods[:, 1]  # (N,)

    for k in prange(num_distances):
        observed_snp_distance = int(genetic_distance_ij[k])
        if observed_snp_distance < 0:
            # Negative genetic distances are not meaningful; leave zeros.
            continue

        # M = 0: direct transmission case
        count = 0
        for j in range(mc_simulations):
            tmrca_observed = observed_snp_distance / (2.0 * clock_rates[j])
            tmrca_expected = direct_tmrca_expected[j]
            if abs(tmrca_observed - tmrca_expected) <= intermediate_generations[j, 0]:
                count += 1.0
        out[k, 0] = count / mc_simulations

        # M > 0: scenarios with intermediate hosts
        for m in range(1, intermediates + 1):
            idx = intermediates - (m - 1)  # Index for summing generation intervals
            count_m = 0

            for j in range(mc_simulations):
                tmrca_observed = observed_snp_distance / (2.0 * clock_rates[j])

                # Path 1: Successive transmission
                sum_gi_successive = 0
                for col in range(idx, intermediates + 1):
                    sum_gi_successive += intermediate_generations[j, col]
                tmrca_successive = direct_tmrca_expected[j] + sum_gi_successive

                # Path 2: Common ancestor
                sum_gi_common = 0
                for col in range(idx, intermediates):
                    sum_gi_common += intermediate_generations[j, col]
                tmrca_common = diff_infection_ij[j] + incubation_period_sum[j] + sum_gi_common

                # Accept if either path's expected TMRCA matches observed (within tolerance)
                match_successive = abs(tmrca_observed - tmrca_successive) <= intermediate_generations[j, 0]
                match_common = abs(tmrca_observed - tmrca_common) <= generation_time_xi[j]
                if match_successive or match_common:  # Inclusion-exclusion on events
                    count_m += 1

            out[k, m] = count_m / float(mc_simulations)

    return out

@dataclass(frozen=True)
class Simulation:
    r"""
    Container for epidemiological random variables from Monte Carlo simulation.

    Holds all arrays produced by :func:`_run_simulations`, which are consumed
    by the temporal and genetic kernels. Each attribute has shape (N,) where N is
    the number of Monte Carlo scenarios.

    Attributes
    ----------
    incubation_periods : NDArrayFloat
        Incubation periods for two cases, shape (N, 2).
        Column 0 = case i, column 1 = case j (days).
    generation_interval : NDArrayFloat
        Generation intervals for transmission segments, shape (N, M+1).
        Column 0 = direct generation interval (infection to transmission).
        Columns 1..M = generation intervals for intermediate transmission steps.
    sampling_delay_i : NDArrayFloat
        Time from transmission to sampling for case i, shape ``(N,)`` (days).
    sampling_delay_j : NDArrayFloat
        Time from transmission to sampling for case j, shape ``(N,)`` (days).
    diff_incubation_ij : NDArrayFloat
        Difference in incubation periods (case i minus case j), shape ``(N,)``.
    generation_time_xi : NDArrayFloat
        Time from common source to case i, shape ``(N,)`` (days).
    diff_infection_ij : NDArrayFloat
        Absolute difference in exposure dates (common-source scenarios),
        shape ``(N,)`` (days).
    clock_rates : NDArrayFloat
        Substitution rates (mutations/day), shape (N,).
    """

    incubation_periods: NDArrayFloat
    generation_interval: NDArrayFloat
    sampling_delay_i: NDArrayFloat
    sampling_delay_j: NDArrayFloat
    diff_incubation_ij: NDArrayFloat
    generation_time_xi: NDArrayFloat
    diff_infection_ij: NDArrayFloat
    clock_rates: NDArrayFloat

def _run_simulations(
        toit: TOIT,
        clock: MolecularClock,
        num_simulations: int,
        no_intermediates: int
) -> Simulation:
    r"""
    Generate epidemiological random variables for Monte Carlo simulation.

    Draws samples from the infectiousness profile (TOIT) and molecular clock models
    to create a complete set of epidemiological scenarios. These are used by both
    temporal and genetic kernels to evaluate the consistency of observed distances
    with various transmission topologies.

    Parameters
    ----------
    toit : TOIT
        Configured infectiousness profile providing sampling primitives.
    clock : MolecularClock
        Molecular clock model for sampling substitution rates.
    num_simulations : int
        Number of Monte Carlo draws (N).
    no_intermediates : int
        Maximum number of intermediate hosts (M).

    Returns
    -------
    Simulation
        Typed container with all epidemiological quantities needed by the kernels.

    See Also
    --------
    Simulation : Container for the returned arrays.
    """
    N = int(num_simulations)
    M = int(no_intermediates)

    inc_period = toit.sample_incubation(size=(N, 2))  # (N, 2)
    gen_interval = toit.generation_time(size=(N, M + 1))  # (N, M+1)
    toit_values = toit.rvs(size=(N, 2))  # (N, 2)
    latent_period = toit.sample_latent(size=N)  # (N,)
    clock_rates = clock.sample_clock_rate_per_day(size=N)  # (N,)

    sim = Simulation(
        incubation_periods=inc_period,
        generation_interval=gen_interval,
        sampling_delay_i=np.abs(gen_interval[:, 0] - inc_period[:, 0]),
        sampling_delay_j=inc_period[:, 1],
        diff_incubation_ij=inc_period[:, 0] - inc_period[:, 1],
        generation_time_xi=latent_period + np.minimum(toit_values[:, 0], toit_values[:, 1]),
        diff_infection_ij=np.abs(toit_values[:, 1] - toit_values[:, 0]),
        clock_rates=clock_rates,
    )
    return sim


# =============================================================================
# Public API
# =============================================================================


def linkage_probability(
        toit: TOIT,
        clock: MolecularClock,
        genetic_distance: ArrayLike,
        temporal_distance: ArrayLike,
        *,
        intermediate_generations: tuple[int, ...] = (0,),
        no_intermediates: int = 10,
        num_simulations: int = 10000,
        cache_unique_distances: bool = True,
) -> float | np.ndarray:
    r"""
    Estimate the probability of transmission linkage given genetic and temporal distances.

    Combines temporal and genetic evidence using Monte Carlo simulation to compute
    :math:`P(\text{link} | g, t)` where g is SNP distance and t is days between
    sampling dates.

    Parameters
    ----------
    toit : TOIT
        Configured infectiousness profile model.
    clock : MolecularClock
        Molecular clock model for substitution rates.
    genetic_distance : array_like
        Observed SNP distance(s), scalar or array. Should be non-negative.
    temporal_distance : array_like
        Observed temporal distance(s) in days. Must have same length as genetic_distance.
    intermediate_generations : tuple[int, ...], default=(0,)
        Which intermediate scenario counts to include. Use (0,) for direct transmission,
        (0, 1) to include 1 intermediate host, etc.
    no_intermediates : int, default=10
        Maximum number of intermediate hosts (M) considered in simulation.
    num_simulations : int, default=10000
        Number of Monte Carlo draws.
    cache_unique_distances : bool, default=True
        If True, deduplicates input distance pairs before computation and caches results,
        reusing them for identical distance combinations. Significantly speeds up
        computation when many duplicate pairs exist. Set to False to compute each
        input pair independently (required for reproducibility with specific orderings).

    Returns
    -------
    float or np.ndarray
        Estimated probability P(link | g, t). Returns scalar if both inputs are scalar,
        otherwise returns array with same shape as input arrays.

    Raises
    ------
    ValueError
        If genetic_distance and temporal_distance have different lengths,
        or if intermediate_generations values exceed no_intermediates.

    Notes
    -----
    The computation evaluates consistency of observed distances across direct
    transmission (m=0) and intermediate transmission scenarios (m>0) using
    the temporal and genetic kernels.

    .. math::

        P(\text{link} | g, t) = P_t(t) \times \sum_{m \in \text{intermediate_generations}} P(m | g)

    where :math:`P_t(t)` is temporal evidence and :math:`P(m | g)` is normalized
    genetic evidence for scenario m.

    Performance
    -----------
    When ``cache_unique_distances=True`` (default), computation is optimized by:
    1. Deduplicating input distance pairs
    2. Running simulations and kernels only for unique distances
    3. Mapping results back to original input order

    This is most beneficial when the input contains many duplicate distance pairs.
    For example, computing over 1000 pairs with only 50 unique combinations will
    be ~20x faster with caching enabled.

    See Also
    --------
    pairwise_linkage_probability_matrix : Evaluate over distance grids.
    temporal_linkage_probability : Temporal evidence only.
    genetic_linkage_probability : Genetic evidence only.
    """
    # 1) Prepare inputs as 1D arrays of same length
    genetic_distance_arr = np.atleast_1d(np.asarray(genetic_distance, dtype=float))
    temporal_distance_arr = np.atleast_1d(np.asarray(temporal_distance, dtype=float))
    if genetic_distance_arr.size != temporal_distance_arr.size:
        raise ValueError(
            "genetic_distance and temporal_distance must have the same length, "
            f"got {genetic_distance_arr.size} vs {temporal_distance_arr.size}.",
        )

    # Early return for empty input
    if genetic_distance_arr.size == 0:
        return np.nan

    is_scalar_input = np.isscalar(genetic_distance) and np.isscalar(temporal_distance)
    num_intermediate_hosts = int(no_intermediates)

    # 2) Optionally deduplicate unique distance pairs
    if cache_unique_distances and genetic_distance_arr.size > 1:
        # Create combined indices for (genetic, temporal) pairs
        g_unique, gi = np.unique(genetic_distance_arr, return_inverse=True)
        t_unique, tj = np.unique(temporal_distance_arr, return_inverse=True)

        # Compute probability matrix for all unique combinations
        prob_matrix = pairwise_linkage_probability_matrix(
            toit=toit,
            clock=clock,
            genetic_distances=g_unique,
            temporal_distances=t_unique,
            intermediate_generations=intermediate_generations,
            no_intermediates=no_intermediates,
            num_simulations=num_simulations,
        )

        # Map back to original input order
        out = np.asarray(prob_matrix, dtype=float)[gi, tj]
    else:
        # Compute directly without deduplication
        sim = _run_simulations(toit, clock, int(num_simulations), num_intermediate_hosts)

        # 3) Temporal evidence
        p_temporal = _temporal_kernel(
            temporal_distance=temporal_distance_arr,
            diff_incubation_ij=sim.diff_incubation_ij,
            generation_interval=sim.generation_interval[:, 0],
        )  # shape (K,)

        # 4) Genetic evidence: p_m for each m=0..M
        p_genetic_by_scenario = _genetic_kernel(
            genetic_distance_ij=genetic_distance_arr,
            clock_rates=sim.clock_rates,
            sampling_delay_i=sim.sampling_delay_i,
            sampling_delay_j=sim.sampling_delay_j,
            intermediate_generations=sim.generation_interval,
            intermediates=num_intermediate_hosts,
            diff_infection_ij=sim.diff_infection_ij,
            incubation_periods=sim.incubation_periods,
            generation_time_xi=sim.generation_time_xi,
        )  # shape (K, M+1)

        # 5) Normalize genetic evidence across m
        total = 1.0 - np.prod(1.0 - p_genetic_by_scenario, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_relative = np.where(total[:, None] > 0.0, p_genetic_by_scenario / total[:, None], 0.0)
        row_sums = p_relative.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_normalized = np.where(row_sums[:, None] > 0.0, p_relative / row_sums[:, None], 0.0)

        # Select columns m specified by intermediate_generations
        cols = np.array(intermediate_generations, dtype=np.int64)
        if cols.min() < 0 or cols.max() > num_intermediate_hosts:
            raise ValueError(
                f"intermediate_generations must be within [0, {num_intermediate_hosts}], got {intermediate_generations}.",
            )
        selected = p_normalized[:, cols].sum(axis=1)  # shape (K,)

        # 6) Combine temporal and genetic evidence
        out = p_temporal * selected

    # 7) Return scalar if scalar input
    if is_scalar_input:
        return float(out[0])
    return out


def pairwise_linkage_probability_matrix(
        toit: TOIT,
        clock: MolecularClock,
        genetic_distances: np.ndarray,
        temporal_distances: np.ndarray,
        *,
        intermediate_generations: tuple[int, ...] = (0,),
        no_intermediates: int = 10,
        num_simulations: int = 10000,
) -> np.ndarray:
    r"""
    Compute a (G x T) matrix of :math:`P(\text{link} | g, t)` over distance grids.

    Evaluates linkage probability for all combinations of genetic and temporal
    distances, returning a 2D matrix where entry (i, j) corresponds to the probability
    for genetic distance g_i and temporal distance t_j.

    Parameters
    ----------
    toit : TOIT
        Configured infectiousness profile model.
    clock : MolecularClock
        Molecular clock model for substitution rates.
    genetic_distances : np.ndarray
        1D array of genetic distances (SNP counts).
    temporal_distances : np.ndarray
        1D array of temporal distances (days).
    intermediate_generations : tuple[int, ...], default=(0,)
        Which intermediate scenario counts to include.
    no_intermediates : int, default=10
        Maximum number of intermediate hosts.
    num_simulations : int, default=10000
        Number of Monte Carlo draws.

    Returns
    -------
    np.ndarray
        Matrix of shape (len(genetic_distances), len(temporal_distances)).
        Entry [i, j] = P(link | genetic_distances[i], temporal_distances[j]).

    See Also
    --------
    estimate_linkage_probability : Compute for single or paired distances.
    """
    genetic_distances_arr = np.asarray(genetic_distances, dtype=float)
    temporal_distances_arr = np.asarray(temporal_distances, dtype=float)

    g_grid, t_grid = np.meshgrid(genetic_distances_arr, temporal_distances_arr, indexing="ij")
    flat_g = g_grid.ravel()
    flat_t = t_grid.ravel()

    flat_p = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=flat_g,
        temporal_distance=flat_t,
        intermediate_generations=intermediate_generations,
        no_intermediates=no_intermediates,
        num_simulations=num_simulations,
    )
    flat_p = np.asarray(flat_p, dtype=float)
    return flat_p.reshape(g_grid.shape)



def temporal_linkage_probability(
        temporal_distance: ArrayLike,
        toit: TOIT,
        num_simulations: int = 10000,
) -> np.ndarray:
    r"""
    Monte Carlo estimate of temporal evidence :math:`P_t` for one or more distances.

    For each temporal distance :math:`t_i` (days between case sampling dates), computes
    the probability that the adjusted interval falls within a directly simulated
    generation interval:

    .. math::

        P_t(i) = \mathbb{P}\left(|t_i + (\tau_{\text{inc},i} - \tau_{\text{inc},j})| \leq GI\right)

    where :math:`\tau_{\text{inc}}` are incubation periods and GI is the generation
    interval, sampled from the TOIT model.

    Parameters
    ----------
    temporal_distance : array_like
        One or more temporal distances (days). Can be scalar or 1D array-like.
    toit : TOIT
        Configured infectiousness profile providing samples for incubation
        periods and generation times.
    num_simulations : int, default=10000
        Number of Monte Carlo draws.

    Returns
    -------
    np.ndarray
        Array of shape (K,), where K = len(np.atleast_1d(temporal_distance)),
        containing the temporal evidence :math:`P_t` for each input distance.

    Notes
    -----
    This function provides only the temporal component of transmission linkage
    evidence. To obtain :math:`P(\text{link} | g, t)`, combine with genetic
    evidence using :func:`estimate_linkage_probability`.

    Reproducibility is controlled by the rng_seed in the provided TOIT instance.
    """
    td = np.atleast_1d(np.asarray(temporal_distance, dtype=float))  # ensure 1D array

    inc_period = toit.sample_incubation(size=(num_simulations, 2))  # (N, 2)
    diff_inc = inc_period[:, 0] - inc_period[:, 1]  # (N,)
    gen_interval = toit.generation_time(size=num_simulations)  # (N,)

    return _temporal_kernel(
        temporal_distance_ij=td,
        diff_incubation_ij=diff_inc,
        generation_interval=gen_interval,
    )


def genetic_linkage_probability(
        toit: TOIT,
        clock: MolecularClock,
        genetic_distance: ArrayLike,
        *,
        num_simulations: int = 10000,
        no_intermediates: int = 10,
        intermediate_generations: tuple[int, ...] | None = None,
        kind: str = "relative",  # "raw" | "relative" | "normalized"
) -> np.ndarray:
    r"""
    Monte Carlo estimate of genetic evidence for transmission linkage.

    Evaluates the consistency of observed SNP distances with expected genetic
    accumulation under different transmission topologies (direct vs. with
    intermediate hosts).

    Parameters
    ----------
    toit : TOIT
        Configured infectiousness profile model providing epidemiological
        parameters for the Monte Carlo simulation.
    clock : MolecularClock
        Molecular clock model for sampling substitution rates.
    genetic_distance : array_like
        Observed SNP distance(s), scalar or array-like.
    num_simulations : int, default=10000
        Number of Monte Carlo draws.
    no_intermediates : int, default=10
        Maximum number of intermediate hosts (M) considered.
    intermediate_generations : tuple[int, ...], optional
        Which intermediate scenario counts to aggregate. If None, returns the
        full matrix of evidence for all scenarios m=0..M. If specified, selects
        and aggregates evidence for specified scenarios.
    kind : {'raw', 'relative', 'normalized'}, default='relative'
        Type of genetic evidence to return:

        - 'raw': :math:`P(m | d)` for each m (raw scenario probabilities)
        - 'relative': :math:`P(m | d) / \sum_m P(m | d)` (relative probabilities)
        - 'normalized': row-normalized version of relative

    Returns
    -------
    np.ndarray
        If ``intermediate_generations`` is None:
            Matrix of shape (K, M+1) with evidence for each SNP distance
            across all intermediate scenarios.
        If ``intermediate_generations`` is specified:
            Array of shape (K,) with aggregated evidence for selected scenarios.

    Notes
    -----
    The genetic evidence for scenario m is computed as the fraction of Monte
    Carlo simulations where the observed SNP distance is consistent with
    expected TMRCA under that transmission topology (direct or with
    m intermediate hosts).
    """
    genetic_distance_arr = np.atleast_1d(np.asarray(genetic_distance, dtype=float))
    num_intermediate_hosts = int(no_intermediates)

    sim = _run_simulations(toit, clock, int(num_simulations), num_intermediate_hosts)

    p_genetic_by_scenario = _genetic_kernel(
        genetic_distance_ij=genetic_distance_arr,
        clock_rates=sim.clock_rates,
        sampling_delay_i=sim.sampling_delay_i,
        sampling_delay_j=sim.sampling_delay_j,
        intermediate_generations=sim.generation_interval,
        intermediates=num_intermediate_hosts,
        diff_infection_ij=sim.diff_infection_ij,
        incubation_periods=sim.incubation_periods,
        generation_time_xi=sim.generation_time_xi,
    )

    total = 1.0 - np.prod(1.0 - p_genetic_by_scenario, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_relative = np.where(total[:, None] > 0.0, p_genetic_by_scenario / total[:, None], 0.0)
    row_sums = p_relative.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_normalized = np.where(row_sums[:, None] > 0.0, p_relative / row_sums[:, None], 0.0)

    if intermediate_generations is not None:
        cols = np.array(intermediate_generations, dtype=np.int64)
        if cols.min() < 0 or cols.max() > num_intermediate_hosts:
            raise ValueError(
                f"intermediate_generations must be within [0, {num_intermediate_hosts}], got {intermediate_generations}.",
            )

        if kind == "relative":
            selected = p_relative[:, cols].mean(axis=1)
        elif kind == "raw":
            selected = p_genetic_by_scenario[:, cols].mean(axis=1)
        elif kind == "normalized":
            selected = p_normalized[:, cols].sum(axis=1)
        else:
            raise ValueError(
                "kind must be 'relative', 'raw', or 'normalized', "
                f"got {kind!r}.",
            )
        return selected

    if kind == "relative":
        return p_relative
    if kind == "raw":
        return p_genetic_by_scenario
    if kind == "normalized":
        return p_normalized
    raise ValueError(f"kind must be 'relative', 'raw', or 'normalized', got {kind!r}.")
