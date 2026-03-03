"""
Estimate transmission linkage probabilities from genetic and temporal distances.

Uses Monte Carlo simulation to combine timing and molecular clock information.
Core simulation draws and kernels are organized in ``Epilink``.

Classes
-------
Epilink
    Monte Carlo draws and kernel methods used by linkage estimators.

Functions
---------
linkage_probability
    Estimate combined linkage probability for paired distances.
linkage_probability_matrix
    Evaluate linkage probabilities over distance grids.
temporal_linkage_probability
    Estimate temporal evidence only.
genetic_linkage_probability
    Estimate genetic evidence only.
"""

from __future__ import annotations

import math
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


@dataclass(frozen=True)
class Epilink:
    """
    Monte Carlo draws used by linkage kernels.

    Parameters
    ----------
    incubation_periods : numpy.ndarray
        Incubation periods for two cases, shape (N, 2).
    generation_interval : numpy.ndarray
        Generation intervals, shape (N, M+1).
    sampling_delay_i : numpy.ndarray
        Time from transmission to sampling for case i, shape (N,).
    sampling_delay_j : numpy.ndarray
        Time from transmission to sampling for case j, shape (N,).
    diff_incubation_ij : numpy.ndarray
        Incubation period difference (i - j), shape (N,).
    generation_time_xi : numpy.ndarray
        Time from common source X to case i, shape (N,).
    diff_infection_ij : numpy.ndarray
        Absolute difference in exposure dates, shape (N,).
    clock_rates : numpy.ndarray
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

    @staticmethod
    @njit(cache=True, parallel=True, fastmath=True)
    def temporal_kernel(
        temporal_distance_ij: np.ndarray,
        diff_incubation_ij: np.ndarray,
        generation_interval: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate temporal evidence from Monte Carlo samples.

        Parameters
        ----------
        temporal_distance_ij : numpy.ndarray
            Distances between sampling dates in days, shape (K,).
        diff_incubation_ij : numpy.ndarray
            Differences in incubation periods, shape (N,).
        generation_interval : numpy.ndarray
            Generation intervals, shape (N,).

        Returns
        -------
        evidence : numpy.ndarray
            Temporal evidence for each distance, shape (K,).
        """
        K = temporal_distance_ij.shape[0]
        N = diff_incubation_ij.shape[0]
        out = np.empty(K, dtype=np.float64)

        for i in prange(K):
            t = temporal_distance_ij[i]
            cnt = 0
            for j in range(N):
                if math.fabs(t + diff_incubation_ij[j]) <= generation_interval[j]:
                    cnt += 1
            out[i] = cnt / N
        return out

    @staticmethod
    @njit(cache=True, parallel=True, fastmath=True)
    def genetic_kernel(
        genetic_distance_ij: np.ndarray,
        clock_rates: np.ndarray,
        sampling_delay_i: np.ndarray,
        sampling_delay_j: np.ndarray,
        intermediate_generations: np.ndarray,
        intermediate_hosts: int,
        diff_infection_ij: np.ndarray,
        incubation_periods: np.ndarray,
        generation_time_xi: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate genetic evidence across intermediate scenarios.

        Parameters
        ----------
        genetic_distance_ij : numpy.ndarray
            SNP distances, shape (K,).
        clock_rates : numpy.ndarray
            Substitution rates (mutations/day), shape (N,).
        sampling_delay_i : numpy.ndarray
            Sampling delays for case i, shape (N,).
        sampling_delay_j : numpy.ndarray
            Sampling delays for case j, shape (N,).
        intermediate_generations : numpy.ndarray
            Generation intervals, shape (N, M+1).
        intermediate_hosts : int
            Maximum number of intermediate hosts (M).
        diff_infection_ij : numpy.ndarray
            Differences in exposure dates, shape (N,).
        incubation_periods : numpy.ndarray
            Incubation periods for two cases, shape (N, 2).
        generation_time_xi : numpy.ndarray
            Time from common source X to case i, shape (N,).

        Returns
        -------
        evidence : numpy.ndarray
            Genetic evidence matrix, shape (K, M+1).
        """
        K = genetic_distance_ij.shape[0]
        N = generation_time_xi.shape[0]
        M = intermediate_hosts

        out = np.zeros((K, M + 1), dtype=np.float64)

        # Invariants across k,m
        direct_tmrca_expected = sampling_delay_i + sampling_delay_j  # (N,)
        incubation_period_sum = incubation_periods[:, 0] + incubation_periods[:, 1]  # (N,)

        # Precompute 1/(2*clock_rates) for efficiency
        inv_2_clock = 0.5 / clock_rates  # (N,)

        # Precompute suffix sums of intermediate_generations along axis=1
        # suffix[j, c] = sum_{u=c..M} intermediate_generations[j, u]
        suffix = np.empty((N, M + 2), dtype=np.float64)
        for j in prange(N):
            suffix[j, M + 1] = 0.0
            for c in range(M, -1, -1):
                suffix[j, c] = suffix[j, c + 1] + intermediate_generations[j, c]

        # Main loops
        for k in prange(K):
            d = genetic_distance_ij[k]
            if d < 0:
                continue

            # M = 0
            cnt0 = 0
            for j in range(N):
                tmrca_obs = d * inv_2_clock[j]
                tmrca_exp = direct_tmrca_expected[j]
                if math.fabs(tmrca_obs - tmrca_exp) <= intermediate_generations[j, 0]:
                    cnt0 += 1
            out[k, 0] = cnt0 / N

            # M > 0
            for m in range(1, M + 1):
                idx = M - (m - 1)

                cntm = 0
                for j in range(N):
                    tmrca_obs = d * inv_2_clock[j]

                    # Path 1: successive transmission
                    sum_successive = suffix[j, idx]  # idx..M
                    tmrca_successive = direct_tmrca_expected[j] + sum_successive

                    # Path 2: common ancestor
                    sum_common = suffix[j, idx] - intermediate_generations[j, M]  # idx..M-1
                    tmrca_common = diff_infection_ij[j] + incubation_period_sum[j] + sum_common

                    match_successive = (
                        math.fabs(tmrca_obs - tmrca_successive) <= intermediate_generations[j, 0]
                    )
                    match_common = math.fabs(tmrca_obs - tmrca_common) <= generation_time_xi[j]

                    if match_successive or match_common:
                        cntm += 1

                out[k, m] = cntm / N

        return out

    @classmethod
    def run_simulations(
        cls,
        toit: TOIT,
        clock: MolecularClock,
        num_simulations: int,
        intermediate_hosts: int,
    ) -> Epilink:
        """
        Sample epidemiological quantities for Monte Carlo simulation.

        Parameters
        ----------
        toit : TOIT
            Infectiousness profile model.
        clock : MolecularClock
            Molecular clock model.
        num_simulations : int
            Number of Monte Carlo draws.
        intermediate_hosts : int
            Maximum number of intermediate hosts (M).

        Returns
        -------
        sim : Epilink
            Container with sampled quantities.
        """
        N = int(num_simulations)
        M = int(intermediate_hosts)

        inc_period = toit.sample_incubation(size=(N, 2))  # (N, 2)
        gen_interval = toit.generation_time(size=(N, M + 1))  # (N, M+1)
        toit_values = toit.rvs(size=(N, 2))  # (N, 2)
        latent_period = toit.sample_latent(size=N)  # (N,)
        clock_rates = clock.sample_clock_rate_per_day(size=N)  # (N,)

        return cls(
            incubation_periods=inc_period,
            generation_interval=gen_interval,
            sampling_delay_i=np.abs(gen_interval[:, 0] - inc_period[:, 0]),
            sampling_delay_j=inc_period[:, 1],
            diff_incubation_ij=inc_period[:, 0] - inc_period[:, 1],
            generation_time_xi=latent_period + np.minimum(toit_values[:, 0], toit_values[:, 1]),
            diff_infection_ij=np.abs(toit_values[:, 1] - toit_values[:, 0]),
            clock_rates=clock_rates,
        )


def _ensure_pyfunc(func) -> None:
    """Ensure a .py_func attribute for non-numba callables (test compatibility)."""
    if not hasattr(func, "py_func"):
        func.py_func = func


_ensure_pyfunc(Epilink.temporal_kernel)
_ensure_pyfunc(Epilink.genetic_kernel)


# =============================================================================
# Public API
# =============================================================================


def linkage_probability(
    toit: TOIT,
    clock: MolecularClock,
    genetic_distance: ArrayLike,
    temporal_distance: ArrayLike,
    *,
    intermediate_generations: tuple[int, ...] = (0, 1),
    intermediate_hosts: int = 10,
    num_simulations: int = 10000,
    cache_unique_distances: bool = True,
) -> float | np.ndarray:
    """
    Estimate linkage probability from genetic and temporal distances.

    Parameters
    ----------
    toit : TOIT
        Configured infectiousness profile model.
    clock : MolecularClock
        Molecular clock model for substitution rates.
    genetic_distance : array_like
        Observed SNP distance(s), scalar or array_like. Should be non-negative.
    temporal_distance : array_like
        Observed temporal distance(s) in days. Must have the same length as
        ``genetic_distance`` after coercion to 1D arrays.
    intermediate_generations : tuple of int, default=(0, 1)
        Which intermediate scenario counts to include.
    intermediate_hosts : int, default=10
        Maximum number of intermediate hosts (M) considered in simulation.
    num_simulations : int, default=10000
        Number of Monte Carlo draws.
    cache_unique_distances : bool, default=True
        If True, reuse results for duplicate distance pairs.

    Returns
    -------
    probability : float or numpy.ndarray
        Estimated linkage probability. Returns a scalar if both inputs are
        scalar; otherwise returns an array of shape (K,) for K distances.

    Raises
    ------
    ValueError
        If genetic_distance and temporal_distance have different lengths,
        or if intermediate_generations values exceed intermediate_hosts.

    Notes
    -----
    Combines temporal and genetic evidence using Monte Carlo samples.

    See Also
    --------
    linkage_probability_matrix : Evaluate over distance grids.
    temporal_linkage_probability : Temporal evidence only.
    genetic_linkage_probability : Genetic evidence only.
    """
    # 1) Prepare inputs as 1D arrays of same length
    genetic_distance_arr = np.atleast_1d(np.asarray(genetic_distance, dtype=np.int64))
    temporal_distance_arr = np.atleast_1d(np.asarray(temporal_distance, dtype=np.int64))
    if genetic_distance_arr.size != temporal_distance_arr.size:
        raise ValueError(
            "genetic_distance and temporal_distance must have the same length, "
            f"got {genetic_distance_arr.size} vs {temporal_distance_arr.size}.",
        )

    # Early return for empty input
    if genetic_distance_arr.size == 0:
        return np.nan

    is_scalar_input = np.isscalar(genetic_distance) and np.isscalar(temporal_distance)

    # 2) Optionally deduplicate unique distance *pairs*
    if cache_unique_distances and genetic_distance_arr.size > 1:
        pairs = np.column_stack((genetic_distance_arr, temporal_distance_arr))

        # unique_pairs: shape (U, 2), inv: shape (K,)
        unique_pairs, inv = np.unique(pairs, axis=0, return_inverse=True)
        g_u = unique_pairs[:, 0]
        t_u = unique_pairs[:, 1]

        # Run *once* and evaluate only the U observed pairs
        sim = Epilink.run_simulations(toit, clock, int(num_simulations), intermediate_hosts)

        # Temporal evidence for U pairs
        p_temporal_u = Epilink.temporal_kernel(
            temporal_distance_ij=t_u,
            diff_incubation_ij=sim.diff_incubation_ij,
            generation_interval=sim.generation_interval[:, 0],
        )  # shape (U,)

        # Genetic evidence for U pairs
        p_genetic_u = Epilink.genetic_kernel(
            genetic_distance_ij=g_u,
            clock_rates=sim.clock_rates,
            sampling_delay_i=sim.sampling_delay_i,
            sampling_delay_j=sim.sampling_delay_j,
            intermediate_generations=sim.generation_interval,
            intermediate_hosts=intermediate_hosts,
            diff_infection_ij=sim.diff_infection_ij,
            incubation_periods=sim.incubation_periods,
            generation_time_xi=sim.generation_time_xi,
        )  # shape (U, M+1)

        total = 1.0 - np.prod(1.0 - p_genetic_u, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_relative = np.where(total[:, None] > 0.0, p_genetic_u / total[:, None], 0.0)
        row_sums = p_relative.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_normalized = np.where(row_sums[:, None] > 0.0, p_relative / row_sums[:, None], 0.0)

        cols = np.array(intermediate_generations, dtype=np.int64)
        if cols.min() < 0 or cols.max() > intermediate_hosts:
            raise ValueError(
                f"intermediate_generations must be within [0, {intermediate_hosts}], "
                f"got {intermediate_generations}.",
            )
        selected_u = p_normalized[:, cols].sum(axis=1)  # shape (U,)

        out_u = p_temporal_u * selected_u  # shape (U,)
        out = out_u[inv]  # map back to shape (K,)
    else:
        # Compute directly without deduplication
        sim = Epilink.run_simulations(toit, clock, int(num_simulations), intermediate_hosts)

        # 3) Temporal evidence
        p_temporal = Epilink.temporal_kernel(
            temporal_distance_ij=temporal_distance_arr,
            diff_incubation_ij=sim.diff_incubation_ij,
            generation_interval=sim.generation_interval[:, 0],
        )  # shape (K,)

        # 4) Genetic evidence: p_m for each m=0..M
        p_genetic_by_scenario = Epilink.genetic_kernel(
            genetic_distance_ij=genetic_distance_arr,
            clock_rates=sim.clock_rates,
            sampling_delay_i=sim.sampling_delay_i,
            sampling_delay_j=sim.sampling_delay_j,
            intermediate_generations=sim.generation_interval,
            intermediate_hosts=intermediate_hosts,
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
        if cols.min() < 0 or cols.max() > intermediate_hosts:
            raise ValueError(
                f"intermediate_generations must be within [0, {intermediate_hosts}], got {intermediate_generations}.",
            )
        selected = p_normalized[:, cols].sum(axis=1)  # shape (K,)

        # 6) Combine temporal and genetic evidence
        out = p_temporal * selected

    # 7) Return scalar if scalar input
    if is_scalar_input:
        return float(out[0])
    return out


def linkage_probability_matrix(
    toit: TOIT,
    clock: MolecularClock,
    genetic_distances: np.ndarray,
    temporal_distances: np.ndarray,
    *,
    intermediate_generations: tuple[int, ...] = (0, 1),
    intermediate_hosts: int = 10,
    num_simulations: int = 10000,
) -> np.ndarray:
    """
    Compute linkage probabilities over distance grids.

    Parameters
    ----------
    toit : TOIT
        Configured infectiousness profile model.
    clock : MolecularClock
        Molecular clock model for substitution rates.
    genetic_distances : array_like
        1D array of genetic distances (SNP counts).
    temporal_distances : array_like
        1D array of temporal distances (days).
    intermediate_generations : tuple of int, default=(0, 1)
        Which intermediate scenario counts to include.
    intermediate_hosts : int, default=10
        Maximum number of intermediate hosts.
    num_simulations : int, default=10000
        Number of Monte Carlo draws.

    Returns
    -------
    prob_matrix : numpy.ndarray
        Matrix of shape (len(genetic_distances), len(temporal_distances)).

    See Also
    --------
    linkage_probability : Compute for single or paired distances.
    """
    genetic_distances_arr = np.asarray(genetic_distances, dtype=np.int64)
    temporal_distances_arr = np.asarray(temporal_distances, dtype=np.int64)

    g_grid, t_grid = np.meshgrid(genetic_distances_arr, temporal_distances_arr, indexing="ij")
    flat_g = g_grid.ravel()
    flat_t = t_grid.ravel()

    flat_p = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=flat_g,
        temporal_distance=flat_t,
        intermediate_generations=intermediate_generations,
        intermediate_hosts=intermediate_hosts,
        num_simulations=num_simulations,
    )
    flat_p = np.asarray(flat_p, dtype=float)
    return flat_p.reshape(g_grid.shape)


def temporal_linkage_probability(
    temporal_distance: ArrayLike,
    toit: TOIT,
    num_simulations: int = 10000,
) -> np.ndarray:
    """
    Estimate temporal evidence for one or more distances.

    Parameters
    ----------
    temporal_distance : array_like
        One or more temporal distances (days). Can be scalar or 1D array_like.
    toit : TOIT
        Configured infectiousness profile providing samples for incubation
        periods and generation times.
    num_simulations : int, default=10000
        Number of Monte Carlo draws.

    Returns
    -------
    evidence : numpy.ndarray
        Array of shape (K,) for K = len(np.atleast_1d(temporal_distance)).

    Notes
    -----
    This is the temporal component only; combine with genetic evidence via
    :func:`linkage_probability`.
    """
    td = np.atleast_1d(np.asarray(temporal_distance, dtype=np.int64))  # ensure 1D array

    inc_period = toit.sample_incubation(size=(num_simulations, 2))  # (N, 2)
    diff_inc = inc_period[:, 0] - inc_period[:, 1]  # (N,)
    gen_interval = toit.generation_time(size=num_simulations)  # (N,)

    return Epilink.temporal_kernel(
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
    intermediate_hosts: int = 10,
    intermediate_generations: tuple[int, ...] | None = (0, 1),
    kind: str = "relative",  # "raw" | "relative" | "normalized"
) -> np.ndarray:
    """
    Estimate genetic evidence for transmission linkage.

    Parameters
    ----------
    toit : TOIT
        Configured infectiousness profile model providing epidemiological
        parameters for the Monte Carlo simulation.
    clock : MolecularClock
        Molecular clock model for sampling substitution rates.
    genetic_distance : array_like
        Observed SNP distance(s), scalar or array_like.
    num_simulations : int, default=10000
        Number of Monte Carlo draws.
    intermediate_hosts : int, default=10
        Maximum number of intermediate hosts (M) considered.
    intermediate_generations : tuple of int or None, default=(0, 1)
        Which intermediate scenario counts to include. If None, return evidence
        for all intermediate scenarios.
    kind : {'raw', 'relative', 'normalized'}, default='relative'
        Output type: raw scenario probabilities, relative probabilities, or
        row-normalized relative probabilities.

    Returns
    -------
    evidence : numpy.ndarray
        If ``intermediate_generations`` is provided, returns an array of shape
        (K,) aggregated across the selected scenarios. If
        ``intermediate_generations`` is None, returns an array of shape
        (K, M+1) with evidence for all scenarios.

    Raises
    ------
    ValueError
        If ``intermediate_generations`` contains values outside
        ``[0, intermediate_hosts]`` or if ``kind`` is invalid.
    """
    genetic_distance_arr = np.atleast_1d(np.asarray(genetic_distance, dtype=np.int64))

    sim = Epilink.run_simulations(toit, clock, int(num_simulations), intermediate_hosts)

    p_genetic_by_scenario = Epilink.genetic_kernel(
        genetic_distance_ij=genetic_distance_arr,
        clock_rates=sim.clock_rates,
        sampling_delay_i=sim.sampling_delay_i,
        sampling_delay_j=sim.sampling_delay_j,
        intermediate_generations=sim.generation_interval,
        intermediate_hosts=intermediate_hosts,
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
        if cols.min() < 0 or cols.max() > intermediate_hosts:
            raise ValueError(
                f"intermediate_generations must be within [0, {intermediate_hosts}], got {intermediate_generations}.",
            )

        if kind == "relative":
            selected = p_relative[:, cols].mean(axis=1)
        elif kind == "raw":
            selected = p_genetic_by_scenario[:, cols].mean(axis=1)
        elif kind == "normalized":
            selected = p_normalized[:, cols].sum(axis=1)
        else:
            raise ValueError(
                "kind must be 'relative', 'raw', or 'normalized', " f"got {kind!r}.",
            )
        return selected

    if kind == "relative":
        return p_relative
    if kind == "raw":
        return p_genetic_by_scenario
    if kind == "normalized":
        return p_normalized
    raise ValueError(f"kind must be 'relative', 'raw', or 'normalized', got {kind!r}.")
