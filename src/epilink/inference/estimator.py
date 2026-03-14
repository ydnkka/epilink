"""Public estimation functions for transmission linkage inference."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..model.clock import MolecularClock
from ..model.profiles import InfectiousnessToTransmissionTime
from .draws import LinkageMonteCarloSamples


def _normalize_genetic_scores(
    genetic_scores: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Normalize raw compatibility weights across intermediate-host scenarios."""

    totals = genetic_scores.sum(axis=1, keepdims=True)
    return np.divide(
        genetic_scores,
        totals,
        out=np.zeros_like(genetic_scores),
        where=totals > 0.0,
    )


def _coerce_intermediate_counts(
    included_intermediate_counts: int | tuple[int, ...] | list[int] | np.ndarray,
    max_intermediate_hosts: int,
) -> np.ndarray:
    """Coerce one or more selected intermediary counts to a validated 1D array."""

    cols = np.atleast_1d(np.asarray(included_intermediate_counts, dtype=np.int64))
    if cols.size == 0 or cols.min() < 0 or cols.max() > max_intermediate_hosts:
        raise ValueError(
            f"included_intermediate_counts must be within [0, {max_intermediate_hosts}], "
            f"got {included_intermediate_counts}.",
        )
    return cols


def estimate_linkage_probability(
    transmission_profile: InfectiousnessToTransmissionTime,
    clock: MolecularClock,
    genetic_distance: npt.ArrayLike,
    temporal_distance: npt.ArrayLike,
    *,
    included_intermediate_counts: tuple[int, ...] = (0,),
    max_intermediate_hosts: int = 10,
    num_simulations: int = 10000,
    cache_unique_distances: bool = True,
) -> float | np.ndarray:
    """Estimate linkage probability from genetic and temporal distances.

    Parameters
    ----------
    transmission_profile : InfectiousnessToTransmissionTime
        Transmission-time model used for temporal simulation.
    clock : MolecularClock
        Molecular clock used for genetic simulation.
    genetic_distance : array_like
        Genetic distances between case pairs.
    temporal_distance : array_like
        Temporal distances between case pairs.
    included_intermediate_counts : tuple of int, default=(0,)
        Intermediate-host counts to include in the final probability.
    max_intermediate_hosts : int, default=10
        Maximum number of intermediate hosts considered in the Monte Carlo
        simulation.
    num_simulations : int, default=10000
        Number of Monte Carlo draws.
    cache_unique_distances : bool, default=True
        If ``True``, compute repeated distance pairs only once.

    Returns
    -------
    float or numpy.ndarray
        Linkage probability for each supplied pair.
    """

    genetic_distance_arr = np.atleast_1d(np.asarray(genetic_distance, dtype=np.int64))
    temporal_distance_arr = np.atleast_1d(np.asarray(temporal_distance, dtype=np.int64))
    if genetic_distance_arr.size != temporal_distance_arr.size:
        raise ValueError(
            "genetic_distance and temporal_distance must have the same length, "
            f"got {genetic_distance_arr.size} vs {temporal_distance_arr.size}.",
        )

    if genetic_distance_arr.size == 0:
        return np.nan

    is_scalar_input = np.isscalar(genetic_distance) and np.isscalar(temporal_distance)

    if cache_unique_distances and genetic_distance_arr.size > 1:
        pairs = np.column_stack((genetic_distance_arr, temporal_distance_arr))
        unique_pairs, inv = np.unique(pairs, axis=0, return_inverse=True)
        g_u = unique_pairs[:, 0]
        t_u = unique_pairs[:, 1]

        simulation_samples = LinkageMonteCarloSamples.run_simulations(
            transmission_profile,
            clock,
            int(num_simulations),
            max_intermediate_hosts,
        )
        p_temporal_u = LinkageMonteCarloSamples.temporal_kernel(
            temporal_distance_ij=t_u,
            diff_incubation_ij=simulation_samples.diff_incubation_ij,
            generation_interval=simulation_samples.generation_intervals[:, 0],
        )
        genetic_scores_u = LinkageMonteCarloSamples.genetic_kernel(
            genetic_distance_ij=g_u,
            clock_rates=simulation_samples.clock_rates,
            sampling_delay_i=simulation_samples.sampling_delay_i,
            sampling_delay_j=simulation_samples.sampling_delay_j,
            generation_intervals=simulation_samples.generation_intervals,
            max_intermediate_hosts=max_intermediate_hosts,
            diff_infection_ij=simulation_samples.diff_infection_ij,
            incubation_periods=simulation_samples.incubation_periods,
        )

        p_posterior_u = _normalize_genetic_scores(genetic_scores_u)
        cols = _coerce_intermediate_counts(included_intermediate_counts, max_intermediate_hosts)
        selected_u = p_posterior_u[:, cols].sum(axis=1)
        out = (p_temporal_u * selected_u)[inv]
    else:
        simulation_samples = LinkageMonteCarloSamples.run_simulations(
            transmission_profile,
            clock,
            int(num_simulations),
            max_intermediate_hosts,
        )
        p_temporal = LinkageMonteCarloSamples.temporal_kernel(
            temporal_distance_ij=temporal_distance_arr,
            diff_incubation_ij=simulation_samples.diff_incubation_ij,
            generation_interval=simulation_samples.generation_intervals[:, 0],
        )
        genetic_scores_by_scenario = LinkageMonteCarloSamples.genetic_kernel(
            genetic_distance_ij=genetic_distance_arr,
            clock_rates=simulation_samples.clock_rates,
            sampling_delay_i=simulation_samples.sampling_delay_i,
            sampling_delay_j=simulation_samples.sampling_delay_j,
            generation_intervals=simulation_samples.generation_intervals,
            max_intermediate_hosts=max_intermediate_hosts,
            diff_infection_ij=simulation_samples.diff_infection_ij,
            incubation_periods=simulation_samples.incubation_periods,
        )

        p_posterior = _normalize_genetic_scores(genetic_scores_by_scenario)
        cols = _coerce_intermediate_counts(included_intermediate_counts, max_intermediate_hosts)
        selected = p_posterior[:, cols].sum(axis=1)
        out = p_temporal * selected

    if is_scalar_input:
        return float(out[0])
    return out


def estimate_linkage_probability_grid(
    transmission_profile: InfectiousnessToTransmissionTime,
    clock: MolecularClock,
    genetic_distances: np.ndarray,
    temporal_distances: np.ndarray,
    *,
    included_intermediate_counts: tuple[int, ...] = (0,),
    max_intermediate_hosts: int = 10,
    num_simulations: int = 10000,
) -> np.ndarray:
    """Compute linkage probabilities on a genetic-by-temporal grid.

    Parameters
    ----------
    transmission_profile : InfectiousnessToTransmissionTime
        Transmission-time model used for temporal simulation.
    clock : MolecularClock
        Molecular clock used for genetic simulation.
    genetic_distances : numpy.ndarray
        Genetic-distance grid values.
    temporal_distances : numpy.ndarray
        Temporal-distance grid values.
    included_intermediate_counts : tuple of int, default=(0,)
        Intermediate-host counts to include in the final probability.
    max_intermediate_hosts : int, default=10
        Maximum number of intermediate hosts considered in the Monte Carlo
        simulation.
    num_simulations : int, default=10000
        Number of Monte Carlo draws.

    Returns
    -------
    numpy.ndarray
        Grid of linkage probabilities with shape
        ``(len(genetic_distances), len(temporal_distances))``.
    """

    genetic_distances_arr = np.asarray(genetic_distances, dtype=np.int64)
    temporal_distances_arr = np.asarray(temporal_distances, dtype=np.int64)

    g_grid, t_grid = np.meshgrid(genetic_distances_arr, temporal_distances_arr, indexing="ij")
    flat_g = g_grid.ravel()
    flat_t = t_grid.ravel()

    flat_p = estimate_linkage_probability(
        transmission_profile=transmission_profile,
        clock=clock,
        genetic_distance=flat_g,
        temporal_distance=flat_t,
        included_intermediate_counts=included_intermediate_counts,
        max_intermediate_hosts=max_intermediate_hosts,
        num_simulations=num_simulations,
    )
    flat_p = np.asarray(flat_p, dtype=float)
    return flat_p.reshape(g_grid.shape)


def estimate_temporal_linkage_probability(
    temporal_distance: npt.ArrayLike,
    transmission_profile: InfectiousnessToTransmissionTime,
    num_simulations: int = 10000,
) -> np.ndarray:
    """Estimate temporal evidence for one or more temporal distances.

    Parameters
    ----------
    temporal_distance : array_like
        Temporal distances between case pairs.
    transmission_profile : InfectiousnessToTransmissionTime
        Transmission-time model used for temporal simulation.
    num_simulations : int, default=10000
        Number of Monte Carlo draws.

    Returns
    -------
    numpy.ndarray
        Temporal evidence values.
    """

    td = np.atleast_1d(np.asarray(temporal_distance, dtype=np.int64))
    inc_period = transmission_profile.sample_incubation_periods(size=(num_simulations, 2))
    diff_inc = inc_period[:, 0] - inc_period[:, 1]
    gen_interval = transmission_profile.sample_generation_intervals(size=num_simulations)

    return LinkageMonteCarloSamples.temporal_kernel(
        temporal_distance_ij=td,
        diff_incubation_ij=diff_inc,
        generation_interval=gen_interval,
    )


def estimate_genetic_linkage_probability(
    transmission_profile: InfectiousnessToTransmissionTime,
    clock: MolecularClock,
    genetic_distance: npt.ArrayLike,
    *,
    num_simulations: int = 10000,
    max_intermediate_hosts: int = 10,
    included_intermediate_counts: tuple[int, ...] | None = (0,),
    output_mode: str = "relative",
) -> np.ndarray:
    """Estimate genetic evidence for transmission linkage.

    Parameters
    ----------
    transmission_profile : InfectiousnessToTransmissionTime
        Transmission-time model used for temporal simulation.
    clock : MolecularClock
        Molecular clock used for genetic simulation.
    genetic_distance : array_like
        Genetic distances between case pairs.
    num_simulations : int, default=10000
        Number of Monte Carlo draws.
    max_intermediate_hosts : int, default=10
        Maximum number of intermediate hosts considered in the Monte Carlo
        simulation.
    included_intermediate_counts : tuple of int or None, default=(0,)
        Intermediate-host counts to extract. If ``None``, return all scenario
        columns.
    output_mode : {"relative", "raw", "normalized"}, default="relative"
        Output transformation to apply to the scenario weights.

    Returns
    -------
    numpy.ndarray
        Genetic evidence values for the requested output mode.
    """

    genetic_distance_arr = np.atleast_1d(np.asarray(genetic_distance, dtype=np.int64))
    simulation_samples = LinkageMonteCarloSamples.run_simulations(
        transmission_profile,
        clock,
        int(num_simulations),
        max_intermediate_hosts,
    )

    genetic_scores_by_scenario = LinkageMonteCarloSamples.genetic_kernel(
        genetic_distance_ij=genetic_distance_arr,
        clock_rates=simulation_samples.clock_rates,
        sampling_delay_i=simulation_samples.sampling_delay_i,
        sampling_delay_j=simulation_samples.sampling_delay_j,
        generation_intervals=simulation_samples.generation_intervals,
        max_intermediate_hosts=max_intermediate_hosts,
        diff_infection_ij=simulation_samples.diff_infection_ij,
        incubation_periods=simulation_samples.incubation_periods,
    )

    p_posterior = _normalize_genetic_scores(genetic_scores_by_scenario)

    if included_intermediate_counts is not None:
        cols = _coerce_intermediate_counts(included_intermediate_counts, max_intermediate_hosts)

        if output_mode == "relative":
            selected = p_posterior[:, cols].sum(axis=1)
        elif output_mode == "raw":
            selected = genetic_scores_by_scenario[:, cols].sum(axis=1)
        elif output_mode == "normalized":
            selected = p_posterior[:, cols].sum(axis=1)
        else:
            raise ValueError(
                "output_mode must be 'relative', 'raw', or 'normalized', " f"got {output_mode!r}.",
            )
        return selected

    if output_mode == "relative":
        return p_posterior
    if output_mode == "raw":
        return genetic_scores_by_scenario
    if output_mode == "normalized":
        return p_posterior
    raise ValueError(
        "output_mode must be 'relative', 'raw', or 'normalized', " f"got {output_mode!r}."
    )


__all__ = [
    "estimate_genetic_linkage_probability",
    "estimate_linkage_probability",
    "estimate_linkage_probability_grid",
    "estimate_temporal_linkage_probability",
]
