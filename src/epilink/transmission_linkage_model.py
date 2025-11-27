"""
Estimate the probability of a transmission link between two cases.

The core model combines genetic distance (SNPs) and temporal distance (days)
using a mechanistic infectiousness model and Monte Carlo simulation.

Public API
----------

* :func:`estimate_linkage_probability`
* :func:`estimate_linkage_probabilities`
* :func:`pairwise_linkage_probability_matrix`
* :func:`temporal_linkage_probability`
* :func:`genetic_linkage_probability`
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from .infectiousness_profile import TOIT

ArrayLike = npt.ArrayLike
NDArrayFloat = npt.NDArray[np.float64]


# =============================================================================
# 1) Numba kernels (nopython) — fast and allocation-conscious
# =============================================================================


@njit(parallel=True, fastmath=True)
def _temporal_kernel(
        temporal_distance: np.ndarray,  # shape (K,)
        diff_inc: np.ndarray,  # shape (N,)
        generation_interval: np.ndarray,  # shape (N,), direct generation interval
) -> np.ndarray:
    """Monte Carlo estimate of temporal evidence for each sampling interval."""
    K = temporal_distance.shape[0]
    N = diff_inc.shape[0]
    out = np.empty(K, dtype=np.float64)
    for i in prange(K):
        t = temporal_distance[i]
        count = 0
        for j in range(N):
            if abs(t + diff_inc[j]) <= generation_interval[j]:
                count += 1
        out[i] = count / N
    return out


@njit(parallel=True, fastmath=True)
def _poisson_pmf(k: int, lam: float) -> float:
    """Compute Poisson pmf P(K=k | λ=lam) in a Numba-friendly way.

    Uses direct factorial for small k and a Stirling approximation for larger
    k to balance stability and speed. Assumes k >= 0.
    """
    if k < 0 or lam < 0.0:
        return 0.0
    if lam == 0.0:
        return 1.0 if k == 0 else 0.0

    # Small k: compute factorial exactly
    if k < 20:
        fact = 1.0
        for i in range(2, k + 1):
            fact *= float(i)
        return np.exp(-lam) * (lam**k) / fact

    # Large k: Stirling approximation for log(k!)
    kf = float(k)
    log_fact = kf * np.log(kf) - kf + 0.5 * np.log(2.0 * np.pi * kf)
    return np.exp(-lam + kf * np.log(lam) - log_fact)


@njit(parallel=True, fastmath=True)
def _genetic_kernel(
        dists: np.ndarray,  # shape (K,)
        clock_rates: np.ndarray,  # shape (N,)
        psi_sA: np.ndarray,  # shape (N,)
        psi_sB: np.ndarray,  # shape (N,)
        generation_interval: np.ndarray,  # shape (N, M+1)
        toit_difference: np.ndarray,  # shape (N,)
        incubation_period: np.ndarray,  # shape (N, 2)
        caseX_to_caseA: np.ndarray,  # shape (N,)
        intermediates: int,  # M (max number of intermediates)
        deterministic: bool,
        mutation_tolerance: int,
        use_time_space: bool,
) -> np.ndarray:
    """Genetic kernel operating in mutation-count space.

    For each genetic distance d in ``dists`` (interpreted as observed SNP
    counts), compute scenario-wise evidence for m = 0..M using either a
    deterministic or Poisson mutation accumulation model.

    Parameters
    ----------
    dists : np.ndarray
        Observed SNP distances, shape (K,).
    clock_rates : np.ndarray
        Per-day clock rates, shape (N,).
    psi_sA, psi_sB, generation_interval, toit_difference, incubation_period,
    caseX_to_caseA, intermediates :
        Epidemiological quantities as produced by ``_run_simulations``; shapes
        match those expected by ``_genetic_kernel``.
    deterministic : bool
        If True, use deterministic matching in mutation space with a tolerance
        of ``mutation_tolerance`` around the expected count. If False, use a
        Poisson pmf averaged over simulations.
    mutation_tolerance : int
        Integer tolerance used in deterministic mode: a simulation contributes
        if ``|d - round(lambda_jm)| <= mutation_tolerance``.
    use_time_space : bool, default=False
        If True, fall back to time-space kernel and ignores ``mutation_tolerance``.
    """
    K = dists.shape[0]
    N = clock_rates.shape[0]
    M = intermediates

    out = np.zeros((K, M + 1), dtype=np.float64)

    # Invariants across m
    dir_tmrca_exp = psi_sA + psi_sB  # (N,)
    inc_period_sum = incubation_period[:, 0] + incubation_period[:, 1]  # (N,)

    for k in prange(K):
        d = int(dists[k])
        if d < 0:
            # Negative genetic distances are not meaningful; leave zeros.
            continue

        # m = 0: direct
        count_or_sum = 0.0
        for j in range(N):
            tmrca = d / (2.0 * clock_rates[j])
            tmrca_expected = dir_tmrca_exp[j]
            lam = 2.0 * clock_rates[j] * tmrca_expected
            if deterministic and not use_time_space:
                expected_count = int(np.rint(lam))
                if abs(d - expected_count) <= mutation_tolerance:
                    count_or_sum += 1.0
            elif deterministic and use_time_space:
                if abs(tmrca - tmrca_expected) <= generation_interval[j, 0]:
                    count_or_sum += 1.0
            else:
                count_or_sum += _poisson_pmf(d, lam)
        out[k, 0] = count_or_sum / float(N)

        # m > 0
        for m in range(1, M + 1):
            idx = M - (m - 1)
            count_or_sum_m = 0.0

            for j in range(N):
                # Successive path: direct TMRCA + sum GI from idx..M
                s_suc = 0.0
                for col in range(idx, M + 1):
                    s_suc += generation_interval[j, col]
                suc_tmrca = dir_tmrca_exp[j] + s_suc

                # Common ancestor path: TOIT difference + incubation periods + sum GI idx..M-1
                s_com = 0.0
                for col in range(idx, M):
                    s_com += generation_interval[j, col]
                common_tmrca = toit_difference[j] + inc_period_sum[j] + s_com

                # Convert both paths to expected mutation counts and combine
                lam_suc = 2.0 * clock_rates[j] * suc_tmrca
                lam_com = 2.0 * clock_rates[j] * common_tmrca

                if deterministic and not use_time_space:
                    expected_suc = int(np.rint(lam_suc))
                    expected_com = int(np.rint(lam_com))
                    match_suc = abs(d - expected_suc) <= mutation_tolerance
                    match_com = abs(d - expected_com) <= mutation_tolerance
                    # Inclusion-exclusion on events
                    if match_suc or match_com:
                        count_or_sum_m += 1.0

                elif deterministic and use_time_space:
                    match_suc = abs(suc_tmrca - dir_tmrca_exp[j]) <= generation_interval[j, 0]
                    match_com = abs(common_tmrca - caseX_to_caseA[j]) <= caseX_to_caseA[j]
                    if match_suc or match_com:
                        count_or_sum_m += 1.0
                else:
                    # Probabilities for each path
                    p_suc = _poisson_pmf(d, lam_suc)
                    p_com = _poisson_pmf(d, lam_com)
                    # Inclusion-exclusion in probability space
                    p_any = p_suc + p_com - (p_suc * p_com)
                    count_or_sum_m += p_any

            out[k, m] = count_or_sum_m / float(N)

    return out


# =============================================================================
# 2) Simulation
# =============================================================================

class SimulationOutputs(TypedDict):
    """Container describing all arrays produced by :func:`_run_simulations`."""

    incubation_period: NDArrayFloat
    generation_interval: NDArrayFloat
    psi_sA: NDArrayFloat
    psi_sB: NDArrayFloat
    diff_inc: NDArrayFloat
    caseX_to_caseA: NDArrayFloat
    toit_difference: NDArrayFloat
    clock_rates: NDArrayFloat

def _run_simulations(
        toit: TOIT,
        num_simulations: int,
        no_intermediates: int
) -> SimulationOutputs:
    """Draw the epidemiological random variables needed by the kernels once.

    Parameters
    ----------
    toit
        Configured infectiousness profile providing the sampling primitives.
    num_simulations
        Number of Monte Carlo draws (rows in each returned array).
    no_intermediates
        Maximum number of intermediate hosts considered, controls the extra
        columns in ``generation_interval``.

    Returns
    -------
    SimulationOutputs
        Typed dictionary containing all arrays consumed by the temporal and
        genetic kernels.
    """
    N = int(num_simulations)
    M = int(no_intermediates)

    inc_period = toit.sample_incubation(size=(N, 2))  # (N, 2)
    gen_interval = toit.generation_time(size=(N, M + 1))  # (N, M+1)
    toit_values = toit.rvs(size=(N, 2))  # (N, 2)
    latent_period = toit.sample_E(size=N)  # (N,)
    clock_rates = toit.sample_clock_rate_per_day(size=N)  # (N,)

    sim = {
        "incubation_period": inc_period,
        "generation_interval": gen_interval,
        "psi_sA": np.abs(gen_interval[:, 0] - inc_period[:, 0]),
        "psi_sB": inc_period[:, 1],
        "diff_inc": inc_period[:, 0] - inc_period[:, 1],
        "caseX_to_caseA": latent_period + np.minimum(toit_values[:, 0], toit_values[:, 1]),
        "toit_difference": np.abs(toit_values[:, 1] - toit_values[:, 0]),
        "clock_rates": clock_rates,
    }
    return sim


# =============================================================================
# 3) Public API
# =============================================================================


def estimate_linkage_probability(
        toit: TOIT,
        genetic_distance: ArrayLike,  # SNPs; scalar or array
        temporal_distance: ArrayLike,  # days; scalar or array (same length as genetic_distance)
        *,
        intermediate_generations: tuple[int, ...] = (0,),  # which m to include in final mixture
        no_intermediates: int = 10,  # max intermediates M used in simulation/kernel
        num_simulations: int = 10000,
        mutation_model: str = "deterministic",  # "deterministic" | "poisson"
        mutation_tolerance: int = 0,  # number of SNP tolerance for deterministic model
        use_time_space: bool = False
) -> float | np.ndarray:
    """End-to-end estimate of :math:`P(link | g, t)`.

    This combines temporal and genetic evidence and supports two mutation
    accumulation models:

    * ``mutation_model='deterministic'`` (default): matches observed SNP counts
      to expected counts within an integer tolerance.
    * if ``use_time_space=True``, falls back to time-space (TMRCA)
      kernel ignoring mutation counts and tolerance.
    * ``mutation_model='poisson'``: uses a Poisson likelihood for SNP counts
      conditional on expected mutations along the transmission tree.

    The rest of the behaviour mirrors the previous implementation: genetic
    evidence is computed for each number of intermediates ``m = 0..M``, then
    normalized and mixed over the requested ``intermediate_generations`` before
    being multiplied by the temporal evidence.
    """
    # 1) Prepare inputs as 1D arrays of same length
    g = np.atleast_1d(np.asarray(genetic_distance, dtype=float))
    t = np.atleast_1d(np.asarray(temporal_distance, dtype=float))
    if g.size != t.size:
        raise ValueError(
            "genetic_distance and temporal_distance must have the same length, "
            f"got {g.size} vs {t.size}.",
        )

    # Early return for empty input
    if g.size == 0:
        return np.nan

    if mutation_model not in ("deterministic", "poisson"):
        raise ValueError(
            "mutation_model must be 'deterministic' or 'poisson', "
            f"got {model!r}.",
        )
    if mutation_tolerance < 0:
        raise ValueError("mutation_tolerance must be non-negative.")

    M = int(no_intermediates)

    # 2) run simulations
    sim = _run_simulations(toit, int(num_simulations), M)

    # 3) Temporal evidence
    p_t = _temporal_kernel(
        temporal_distance=t,
        diff_inc=sim["diff_inc"],
        generation_interval=sim["generation_interval"][:, 0],
    )  # shape (K,)

    # 4) Genetic evidence: p_m for each m=0..M
    use_deterministic = mutation_model == "deterministic"
    p_m = _genetic_kernel(
        dists=g,
        clock_rates=sim["clock_rates"],
        psi_sA=sim["psi_sA"],
        psi_sB=sim["psi_sB"],
        generation_interval=sim["generation_interval"],
        toit_difference=sim["toit_difference"],
        incubation_period=sim["incubation_period"],
        caseX_to_caseA=sim["caseX_to_caseA"],
        intermediates=M,
        deterministic=use_deterministic,
        mutation_tolerance=int(mutation_tolerance),
        use_time_space=use_time_space,
    )  # shape (K, M+1)

    # 5) Normalize genetic evidence across m
    total = 1.0 - np.prod(1.0 - p_m, axis=1)  # aggregate evidence over scenarios
    with np.errstate(divide="ignore", invalid="ignore"):
        p_rel = np.where(total[:, None] > 0.0, p_m / total[:, None], 0.0)
    row_sums = p_rel.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_norm = np.where(row_sums[:, None] > 0.0, p_rel / row_sums[:, None], 0.0)

    # Select columns m specified by intermediate_generations
    cols = np.array(intermediate_generations, dtype=np.int64)
    if cols.min() < 0 or cols.max() > M:
        raise ValueError(
            f"intermediate_generations must be within [0, {M}], got {intermediate_generations}.",
        )
    selected = p_norm[:, cols].sum(axis=1)  # shape (K,)

    # 6) Combine temporal and genetic evidence
    out = p_t * selected

    # 7) Return scalar if scalar input
    if np.isscalar(genetic_distance) and np.isscalar(temporal_distance):
        return float(out[0])
    return out


def pairwise_linkage_probability_matrix(
        toit: TOIT,
        genetic_distances: np.ndarray,  # 1D
        temporal_distances: np.ndarray,  # 1D
        *,
        intermediate_generations: tuple[int, ...] = (0,),
        no_intermediates: int = 10,
        num_simulations: int = 10000,
        mutation_model: str = "deterministic",
        mutation_tolerance: int = 0,
        use_time_space: bool = False,
) -> np.ndarray:
    """Compute a (G x T) matrix of `P(link | g, t)` over distance grids.

    Parameters are forwarded to :func:`estimate_linkage_probability`, including
    ``mutation_model`` and ``mutation_tolerance``.
    """
    gd = np.asarray(genetic_distances, dtype=float)
    td = np.asarray(temporal_distances, dtype=float)

    g_grid, t_grid = np.meshgrid(gd, td, indexing="ij")
    flat_g = g_grid.ravel()
    flat_t = t_grid.ravel()

    flat_p = estimate_linkage_probability(
        toit=toit,
        genetic_distance=flat_g,
        temporal_distance=flat_t,
        intermediate_generations=intermediate_generations,
        no_intermediates=no_intermediates,
        num_simulations=num_simulations,
        mutation_model=mutation_model,
        mutation_tolerance=mutation_tolerance,
        use_time_space=use_time_space,
    )
    flat_p = np.asarray(flat_p, dtype=float)
    return flat_p.reshape(g_grid.shape)

def estimate_linkage_probabilities(
        toit: TOIT,
        genetic_distance: ArrayLike,
        temporal_distance: ArrayLike,
        *,
        intermediate_generations: tuple[int, ...] = (0,),
        no_intermediates: int = 10,
        num_simulations: int = 10000,
        mutation_model: str = "deterministic",
        mutation_tolerance: int = 0,
        use_time_space: bool = False
) -> np.ndarray:
    """Vectorized helper to compute `P(link | g, t)` for many observations.

    Additional keyword arguments are forwarded to
    :func:`estimate_linkage_probability`, including ``mutation_model`` and
    ``mutation_tolerance``.
    """
    g = np.atleast_1d(np.asarray(genetic_distance, dtype=float))
    t = np.atleast_1d(np.asarray(temporal_distance, dtype=float))
    if g.size != t.size:
        raise ValueError(
            "genetic_distance and temporal_distance must have the same length, "
            f"got {g.size} vs {t.size}.",
        )

    if g.size == 0:
        return np.array([], dtype=float)

    gd_unique, gi = np.unique(g, return_inverse=True)
    td_unique, tj = np.unique(t, return_inverse=True)

    prob_matrix = pairwise_linkage_probability_matrix(
        toit=toit,
        genetic_distances=gd_unique,
        temporal_distances=td_unique,
        intermediate_generations=intermediate_generations,
        no_intermediates=no_intermediates,
        num_simulations=num_simulations,
        mutation_model=mutation_model,
        mutation_tolerance=mutation_tolerance,
        use_time_space=use_time_space,
    )

    expected_shape = (gd_unique.size, td_unique.size)
    if getattr(prob_matrix, "shape", None) != expected_shape:
        raise ValueError(
            "pairwise_linkage_probability_matrix returned shape "
            f"{getattr(prob_matrix, 'shape', None)}, expected {expected_shape}",
        )

    probs = np.asarray(prob_matrix, dtype=float)[gi, tj]
    return probs


def temporal_linkage_probability(
        temporal_distance: ArrayLike,
        toit: TOIT,
        num_simulations: int = 10000,
) -> np.ndarray:
    """
    Monte Carlo estimate of the temporal evidence P_t for one or more sampling-intervals.

    For each temporal distance t_i (days between case sampling dates), this returns
    the probability that the (absolute) adjusted interval falls within a directly
    simulated generation interval:
        P_t(i) = mean_j [ |t_i + (inc_A_j - inc_B_j)| <= GI_j ]

    where inc_A/inc_B are incubation periods for cases A and B, and GI is the
    generation interval, all sampled from the TOIT model.

    Parameters
    ----------
    temporal_distance : ArrayLike
        One or more temporal distances (days). Can be scalar or 1D array-like.
        The sign convention is not enforced; negative values are handled.
    toit : TOIT
        A configured infectiousness/clock model instance used to draw incubation
        and generation-time samples (see infectiousness_profile.TOIT).
    num_simulations : int, default=10000
        Number of Monte Carlo draws.

    Returns
    -------
    np.ndarray
        A 1D array of shape (K,), where K = len(np.atleast_1d(temporal_distance)),
        containing the temporal evidence P_t for each input temporal distance.

    Notes
    -----
    - Reproducibility is controlled by the rng_seed in the provided TOIT instance.
    - This function provides only the temporal component; combine with the genetic
      component (or use estimate_linkage_probability) to obtain `P(link | g, t)`.

    Examples
    --------
    >>> toit = TOIT(rng_seed=123)
    >>> temporal_linkage_probability([0, 5, 10], toit=toit, num_simulations=5000)
    array([0.42..., 0.33..., 0.21...])
    """
    td = np.atleast_1d(np.asarray(temporal_distance, dtype=float))  # ensure 1D array

    inc_period = toit.sample_incubation(size=(num_simulations, 2))  # (N, 2)
    diff_inc = inc_period[:, 0] - inc_period[:, 1]  # (N,)
    gen_interval = toit.generation_time(size=num_simulations)  # (N,)

    return _temporal_kernel(
        temporal_distance=td,
        diff_inc=diff_inc,
        generation_interval=gen_interval,
    )


def genetic_linkage_probability(
        toit: TOIT,
        genetic_distance: ArrayLike,
        *,
        num_simulations: int = 10000,
        no_intermediates: int = 10,
        intermediate_generations: tuple[int, ...] | None = None,
        kind: str = "relative",  # "raw" | "relative" | "normalized"
        mutation_model: str = "deterministic",
        mutation_tolerance: int = 0,
        use_time_space: bool = False,
) -> np.ndarray:
    """Monte Carlo estimate of the genetic evidence across scenarios.
    """
    g = np.atleast_1d(np.asarray(genetic_distance, dtype=float))
    M = int(no_intermediates)

    if mutation_model not in ("deterministic", "poisson"):
        raise ValueError(
            "mutation_model must be 'deterministic' or 'poisson', "
            f"got {model!r}.",
        )
    if mutation_tolerance < 0:
        raise ValueError("mutation_tolerance must be non-negative.")

    sim = _run_simulations(toit, int(num_simulations), M)

    use_deterministic = mutation_model == "deterministic"
    p_m = _genetic_kernel(
        dists=g,
        clock_rates=sim["clock_rates"],
        psi_sA=sim["psi_sA"],
        psi_sB=sim["psi_sB"],
        generation_interval=sim["generation_interval"],
        toit_difference=sim["toit_difference"],
        incubation_period=sim["incubation_period"],
        caseX_to_caseA=sim["caseX_to_caseA"],
        intermediates=M,
        deterministic=use_deterministic,
        mutation_tolerance=int(mutation_tolerance),
        use_time_space=use_time_space,
    )

    total = 1.0 - np.prod(1.0 - p_m, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_rel = np.where(total[:, None] > 0.0, p_m / total[:, None], 0.0)
    row_sums = p_rel.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_norm = np.where(row_sums[:, None] > 0.0, p_rel / row_sums[:, None], 0.0)

    if intermediate_generations is not None:
        cols = np.array(intermediate_generations, dtype=np.int64)
        if cols.min() < 0 or cols.max() > M:
            raise ValueError(
                f"intermediate_generations must be within [0, {M}], got {intermediate_generations}.",
            )

        if kind == "relative":
            selected = p_rel[:, cols].mean(axis=1)
        elif kind == "raw":
            selected = p_m[:, cols].mean(axis=1)
        elif kind == "normalized":
            selected = p_norm[:, cols].sum(axis=1)
        else:
            raise ValueError(
                "kind must be 'relative', 'raw', or 'normalized', "
                f"got {kind!r}.",
            )
        return selected

    if kind == "relative":
        return p_rel
    if kind == "raw":
        return p_m
    if kind == "normalized":
        return p_norm
    raise ValueError(f"kind must be 'relative', 'raw', or 'normalized', got {kind!r}.")
